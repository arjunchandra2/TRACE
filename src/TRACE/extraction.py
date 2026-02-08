import json
import numpy as np
import librosa
import aubio
import soundfile as sf
import pyloudnorm as pyln
import torch
import torchaudio
import os
from pathlib import Path
import warnings

# Model imports
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from speechbrain.inference import EncoderClassifier
from funasr import AutoModel

# ==================== MONKEY PATCH FOR PYLOUDNORM ====================
from pyloudnorm import util

def patched_integrated_loudness(self, data, return_contour=True):
    """ Measure the integrated gated loudness of a signal.
    
    Uses the weighting filters and block size defined by the meter
    the integrated loudness is measured based upon the gating algorithm
    defined in the ITU-R BS.1770-4 specification. 

    Input data must have shape (samples, ch) or (samples,) for mono audio.
    Supports up to 5 channels and follows the channel ordering: 
    [Left, Right, Center, Left surround, Right surround]

    Params
    -------
    data : ndarray
        Input multichannel audio data.
    return_contour : bool
        Whether to return the filtered loudness contour

    Returns
    -------
    LUFS : float
        Integrated gated loudness of the input measured in dB LUFS.
    filtered_contour : list (optional)
        List of loudness values for blocks passing both gates
    """
    input_data = data.copy()
    util.valid_audio(input_data, self.rate, self.block_size)

    if input_data.ndim == 1:
        input_data = np.reshape(input_data, (input_data.shape[0], 1))

    numChannels = input_data.shape[1]        
    numSamples  = input_data.shape[0]

    # Apply frequency weighting filters - account for the acoustic response of the head and auditory system
    for (filter_class, filter_stage) in self._filters.items():
        for ch in range(numChannels):
            input_data[:,ch] = filter_stage.apply_filter(input_data[:,ch])

    G = [1.0, 1.0, 1.0, 1.41, 1.41]  # channel gains
    T_g = self.block_size  # 400 ms gating block standard
    Gamma_a = -70.0  # -70 LKFS = absolute loudness threshold
    overlap = 0.75  # overlap of 75% of the block duration
    step = 1.0 - overlap  # step size by percentage

    T = numSamples / self.rate  # length of the input in seconds
    numBlocks = int(np.round(((T - T_g) / (T_g * step)))+1)  # total number of gated blocks
    j_range = np.arange(0, numBlocks)  # indexed list of total blocks
    z = np.zeros(shape=(numChannels,numBlocks))  # instantiate array - transpose of input

    for i in range(numChannels):  # iterate over input channels
        for j in j_range:  # iterate over total frames
            l = int(T_g * (j * step    ) * self.rate)  # lower bound of integration (in samples)
            u = int(T_g * (j * step + 1) * self.rate)  # upper bound of integration (in samples)
            # calculate mean square of the filtered for each block (see eq. 1)
            z[i,j] = (1.0 / (T_g * self.rate)) * np.sum(np.square(input_data[l:u,i]))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # loudness for each jth block (see eq. 4)
        l = [-0.691 + 10.0 * np.log10(np.sum([G[i] * z[i,j] for i in range(numChannels)])) for j in j_range]
    
    # Store for compatibility with other methods like loudness_range
    self.blockwise_loudness = l
    
    # find gating block indices above absolute threshold
    J_g = [j for j,l_j in enumerate(l) if l_j >= Gamma_a]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # calculate the average of z[i,j] as shown in eq. 5
        z_avg_gated = [np.mean([z[i,j] for j in J_g]) for i in range(numChannels)]
    
    # calculate the relative threshold value (see eq. 6)
    Gamma_r = -0.691 + 10.0 * np.log10(np.sum([G[i] * z_avg_gated[i] for i in range(numChannels)])) - 10.0

    # find gating block indices above relative and absolute thresholds (end of eq. 7)
    J_g = [j for j,l_j in enumerate(l) if (l_j > Gamma_r and l_j > Gamma_a)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # calculate the average of z[i,j] as shown in eq. 7 with blocks above both thresholds
        z_avg_gated = np.nan_to_num(np.array([np.mean([z[i,j] for j in J_g]) for i in range(numChannels)]))
    
    # calculate final loudness gated loudness (see eq. 7)
    with np.errstate(divide='ignore'):
        LUFS = -0.691 + 10.0 * np.log10(np.sum([G[i] * z_avg_gated[i] for i in range(numChannels)]))

    filtered_contour = [loudness for index, loudness in enumerate(l) if index in J_g]

    if return_contour:
        return LUFS, filtered_contour
        
    return LUFS

# Apply the monkey patch
pyln.Meter.integrated_loudness = patched_integrated_loudness
# ==================== END MONKEY PATCH ====================


class SignalExtraction:
    """
    Unified audio analysis class that extracts:
    - Transcription (Whisper)
    - Audio properties (pitch, loudness, speech rate)
    - Emotion scores (emotion2vec)
    - Accent scores (SpeechBrain)
    - Audio quality (DNSMOS)
    """
    
    def __init__(self, cache_path, dnsmos_model_dir="DNSMOS/models"):
        """
        Initialize all models.
        
        Args:
            cache_path: Path for HuggingFace model cache
            dnsmos_model_dir: Directory containing DNSMOS ONNX models
        """
        self.cache_path = Path(cache_path)
        self.dnsmos_model_dir = Path(dnsmos_model_dir)
        
        print("Initializing feeature extraction models...")
        self._init_whisper()
        self._init_accent_classifier()
        self._init_emotion_model()
        self._init_dnsmos()
        print("All models loaded successfully!")
    
    def _init_whisper(self):
        """Initialize Whisper ASR model"""
        print("Loading Whisper model...")
        self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3",
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=str(self.cache_path)
        )
        
        self.whisper_processor = AutoProcessor.from_pretrained(
            "openai/whisper-large-v3",
            cache_dir=str(self.cache_path)
        )
        
        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
            tokenizer=self.whisper_processor.tokenizer,
            feature_extractor=self.whisper_processor.feature_extractor,
            torch_dtype="auto",
            device_map="auto",
        )
    
    def _init_accent_classifier(self):
        """Initialize SpeechBrain accent classifier"""
        print("Loading accent classifier...")
        self.accent_classifier = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            savedir=str(self.cache_path / "models--sb-accent")
        )
        
        self.accent_labels = [
            'england', 'us', 'canada', 'australia', 'indian', 'scotland', 'ireland',
            'african', 'malaysia', 'newzealand', 'southatlandtic', 'bermuda',
            'philippines', 'hongkong', 'wales', 'singapore'
        ]
    
    def _init_emotion_model(self):
        """Initialize emotion2vec model"""
        print("Loading emotion model...")

        os.environ['MODELSCOPE_CACHE'] = str(self.cache_path / "modelscope")
         
        self.emotion_model = AutoModel(
                model="iic/emotion2vec_plus_large",
                hub="ms",  # ModelScope
                device="cuda" if torch.cuda.is_available() else "cpu",
                disable_update=True,
        )
    
        self.emotion_keys = [
            "angry", "disgusted", "fearful", "happy", "neutral", 
            "other", "sad", "surprised", "unknown"
    ]
    
    def _init_dnsmos(self):
        """Initialize DNSMOS scorer"""
        print("Loading DNSMOS models...")
        # Import locally to avoid issues if not available
        from DNSMOS.dnsmos_single import ComputeScore, SAMPLING_RATE
        
        self.dnsmos_scorer = ComputeScore(
            model_path=str(self.dnsmos_model_dir / "sig_bak_ovr.onnx"),
            p_model_path=str(self.dnsmos_model_dir / "p_sig_bak_ovr.onnx"),
            p808_model_path=str(self.dnsmos_model_dir / "model_v8.onnx")
        )
        self.dnsmos_sampling_rate = SAMPLING_RATE
    
    # ==================== ASR ====================
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using Whisper with word-level timestamps.
        
        Returns:
            transcript (str): Full transcription text
            word_chunks (list): List of (word, start_time, end_time) tuples
        """
        transcription = self.whisper_pipe(
            audio_path,
            return_timestamps="word",
            generate_kwargs={"language": "english"}
        )
        
        transcript = transcription["text"]
        word_chunks = [
            (chunk["text"].strip(), chunk["timestamp"][0], chunk["timestamp"][1])
            for chunk in transcription["chunks"]
        ]
        
        return transcript, word_chunks
    
    # ==================== Audio Properties ====================
    
    def _pitch_stats_aubio(self, audio_path, samplerate=24000, hop_size=512, 
                          min_pitch=80.0, max_pitch=300.0, contour_length=20):
        """Extract pitch statistics using Aubio"""
        # Load audio
        y, _ = librosa.load(audio_path, sr=samplerate)
        
        # Setup Aubio pitch detector
        win_s = 2048
        pitch_o = aubio.pitch("yin", win_s, hop_size, samplerate)
        pitch_o.set_unit("Hz")
        pitch_o.set_silence(-40)
        
        # Extract pitches
        pitches = []
        for i in range(0, len(y) - hop_size, hop_size):
            frame = y[i:i+hop_size].astype(np.float32)
            pitch_val = pitch_o(frame)[0]
            confidence = pitch_o.get_confidence()
            if np.isnan(pitch_val) or confidence < 0.8:
                continue
            if min_pitch <= pitch_val <= max_pitch:
                pitches.append(pitch_val)
        
        pitches = np.array(pitches)
        
        # Compute stats and contour
        if len(pitches) == 0:
            return float('nan'), float('nan'), np.full(contour_length, np.nan)
        
        mean_pitch = np.mean(pitches)
        std_pitch = np.std(pitches)
        
        # Interpolated contour
        x_orig = np.linspace(0, 1, len(pitches))
        x_new = np.linspace(0, 1, contour_length)
        contour = np.interp(x_new, x_orig, pitches)
        
        return mean_pitch, std_pitch, contour
    
    def _loudness_stats_lufs(self, audio_path, contour_length=20):
        """Compute LUFS loudness statistics"""
        data, rate = sf.read(audio_path)
        meter = pyln.Meter(rate)
        loudness, loudness_contour = meter.integrated_loudness(data, return_contour=True)
        
        loudness_stdev = np.std(loudness_contour)
        
        # Interpolated contour
        x_orig = np.linspace(0, 1, len(loudness_contour))
        x_new = np.linspace(0, 1, contour_length)
        contour_interp = np.interp(x_new, x_orig, loudness_contour)
        
        return loudness, loudness_stdev, contour_interp
    
    def _calculate_speech_rate(self, word_chunks, audio_path):
        """Calculate speech rate (WPM) including pauses"""
        num_words = len(word_chunks)
        duration = librosa.get_duration(path=audio_path)
        wpm = (num_words / duration) * 60
        return round(wpm, 2)
    
    def _calculate_articulation_rate(self, word_chunks):
        """Calculate articulation rate (WPM) excluding pauses"""
        num_words = len(word_chunks)
        speech_duration = sum(end - start for _, start, end in word_chunks)
        articulation_rate = (num_words / speech_duration) * 60
        return round(articulation_rate, 2)
    
    def extract_audio_properties(self, audio_path, word_chunks):
        """
        Extract comprehensive audio properties.
        
        Args:
            audio_path: Path to audio file
            word_chunks: List of (word, start, end) tuples from transcription
            
        Returns:
            dict: Audio properties including pitch, loudness, and speech rates
        """
        mean_pitch, std_pitch, pitch_contour = self._pitch_stats_aubio(audio_path)
        integrated_loudness, std_loudness, loudness_contour = self._loudness_stats_lufs(audio_path)
        speech_rate = self._calculate_speech_rate(word_chunks, audio_path)
        articulation_rate = self._calculate_articulation_rate(word_chunks)
        
        return {
            "Mean_Pitch_Hz": round(float(mean_pitch), 2),
            "Std_Dev_Pitch_Hz": round(float(std_pitch), 2),
            "Full_Pitch_Contour_Hz": np.round(pitch_contour, 2).tolist(),
            "Integrated_Loudness_LUFS": round(float(integrated_loudness), 2),
            "Std_Dev_Loudness_LUFS": round(float(std_loudness), 2),
            "Full_Loudness_Contour_LUFS": np.round(loudness_contour, 2).tolist(),
            "Speech_Rate_WPM": speech_rate,
            "Articulation_Rate_WPM": articulation_rate,
        }
    
    # ==================== Emotion ====================
    
    def extract_emotion(self, audio_path):
        """
        Extract emotion scores using emotion2vec.
        
        Returns:
            dict: Emotion probabilities for 9 emotion categories
        """
        rec_result = self.emotion_model.generate(
            audio_path,
            granularity="utterance",
            extract_embedding=False,
        )
        
        scores = rec_result[0]['scores']
        return {k: round(s, 3) for k, s in zip(self.emotion_keys, scores)}
    
    # ==================== Accent ====================
    
    def extract_accent(self, audio_path):
        """
        Extract accent similarity scores.
        
        Returns:
            dict: Cosine similarity scores for 16 accent categories
        """
        cos_sim_scores, score, index, text_lab = self.accent_classifier.classify_file(audio_path)
        return {
            label: round(float(cos_sim_scores[0][i]), 3) 
            for i, label in enumerate(self.accent_labels)
        }
    
    # ==================== Audio Quality ====================
    
    def _dnsmos_score(self, audio_path):
        """Compute DNSMOS quality scores"""
        dnsmos_scores = self.dnsmos_scorer(audio_path, sampling_rate=self.dnsmos_sampling_rate)
        
        return {
            "DNSMOS_Personalized_Signal_Quality": f"{dnsmos_scores['P_SIG']:.2f} / 5.00",
            "DNSMOS_Personalized_Background_Quality": f"{dnsmos_scores['P_BAK']:.2f} / 5.00",
            "DNSMOS_Personalized_Overall_Quality": f"{dnsmos_scores['P_OVRL']:.2f} / 5.00",
            "P808_Overall_Quality": f"{dnsmos_scores['P808_MOS']:.2f} / 5.00",
        }
    
    def extract_quality_scores(self, audio_path, transcription):
        """
        Extract audio quality metrics.
        
        Args:
            audio_path: Path to audio file
            transcription: Transcribed text from ASR
            
        Returns:
            dict: Quality scores including DNSMOS and WER
        """
        dnsmos_scores = self._dnsmos_score(audio_path)
        
        return {
            **dnsmos_scores,
        }
    
    # ==================== Full Analysis Pipeline ====================
    
    def analyze_audio(self, audio_path):
        """
        Run complete analysis pipeline on an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Complete analysis results including all metrics
        """
        # Transcription
        transcription, word_chunks = self.transcribe(audio_path)
        
        # Extract all features
        audio_props = self.extract_audio_properties(audio_path, word_chunks)
        emotion_scores = self.extract_emotion(audio_path)
        accent_scores = self.extract_accent(audio_path)
        quality_scores = self.extract_quality_scores(audio_path, transcription)

        
        return {
            "agent_response": transcription,
            "agent_emotion": emotion_scores,
            "agent_accent": accent_scores,
            "agent_audio_quality": quality_scores,
            "agent_audio_properties": audio_props,
        }
    
    def save_analysis(self, audio_path, output_path=None):
        """
        Analyze audio and save results to JSON.
        
        Args:
            audio_path: Path to audio file
            output_path: Path for output JSON (defaults to audio_path with .json extension)
        """
        if output_path is None:
            output_path = Path(audio_path).with_suffix('.json')
        
        results = self.analyze_audio(audio_path)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return output_path