<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
Hearing Between the Lines: Unlocking the Reasoning Power of LLMs for Speech Evaluation
</h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://arjunchandra2.github.io/" target="_blank" style="text-decoration: none;">Arjun Chandra<sup>1,*</sup></a>&nbsp;,&nbsp;
    <a href="https://www.linkedin.com/in/kevin-miller-321451133/" target="_blank" style="text-decoration: none;">Kevin Miller<sup>1,*</sup></a>&nbsp;,&nbsp;
    <a href="https://www.amazon.science/author/venkatesh-ravichandran" target="_blank" style="text-decoration: none;">Venkatesh Ravichandran<sup>2</sup></a><br>
    <a href="https://www.amazon.science/author/constantinos-papayiannis" target="_blank" style="text-decoration: none;">Constantinos Papayiannis<sup>2</sup></a>&nbsp;,&nbsp;
    <a href="https://venkatesh-saligrama.github.io/" target="_blank" style="text-decoration: none;">Venkatesh Saligrama<sup>1</sup></a><br>
<sup>*</sup>Equal Contribution<br>
Boston University<sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;Amazon AGI<sup>2</sup><br>
</p>

<p align='center';>
<b>
<em>EACL 2026 Findings</em> <br>
</b>
</p>

<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://arxiv.org/abs/2601.13742" target="_blank" style="text-decoration: none;">arXiv</a>&nbsp;
</b>
</p>

------------
We propose TRACE (Textual Reasoning over Audio Cues for Evaluation), a novel framework that enables LLM judges to reason over audio cues to achieve cost-efficient and human-aligned S2S evaluation. 

<p align="center">
  <img src="assets/overview.png" alt="TRACE overview" width="900"/>
</p>


## Data
**Data coming soon!*

## Usage

### Quick Start
```python
from trace import TRACE

# Initialize TRACE with your API keys and model cache directory
trace = TRACE(
    openai_api_key="your-openai-key",      # Or set OPENAI_API_KEY env variable
    google_api_key="your-google-key",      # Or set GOOGLE_API_KEY env variable
    model_cache_path="./model_cache"       # Directory for caching feature extraction models
)

# Evaluate two audio responses
result = trace.judge_audio(
    audio1_path="system_a_response.wav",       # Path to first audio response
    audio2_path="system_b_response.wav",       # Path to second audio response
    instruction_path="user_instruction.wav",   # Path to the original instruction audio
    model="gpt-4o",                            # LLM model to use for judging
    llm_provider="openai",                     # "openai" or "google"
    fusion_rule="speakbench",                  # Fusion rule: "speakbench" or "s2sarena"
    save_extractions=True,                     # Save extracted audio features to JSON
    extraction_dir="./extractions"             # Directory to save feature JSONs (optional)
)

# Print results
print(f"Winner: {result['fused_rating']}")
print(f"\nDimension ratings:")
print(f"  Content: {result['dimension_ratings']['content']}")
print(f"  Voice Quality: {result['dimension_ratings']['voice_quality']}")
print(f"  Paralinguistics: {result['dimension_ratings']['paralinguistics']}")
print(f"\nReasoning: {result['reasoning']}")
```

### Using Google Gemini
```python
# Use Gemini with default schema
result = trace.judge_audio(
    audio1_path="system_a_response.wav",
    audio2_path="system_b_response.wav",
    instruction_path="user_instruction.wav",
    model="gemini-2.5-flash",                 # Gemini model
    llm_provider="google",                    # Use Google provider
    fusion_rule="speakbench"
)

# Use Gemini with custom response schema
from pydantic import BaseModel

class CustomJudgement(BaseModel):
    reasoning: str
    content: str
    voice_quality: str
    paralinguistics: str

result = trace.judge_audio(
    audio1_path="system_a_response.wav",
    audio2_path="system_b_response.wav",
    instruction_path="user_instruction.wav",
    model="gemini-2.5-flash",
    llm_provider="google",
    fusion_rule="speakbench",
    response_schema=CustomJudgement              # Custom Pydantic schema
)
```

### Result Dictionary

The `judge_audio()` method returns a comprehensive dictionary with the following structure:
```python
{
    # Input metadata
    "audio1_path": str,              # Path to first audio file
    "audio2_path": str,              # Path to second audio file
    "instruction_path": str,         # Path to instruction audio file
    
    # Model configuration
    "model": str,                    # LLM model used (e.g., "gpt-4o")
    "llm_provider": str,             # Provider used ("openai" or "google")
    "fusion_rule": str,              # Fusion rule applied ("speakbench" or "s2sarena")
    
    # Extracted features (if save_extractions=True)
    "extraction_paths": {
        "instruction": str,          # Path to instruction feature JSON
        "audio1": str,               # Path to audio1 feature JSON
        "audio2": str                # Path to audio2 feature JSON
    },
    
    # LLM judgement results
    "dimension_ratings": {
        "content": str,              # Rating: "1", "2", "both_good", or "both_bad"
        "voice_quality": str,        # Rating: "1", "2", "both_good", or "both_bad"
        "paralinguistics": str       # Rating: "1", "2", "both_good", or "both_bad"
    },
    
    "reasoning": str,                # LLM's explanation of the comparison
    
    # Final result
    "fused_rating": str,             # Fused rating: "1", "2", "both_good", or "both_bad"
    
    # Error handling
    "error": str or None             # Error message if evaluation failed, None otherwise
}
```

### Fusion Rules

TRACE supports two fusion rules for combining dimension-wise ratings:

**SpeakBench Fusion** (`fusion_rule="speakbench"`):
- Hierarchy: Content > Paralinguistics > Voice Quality
- Includes "Dominant both_bad" rule

**S2SArena Fusion** (`fusion_rule="s2sarena"`):
- Hierarchy: Content > Paralinguistics > Voice Quality
- Uses acceptability capping with RatingMin operator
- Handles typed ties with acceptability constraints

### Extracted Audio Features

When `save_extractions=True`, TRACE saves detailed audio features for each file:
```json
{
  "agent_response": "Transcribed text...",
  "agent_emotion": {
    "angry": 0.023,
    "disgusted": 0.012,
    "fearful": 0.008,
    "happy": 0.456,
    "neutral": 0.234,
    "other": 0.067,
    "sad": 0.045,
    "surprised": 0.123,
    "unknown": 0.032
  },
  "agent_accent": {
    "england": 0.234,
    "us": 0.789,
    "canada": 0.156,
    ...
  },
  "agent_audio_quality": {
    "DNSMOS_Personalized_Signal_Quality": "4.23 / 5.00",
    "DNSMOS_Personalized_Background_Quality": "4.56 / 5.00",
    "DNSMOS_Personalized_Overall_Quality": "4.12 / 5.00",
    "P808_Overall_Quality": "4.34 / 5.00"
  },
  "agent_audio_properties": {
    "Mean_Pitch_Hz": 187.45,
    "Std_Dev_Pitch_Hz": 34.23,
    "Full_Pitch_Contour_Hz": [180.2, 185.3, ...],
    "Integrated_Loudness_LUFS": -23.45,
    "Std_Dev_Loudness_LUFS": 2.34,
    "Full_Loudness_Contour_LUFS": [-24.5, -23.2, ...],
    "Speech_Rate_WPM": 145.67,
    "Articulation_Rate_WPM": 178.23
  }
}
```


## Citation

If you use TRACE in your research, please cite our paper:

```bibtex
@misc{chandra2026hearinglinesunlockingreasoning,
      title={Hearing Between the Lines: Unlocking the Reasoning Power of LLMs for Speech Evaluation}, 
      author={Arjun Chandra and Kevin Miller and Venkatesh Ravichandran and Constantinos Papayiannis and Venkatesh Saligrama},
      year={2026},
      eprint={2601.13742},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.13742}, 
}
```




