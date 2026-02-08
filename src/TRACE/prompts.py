"""
Prompts and schemas for TRACE audio evaluation system.
"""

from pydantic import BaseModel


class TraceJudgement(BaseModel):
    """Default response schema for TRACE evaluations."""
    reasoning: str
    content: str
    voice_quality: str
    paralinguistics: str


TRACE_EVALUATION_PROMPT = """You are an evaluator of audio outputs produced by different audio-capable large language models. Your task is to compare two audio responses (Audio 1 and Audio 2) generated according to a user's instruction. Evaluate based on these criteria:

1. Content
- Does the content fulfill the user's request accurately?
- Did the content of the response appropriately address the user's instruction?

2. Voice Quality
- How good is the voice quality of the response?
- Does it sound natural/human, does it mispronounce words, does it have pops or echoes?

3. Paralinguistics:
- Does the response correctly perceive emotion from user's tone of voice, does it correctly express emotion through tone of voice, does it correctly follow paralinguistic instructions?
- This includes both implicit audio instruction like emotional intelligence and explicit audio instruction following

Avoid position bias and don't let response length influence your evaluation. After your analysis, output valid JSON with exactly 4 keys:
- "reasoning": your explanation of the comparison along each dimension
- "content": your rating for content dimension. a string value '1' if the first audio is better, '2' if the second audio is better, 'both_bad' if they are equally bad, or 'both_good' if they are equally good
- "voice_quality": your rating for voice quality dimension. a string value '1' if the first audio is better, '2' if the second audio is better, 'both_bad' if they are equally bad, or 'both_good' if they are equally good
- "paralinguistics": your rating for the paralinguistics dimension. a string value '1' if the first audio is better, '2' if the second audio is better, 'both_bad' if they are equally bad, or 'both_good' if they are equally good

You should only pick a winner along each dimension if there is a clear and obvious difference between the quality of the two responses. If it comes down to minor details, then you should opt for using 'both_bad' or 'both_good' instead.

The responses audios (Audio 1 and Audio 2) will be given to you as JSON objects with the following information:

{{
  "agent_response": a transcription of the response,
  "agent_emotion": a vector of emotion scores for the agent's response from the emotion2vec model,
  "agent_accent": a vector of cosine similarity scores for the agent's accent,
  "agent_audio_quality": {{
    "DNSMOS_Personalized_Signal_Quality": signal quality score from DNSMOS model (1-5, higher is better),
    "DNSMOS_Personalized_Background_Quality": background noise quality score from DNSMOS model (1-5, higher is better),
    "DNSMOS_Personalized_Overall_Quality": overall naturalness and audio quality score from DNSMOS model (1-5, higher is better),
    "P808_Overall_Quality": overall naturalness and audio quality score from P.808 recommendation standard (1-5, higher is better)
  }},
  "agent_audio_properties": {{
    "Mean_Pitch_Hz": mean pitch (fundamental frequency) of agent response,
    "Std_Dev_Pitch_Hz": standard deviation in pitch,
    "Full_Pitch_Contour_Hz": full pitch contour,
    "Integrated_Loudness_LUFS": average loudness of the agent response measured in LUFS,
    "Std_Dev_Loudness_LUFS": standard deviation in loudness,
    "Full_Loudness_Contour_LUFS": full loudness contour,
    "Speech_Rate_WPM": speech rate in words per minute,
    "Articulation_Rate_WPM": speech rate in words per minute excluding pauses and gaps in speech
  }}
}}

Here is the user's input prompt:

{user_prompt}

Here is Audio 1 response JSON:

{model_a}

Here is Audio 2 response JSON:

{model_b}

Respond ONLY in text and output valid JSON with keys "reasoning", "content", "voice_quality", and "paralinguistics"."""