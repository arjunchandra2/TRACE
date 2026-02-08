import os
import sys
from pathlib import Path
import json

from extraction import SignalExtraction
from llm import OpenAILLM, GoogleLLM
from fusion import Fusion
from prompts import TRACE_EVALUATION_PROMPT, TraceJudgement


class TRACE: 
    """
    TRACE: A scalable and interpretable interface for automatic speech-to-speech evaluation  
    """

    def __init__(self, openai_api_key=None, google_api_key=None, model_cache_path=None):        
        # Initialize extraction module
        self.extractor = SignalExtraction(cache_path=model_cache_path)

        # Initialize LLM module 
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            self.openai_llm = OpenAILLM(api_key=api_key)
        else:
            self.openai_llm = None

        api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self.google_llm = GoogleLLM(api_key=api_key)
        else:
            self.google_llm = None

        # Initialize fusion module 
        self.fusion = Fusion()

    def judge_audio(
        self,
        audio1_path: str,
        audio2_path: str,
        instruction_path: str,
        model: str = "gpt-4o",
        llm_provider: str = "openai",
        fusion_rule: str = "speakbench",
        save_extractions: bool = False,
        extraction_dir: str = None,
        response_schema = None
    ) -> dict:
        """
        Judge two audio responses against an instruction.
        
        Args:
            audio1_path: Path to first audio response
            audio2_path: Path to second audio response
            instruction_path: Path to instruction audio
            model: LLM model name to use for judging
            llm_provider: "openai" or "google"
            fusion_rule: Fusion rule to apply ("speakbench" or "s2sarena")
            save_extractions: Whether to save extracted features as JSON
            extraction_dir: Directory to save extractions (defaults to same dir as audio files)
            response_schema: Optional Pydantic BaseModel for Gemini structured output 
                           (defaults to TraceJudgement if None and using Google provider)
            
        Returns:
            dict: Complete evaluation results with ratings, reasoning, and metadata
        """
        result = {
            "audio1_path": audio1_path,
            "audio2_path": audio2_path,
            "instruction_path": instruction_path,
            "model": model,
            "llm_provider": llm_provider,
            "fusion_rule": fusion_rule,
            "error": None,
            "extraction_paths": {},
            "dimension_ratings": {},
            "reasoning": None,
            "fused_rating": None
        }
        
        try:
            # Step 1: Extract features from all audio files
            print("Extracting features from instruction audio...")
            instruction_data = self.extractor.analyze_audio(instruction_path)
            
            print("Extracting features from audio 1...")
            audio1_data = self.extractor.analyze_audio(audio1_path)
            
            print("Extracting features from audio 2...")
            audio2_data = self.extractor.analyze_audio(audio2_path)
            
            # Step 2: Save extractions if requested
            if save_extractions:
                if extraction_dir is None:
                    extraction_dir = str(Path(audio1_path).parent)
                
                extraction_dir = Path(extraction_dir)
                extraction_dir.mkdir(parents=True, exist_ok=True)
                
                # Save instruction extraction
                instruction_json_path = extraction_dir / f"{Path(instruction_path).stem}_extraction.json"
                with open(instruction_json_path, 'w') as f:
                    json.dump(instruction_data, f, indent=2)
                result["extraction_paths"]["instruction"] = str(instruction_json_path)
                
                # Save audio 1 extraction
                audio1_json_path = extraction_dir / f"{Path(audio1_path).stem}_extraction.json"
                with open(audio1_json_path, 'w') as f:
                    json.dump(audio1_data, f, indent=2)
                result["extraction_paths"]["audio1"] = str(audio1_json_path)
                
                # Save audio 2 extraction
                audio2_json_path = extraction_dir / f"{Path(audio2_path).stem}_extraction.json"
                with open(audio2_json_path, 'w') as f:
                    json.dump(audio2_data, f, indent=2)
                result["extraction_paths"]["audio2"] = str(audio2_json_path)
            
            # Step 3: Construct prompt with extracted features
            user_prompt = instruction_data["agent_response"]
            model_a_json = json.dumps(audio1_data, indent=2)
            model_b_json = json.dumps(audio2_data, indent=2)
            
            full_prompt = TRACE_EVALUATION_PROMPT.format(
                user_prompt=user_prompt,
                model_a=model_a_json,
                model_b=model_b_json
            )
            
            # Step 4: Get LLM judgment
            print(f"Getting judgment from {llm_provider} model {model}...")
            if llm_provider == "openai":
                if self.openai_llm is None:
                    raise ValueError("OpenAI API key not provided")
                llm_response = self.openai_llm.run(full_prompt, model_name=model, is_json=True)
            elif llm_provider == "google":
                if self.google_llm is None:
                    raise ValueError("Google API key not provided")
                
                # Use default schema if none provided
                schema = response_schema if response_schema is not None else TraceJudgement
                llm_response = self.google_llm.run(
                    full_prompt, 
                    model_name=model, 
                    is_json=True,
                    response_schema=schema
                )
            else:
                raise ValueError(f"Unknown LLM provider: {llm_provider}")
            
            # Step 5: Extract dimension ratings
            result["reasoning"] = llm_response.get("reasoning", "")
            result["dimension_ratings"] = {
                "content": llm_response.get("content", "nan"),
                "voice_quality": llm_response.get("voice_quality", "nan"),
                "paralinguistics": llm_response.get("paralinguistics", "nan")
            }
            
            # Step 6: Fuse ratings
            print(f"Fusing ratings with {fusion_rule} rule...")
            fused_rating = self.fusion.fuse(
                rule=fusion_rule,
                content=result["dimension_ratings"]["content"],
                para=result["dimension_ratings"]["paralinguistics"],
                vq=result["dimension_ratings"]["voice_quality"]
            )
            result["fused_rating"] = fused_rating
            
            print(f"Evaluation complete! Fused rating: {fused_rating}")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        return result


if __name__ == "__main__":
    
    # Example usage
    trace = TRACE(model_cache_path="/projectnb/ivc-ml/ac25/test")
    
    # Example judgment with OpenAI
    result = trace.judge_audio(
        audio1_path="/projectnb/ivc-ml/ac25/test/unit_test/audio_a.wav",
        audio2_path="/projectnb/ivc-ml/ac25/test/unit_test/audio_b.wav",
        instruction_path="/projectnb/ivc-ml/ac25/test/unit_test/instruction.wav",
        model="gpt-4o",
        llm_provider="openai",
        fusion_rule="speakbench"
    )
    
    # Example with Google and default schema
    # result = trace.judge_audio(
    #     audio1_path="/projectnb/ivc-ml/ac25/test/unit_test/audio_a.wav",
    #     audio2_path="/projectnb/ivc-ml/ac25/test/unit_test/audio_b.wav",
    #     instruction_path="/projectnb/ivc-ml/ac25/test/unit_test/instruction.wav",
    #     model="gemini-2.5-flash",
    #     llm_provider="google",
    #     fusion_rule="speakbench"
    # )
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nDimension Ratings:")
    for dim, rating in result["dimension_ratings"].items():
        print(f"  {dim}: {rating}")
    print(f"\nFused Rating: {result['fused_rating']}")
    print(f"\nReasoning:\n{result['reasoning']}")


    if result["error"]:
        print(f"\nError: {result['error']}")