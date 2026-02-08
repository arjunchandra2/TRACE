import json
from openai import OpenAI
from google import genai
from pydantic import BaseModel


class OpenAILLM:
    """Model-agnostic OpenAI LLM client."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def _extract_json(self, response: str) -> dict:
        """Extract JSON from response text."""
        start, end = response.find('{'), response.rfind('}')
        assert start >= 0 and end >= 0, f"Could not find JSON in response: {response}"
        json_str = response[start:end + 1]
        return json.loads(json_str)
    
    def run(self, query: str, model_name: str = "gpt-4o", is_json: bool = False) -> str | dict:
        """
        Run a completion with the specified model.
        
        Args:
            query: The prompt/query to send
            model_name: The OpenAI model to use (e.g., "gpt-4o", "gpt-4o-mini")
            is_json: Whether to parse the response as JSON
            
        Returns:
            str or dict: The model's response, optionally parsed as JSON
        """
        messages = [{"role": "system", "content": query}]
        
        completion_kwargs = {
            "model": model_name,
            "messages": messages,
        }
        
        if is_json:
            completion_kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**completion_kwargs)
        response_text = response.choices[0].message.content
        
        if is_json:
            return self._extract_json(response_text)
        else:
            return response_text


class GoogleLLM:
    """Model-agnostic Google Gemini LLM client."""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
    
    def _extract_json(self, response_text: str) -> dict:
        """Extract JSON from response text."""
        start, end = response_text.find('{'), response_text.rfind('}')
        assert start >= 0 and end >= 0, f"Could not find JSON in response: {response_text}"
        json_str = response_text[start:end + 1]
        return json.loads(json_str)
    
    def run(
        self, 
        query: str, 
        model_name: str = "gemini-2.5-flash",
        is_json: bool = False,
        response_schema: BaseModel = None
    ) -> str | dict:
        """
        Run a completion with the specified model.
        
        Args:
            query: The prompt/query to send
            model_name: The Gemini model to use (e.g., "gemini-2.5-flash", "gemini-1.5-pro")
            is_json: Whether to parse the response as JSON
            response_schema: Optional Pydantic model for structured output
            
        Returns:
            str or dict: The model's response, optionally parsed as JSON
        """
        config = {}
        
        if is_json or response_schema:
            config["response_mime_type"] = "application/json"
            if response_schema:
                config["response_schema"] = response_schema
        
        response = self.client.models.generate_content(
            model=model_name,
            contents=query,
            config=config if config else None,
        )
        
        response_text = response.text
        
        if is_json:
            return self._extract_json(response_text)
        else:
            return response_text