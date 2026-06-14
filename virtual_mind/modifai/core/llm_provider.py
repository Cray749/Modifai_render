"""
LLM Provider Abstraction.
Supports AWS Bedrock and Google Gemini for running Modifai locally or in production.
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def safe_json_generation(
    func,
    max_retries: int = 1,
    return_raw: bool = False,
) -> Any:
    """
    Helper to retry JSON generation and parsing.
    Ensures that the output is always a parsed dictionary.
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            raw_text = func()
            
            if return_raw:
                return raw_text
                
            # Clean up potential markdown formatting
            text = raw_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError) as e:
            if return_raw:
                raise e
            attempt += 1
            if attempt > max_retries:
                raise ValueError(f"Failed to generate valid JSON after {max_retries + 1} attempts: {e}")
            logger.warning("JSON generation failed, retrying (attempt %d): %s", attempt, e)
            time.sleep(1)
    
    raise ValueError("Unreachable")


class BaseLLMProvider(ABC):
    """Common interface for reasoning agents."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        return_raw: bool = False,
        **kwargs,
    ) -> Any:
        """
        Generate structured output.
        
        Args:
            system_prompt: The system prompt.
            user_prompt: The user prompt.
            response_schema: The JSON schema definition (e.g., from Bedrock tool inputSchema).
            tool_name: (Optional) The name of the tool to force, for logging/consistency.
            
        Returns:
            A dictionary containing the parsed JSON output.
        """
        pass


class BedrockProvider(BaseLLMProvider):
    def __init__(self, model_id: str, region: str):
        import boto3
        self.model_id = model_id
        self.region = region
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        return_raw: bool = False,
        **kwargs,
    ) -> Any:
        
        tool_name = tool_name or "generate_output"
        
        kwargs_converse = {
            "modelId": self.model_id,
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
        }
        
        if response_schema is not None:
            kwargs_converse["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool_name,
                            "description": "Output generation tool",
                            "inputSchema": {"json": response_schema}
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": tool_name}},
            }
        
        
        inference_config = {}
        if "temperature" in kwargs:
            inference_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            inference_config["maxTokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            inference_config["topP"] = kwargs["top_p"]
            
        if inference_config:
            kwargs_converse["inferenceConfig"] = inference_config
        
        response = self._client.converse(**kwargs_converse)
        
        if response_schema is not None and not return_raw:
            content_blocks = response["output"]["message"]["content"]
            for block in content_blocks:
                if block.get("toolUse", {}).get("name") == tool_name:
                    return block["toolUse"]["input"]
            raise ValueError(f"Bedrock did not call tool {tool_name}. Response: {content_blocks}")
        else:
            raw_text = response["output"]["message"]["content"][0]["text"]
            if return_raw:
                return raw_text
                
            # Rely on safe_json_generation logic at call site if needed, or parse here.
            # BaseLLMProvider contract: returns Dict. So we must parse it.
            text = raw_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())


class GeminiProvider(BaseLLMProvider):
    def __init__(self, model_id: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        import google.generativeai as genai
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set to use GeminiProvider")
            
        genai.configure(api_key=api_key)
        self.model_id = model_id
        
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        return_raw: bool = False,
        **kwargs,
    ) -> Any:
        
        import google.generativeai as genai
        
        system_instruction = system_prompt
        if response_schema:
            schema_str = json.dumps(response_schema, indent=2)
            system_instruction += (
                f"\n\nIMPORTANT: You MUST output a raw JSON object matching "
                f"the following schema. DO NOT wrap the output in markdown blocks like ```json.\n"
                f"{schema_str}"
            )

        temperature = kwargs.get("temperature", 0.1)
        max_tokens = kwargs.get("max_tokens")
        top_p = kwargs.get("top_p")
        
        gen_kwargs = {
            "temperature": temperature,
            "response_mime_type": "application/json",
        }
        if max_tokens is not None:
            gen_kwargs["max_output_tokens"] = max_tokens
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
            
        model = genai.GenerativeModel(
            model_name=self.model_id,
            system_instruction=system_instruction,
            generation_config=genai.types.GenerationConfig(**gen_kwargs)
        )
        
        def _do_generate():
            response = model.generate_content(user_prompt)
            if not response.text:
                raise ValueError("Empty response from Gemini")
            return response.text
            
        # We rely on safe_json_generation to handle simple formatting issues 
        # and validate that it's a dict. 
        return safe_json_generation(_do_generate, max_retries=1, return_raw=return_raw)


class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, model_id: Optional[str] = None, api_key: Optional[str] = None):
        import requests
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set to use OpenRouterProvider")
            
        env_model = os.environ.get("OPENROUTER_MODEL")
        self.primary_model = model_id or env_model or "deepseek/deepseek-chat-v3"
        
        self.fallback_models = [
            "deepseek/deepseek-chat-v3",
            "qwen/qwen3-235b-a22b",
            "google/gemini-2.5-flash-lite"
        ]
        
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        return_raw: bool = False,
        **kwargs,
    ) -> Any:
        
        import requests
        
        system_instruction = system_prompt
        if response_schema:
            schema_str = json.dumps(response_schema, indent=2)
            system_instruction += (
                f"\n\nIMPORTANT: You MUST output a raw JSON object matching "
                f"the following schema. DO NOT wrap the output in markdown blocks like ```json.\n"
                f"{schema_str}"
            )
            
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        models_to_try = [self.primary_model]
        for m in self.fallback_models:
            if m not in models_to_try:
                models_to_try.append(m)
                
        last_error = None
        for current_model in models_to_try:
            payload = {
                "model": current_model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.1),
            }
            if "max_tokens" in kwargs:
                payload["max_tokens"] = kwargs["max_tokens"]
            if "top_p" in kwargs:
                payload["top_p"] = kwargs["top_p"]
            
            def _do_generate():
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=90
                )
                response.raise_for_status()
                data = response.json()
                if "choices" not in data or len(data["choices"]) == 0:
                    raise ValueError(f"Empty choices in response: {data}")
                text = data["choices"][0]["message"]["content"]
                if not text:
                    raise ValueError("Empty response text")
                return text
                
            try:
                return safe_json_generation(_do_generate, max_retries=1, return_raw=return_raw)
            except Exception as e:
                logger.warning("OpenRouter generation failed for model %s: %s. Trying next model...", current_model, e)
                last_error = e
                continue
                
        raise ValueError(f"All fallback models failed. Last error: {last_error}")


def get_llm_provider() -> BaseLLMProvider:
    """
    Factory to return the configured LLM provider based on LLM_PROVIDER.
    Defaults to gemini.
    """
    provider_type = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider_type == "bedrock":
        model_id = os.environ.get("AWS_MODEL_ID", "amazon.nova-micro-v1:0")
        region = os.environ.get("AWS_REGION", "us-east-1")
        return BedrockProvider(model_id=model_id, region=region)
    elif provider_type == "gemini":
        model_id = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        return GeminiProvider(model_id=model_id)
    elif provider_type == "openrouter":
        model_id = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3")
        return OpenRouterProvider(model_id=model_id)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider_type}")
