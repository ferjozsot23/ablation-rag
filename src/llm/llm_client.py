"""
LLM client for OpenAI-compatible chat endpoints.

Two configurations are supported, both read from environment variables (a `.env`
file in the project root is loaded automatically):

1. Default client (systems S1-S3, entity extraction):
   - OPENAI_API_KEY:  API key (falls back to LLM_API_KEY)
   - OPENAI_BASE_URL: base URL (default: https://api.openai.com/v1)
   - LLM_MODEL:       model name (default: gpt-4o-mini)

2. Self-hosted OpenAI-compatible endpoint (used by S4 and the S5 agent), see
   `glm_endpoint()`:
   - GLM_BASE_URL: base URL of the server, e.g. http://<host>:<port>/v1 (required)
   - GLM_API_KEY:  API key (default: "EMPTY", as most local servers ignore it)
   - GLM_MODEL:    model name (default: zai-org/GLM-5.3-Flash)
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from openai import OpenAI

# Auto-load the .env file if present
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # python-dotenv not installed, rely on the system environment


def glm_endpoint() -> Dict[str, str]:
    """
    Connection settings for the self-hosted OpenAI-compatible endpoint.

    Returns:
        Dict with `base_url`, `api_key` and `model`, ready to be passed to
        `LLMClient(...)` or `ChatOpenAI(...)`.

    Raises:
        RuntimeError: if GLM_BASE_URL is not set.
    """
    base_url = os.getenv('GLM_BASE_URL')
    if not base_url:
        raise RuntimeError(
            "GLM_BASE_URL is not set. Add it to .env (see .env.example) to point "
            "S4/S5 at your OpenAI-compatible server."
        )
    return {
        'base_url': base_url,
        'api_key': os.getenv('GLM_API_KEY', 'EMPTY'),
        'model': os.getenv('GLM_MODEL', 'zai-org/GLM-5.3-Flash'),
    }


class LLMClient:
    """
    Thin client for LLMs behind an OpenAI-compatible API.

    Works with the OpenAI API (gpt-4o-mini, gpt-4o, ...) and with local or remote
    servers that implement the same protocol (vLLM, Ollama, ...).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8000,
        timeout: float = 300.0  # 5 min, self-hosted servers can be slow
    ):
        """
        Args:
            base_url: server base URL (default: env OPENAI_BASE_URL or the OpenAI API)
            api_key: API key (default: env OPENAI_API_KEY, then LLM_API_KEY)
            model: model name (default: env LLM_MODEL or gpt-4o-mini)
            temperature: sampling temperature (default: 0.0, deterministic)
            max_tokens: maximum output tokens (default: 8000)
            timeout: request timeout in seconds (default: 300)
        """
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        self.api_key = api_key or os.getenv('OPENAI_API_KEY') or os.getenv('LLM_API_KEY', 'EMPTY')
        self.model = model or os.getenv('LLM_MODEL', 'gpt-4o-mini')

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an answer from the LLM.

        Args:
            prompt: user prompt
            temperature: overrides self.temperature
            max_tokens: overrides self.max_tokens
            system_message: optional system message

        Returns:
            Dict with:
                - answer: str - generated answer (None on error)
                - prompt_tokens: int - prompt tokens
                - completion_tokens: int - completion tokens
                - total_tokens: int - total tokens
                - time_seconds: float - generation time
                - model: str - model used
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok
            )

            elapsed = time.time() - start_time
            answer = response.choices[0].message.content
            usage = response.usage

            return {
                'answer': answer,
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
                'reasoning_tokens': getattr(usage, 'reasoning_tokens', None),  # if reported
                'llm_calls': 1,
                'time_seconds': elapsed,
                'model': self.model
            }

        except Exception as e:
            # Return the error together with call metadata
            elapsed = time.time() - start_time
            return {
                'answer': None,
                'error': str(e),
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'reasoning_tokens': None,
                'llm_calls': 1,  # the attempt still counts as a call
                'time_seconds': elapsed,
                'model': self.model
            }


def get_llm_client(
    temperature: float = 0.0,
    max_tokens: int = 8000
) -> LLMClient:
    """
    Factory for a client configured from the environment.

    Args:
        temperature: sampling temperature (default: 0.0)
        max_tokens: maximum output tokens (default: 8000)
    """
    return LLMClient(
        temperature=temperature,
        max_tokens=max_tokens
    )


def main():
    """Smoke test for the LLM client."""
    print("\n" + "="*60)
    print("LLM CLIENT - TEST")
    print("="*60)

    print("\n1. Initializing client...")
    client = get_llm_client()
    print(f"   Base URL: {client.base_url}")
    print(f"   Model: {client.model}")
    print(f"   Temperature: {client.temperature}")
    print(f"   Max tokens: {client.max_tokens}")

    print("\n2. Test query: 'What is 2+2?'")
    result = client.generate(
        prompt="What is 2+2? Answer in one sentence.",
        system_message="You are a helpful assistant."
    )

    if result.get('error'):
        print(f"   ERROR: {result['error']}")
    else:
        print(f"\n   Answer: {result['answer']}")
        print(f"   Tokens: {result['prompt_tokens']} prompt + {result['completion_tokens']} completion = {result['total_tokens']} total")
        print(f"   Time: {result['time_seconds']:.2f}s")

    print("\n" + "="*60)
    print("Test completed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
