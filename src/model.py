"""
LLM inference wrapper for OpenAI and Anthropic APIs.
Provides unified interface for calling different LLM providers.
"""

import os
import time
from typing import Dict, Optional, List
import openai
import anthropic


class LLMInference:
    """Unified LLM inference wrapper supporting multiple providers."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout: int = 60,
    ):
        """
        Initialize LLM inference client.

        Args:
            provider: LLM provider (openai or anthropic)
            model: Model identifier
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 = deterministic)
            timeout: Request timeout in seconds
        """
        self.provider = provider.lower()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # Initialize client
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = openai.OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = anthropic.Anthropic(api_key=api_key)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Track token usage
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    def __call__(
        self, prompt: str, system_prompt: Optional[str] = None, max_retries: int = 3
    ) -> Dict:
        """
        Call LLM with prompt and return response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_retries: Number of retries on failure

        Returns:
            Dict with keys: response, tokens, cost_usd
        """
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    return self._call_openai(prompt, system_prompt)
                elif self.provider == "anthropic":
                    return self._call_anthropic(prompt, system_prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2**attempt)  # exponential backoff

        raise RuntimeError("All retry attempts failed")

    def _call_openai(self, prompt: str, system_prompt: Optional[str]) -> Dict:
        """Call OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
        )

        # Extract response
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens

        # Estimate cost (approximate, update with current pricing)
        cost_usd = self._estimate_cost_openai(tokens)

        # Track totals
        self.total_tokens += tokens
        self.total_cost_usd += cost_usd

        return {"response": content, "tokens": tokens, "cost_usd": cost_usd}

    def _call_anthropic(self, prompt: str, system_prompt: Optional[str]) -> Dict:
        """Call Anthropic API."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout,
        )

        # Extract response
        content = message.content[0].text
        tokens = message.usage.input_tokens + message.usage.output_tokens

        # Estimate cost
        cost_usd = self._estimate_cost_anthropic(
            message.usage.input_tokens, message.usage.output_tokens
        )

        # Track totals
        self.total_tokens += tokens
        self.total_cost_usd += cost_usd

        return {"response": content, "tokens": tokens, "cost_usd": cost_usd}

    def _estimate_cost_openai(self, tokens: int) -> float:
        """Estimate OpenAI API cost. Update with current pricing."""
        # Approximate pricing for gpt-4o-mini (as of 2024)
        # Input: $0.15 / 1M tokens, Output: $0.60 / 1M tokens
        # Simplified: assume 50/50 split
        cost_per_token = (0.15 + 0.60) / 2 / 1_000_000
        return tokens * cost_per_token

    def _estimate_cost_anthropic(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate Anthropic API cost. Update with current pricing."""
        # Approximate pricing for Claude Sonnet (as of 2024)
        # Input: $3 / 1M tokens, Output: $15 / 1M tokens
        input_cost = input_tokens * 3.0 / 1_000_000
        output_cost = output_tokens * 15.0 / 1_000_000
        return input_cost + output_cost

    def get_stats(self) -> Dict:
        """Get total usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }


def create_llm(cfg) -> LLMInference:
    """
    Create LLM inference client from Hydra config.

    Args:
        cfg: Hydra config object

    Returns:
        LLMInference instance
    """
    return LLMInference(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        max_tokens=cfg.llm.max_tokens,
        temperature=cfg.llm.temperature,
        timeout=cfg.llm.timeout,
    )
