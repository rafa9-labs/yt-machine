"""
═══════════════════════════════════════════════════════════════════════════════
LangChain LLM Interface — Phase 2 of the Industrial-Grade Stack Migration
═══════════════════════════════════════════════════════════════════════════════

WHY THIS FILE EXISTS:
Your old `brain/llm_interface.py` manually handled:
  - HTTP requests to Ollama via `requests.post()`          →  Now: `ChatOllama`
  - Streaming NDJSON line parsing                          →  Now: Built-in to ChatOllama
  - JSON extraction with 60 lines of brace-counting        →  Now: `PydanticOutputParser`
  - Manual model fallback loop (20 lines)                  →  Now: `with_fallbacks()`
  - Circuit breaker state (`_primary_failures` counter)    →  Now: LangChain retry config

ARCHITECTURE:
  LangChainInterface  →  reads config/system_prompts.json
                       →  creates ChatOllama instances (primary + fallback)
                       →  exposes helper methods for building chains

  Each chain file (news_analysis.py, debate.py, etc.) uses this interface
  to create a LangChain Runnable with:
    prompt_template | llm | output_parser

  The chain is then invoked with:  chain.invoke({"input_text": "..."})
  And returns a Pydantic model, not a raw dict.

IMPORT IN YOUR CODE:
  from src.brain.langchain_interface import LangChainInterface
  interface = LangChainInterface()
  chain = interface.build_structured_chain(NewsAnalysis, "news_processor")
  result = chain.invoke({"input_text": "Iran sanctions..."})
  # result is a NewsAnalysis pydantic model — guaranteed
"""

import json
import re
from pathlib import Path
from typing import Optional, Type, TypeVar

T = TypeVar("T")


class LangChainInterface:
    """
    Core LLM interface built on LangChain.

    Reads your existing config/system_prompts.json and creates properly
    configured ChatOllama instances with fallback support.
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = Path(__file__).parent.parent.parent
            config_path = base_dir / "config" / "system_prompts.json"

        self.config_path = Path(config_path)
        self.config = self._load_config()

        model_config = self.config["model_config"]
        self.base_url = model_config["base_url"]
        self.default_model = model_config["default_model"]
        self.fallback_model = model_config.get("fallback_model", "llama3.2:latest")
        self.num_ctx = model_config.get("num_ctx", 4096)
        self.task_models = model_config.get("task_models", {})
        self._llm_cache = {}

    def _load_config(self) -> dict:
        """Load system_prompts.json — same config your old code reads."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ──────────────────────────────────────────────────────────────────
    # Token Cleaning — Strip model thinking tokens before parsing
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_thinking_tokens(text: str) -> str:
        """Strip model thinking/special tokens from LLM output."""
        text = re.sub(
            r'<\|\s*channel\s*(?:\|?\s*)?>\s*thought\s*<\s*channel\s*(?:\|?\s*)?>',
            '', text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r'<\|\s*channel\s*(?:\|?\s*)?>\s*output\s*<\s*channel\s*(?:\|?\s*)?>',
            '', text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r'<think\b.*?</think\s*>?', '', text, flags=re.DOTALL)
        text = re.sub(r'</?think[^>]*>?', '', text)
        text = re.sub(r'<\|\s*channel\s*(?:\|?\s*)?>', '', text, flags=re.IGNORECASE)
        return text.strip()

    @classmethod
    def _clean_llm_output(cls, msg):
        """Strip thinking tokens from AIMessage.content before parsing."""
        if hasattr(msg, 'content'):
            msg.content = cls._strip_thinking_tokens(msg.content)
        return msg

    # ──────────────────────────────────────────────────────────────────
    # LLM Instance Factory
    # ──────────────────────────────────────────────────────────────────

    def get_llm(self, model_name: str = None, temperature: float = 0.7,
                max_tokens: int = 500):
        from langchain_ollama import ChatOllama

        if model_name is None:
            model_name = self.default_model

        cache_key = f"{model_name}|{temperature}|{max_tokens}"

        if cache_key not in self._llm_cache:
            self._llm_cache[cache_key] = ChatOllama(
                model=model_name,
                base_url=self.base_url,
                temperature=temperature,
                num_predict=max_tokens,
                num_ctx=self.num_ctx,
            )

        return self._llm_cache[cache_key]

    def get_llm_with_fallback(self, primary_model: str = None,
                               fallback_models: list = None,
                               temperature: float = 0.7,
                               max_tokens: int = 500):
        if primary_model is None:
            primary_model = self.default_model

        if fallback_models is None:
            fallback_models = [self.fallback_model, "llama3.2:latest"]

        primary = self.get_llm(primary_model, temperature, max_tokens)
        fallbacks = [self.get_llm(m, temperature, max_tokens) for m in fallback_models]

        return primary.with_fallbacks(fallbacks)

    # ──────────────────────────────────────────────────────────────────
    # Prompt Template Builder
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _escape_for_template(text: str) -> str:
        """Escape literal curly braces for LangChain ChatPromptTemplate.
        
        Doubles all { and } so Python string formatting treats them as literals.
        Only called when building LangChain templates — raw LLM path uses unescaped text.
        """
        return text.replace('{', '{{').replace('}', '}}')

    def get_prompt(self, task_name: str):
        from langchain_core.prompts import ChatPromptTemplate

        prompt_config = self.config["prompts"][task_name]
        system_prompt = self._escape_for_template(prompt_config["system_prompt"])

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input_text}"),
        ])

    def get_prompt_config(self, task_name: str) -> dict:
        """Get temperature, max_tokens, and model for a specific task."""
        prompt_config = self.config["prompts"][task_name]
        task_model = self.task_models.get(task_name, self.default_model)
        return {
            "temperature": prompt_config.get("temperature", 0.7),
            "max_tokens": prompt_config.get("max_tokens", 500),
            "model": task_model,
            "system_prompt": prompt_config["system_prompt"],
        }

    # ──────────────────────────────────────────────────────────────────
    # Chain Builders — The Core Pattern
    # ──────────────────────────────────────────────────────────────────

    def build_structured_chain(self, pydantic_model: Type[T], task_name: str):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.runnables import RunnableLambda

        config = self.get_prompt_config(task_name)

        parser = PydanticOutputParser(pydantic_object=pydantic_model)

        escaped_system = self._escape_for_template(config["system_prompt"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", escaped_system + "\n\n{format_instructions}"),
            ("human", "{input_text}"),
        ]).partial(format_instructions=parser.get_format_instructions())

        llm = self.get_llm_with_fallback(
            primary_model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )

        chain = prompt | llm | RunnableLambda(self._clean_llm_output) | parser

        return chain

    def build_text_chain(self, task_name: str):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda

        config = self.get_prompt_config(task_name)

        escaped_system = self._escape_for_template(config["system_prompt"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", escaped_system),
            ("human", "{input_text}"),
        ])

        llm = self.get_llm_with_fallback(
            primary_model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )

        chain = prompt | llm | RunnableLambda(self._clean_llm_output) | StrOutputParser()
        return chain
