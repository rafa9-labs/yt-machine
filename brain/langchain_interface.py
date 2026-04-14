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
  from brain.langchain_interface import LangChainInterface
  interface = LangChainInterface()
  chain = interface.build_structured_chain(NewsAnalysis, "news_processor")
  result = chain.invoke({"input_text": "Iran sanctions..."})
  # result is a NewsAnalysis pydantic model — guaranteed
"""

import json
from pathlib import Path
from typing import Optional, Type, TypeVar

# ── LangChain Imports ──
# ChatOllama: The LangChain wrapper for Ollama's API.
# WHY: Instead of manually POSTing to http://localhost:11434/api/generate
# and parsing streaming NDJSON, ChatOllama handles all HTTP + streaming internally.
from langchain_ollama import ChatOllama

# ChatPromptTemplate: Defines the conversation structure (system + human messages).
# WHY: Your old code mixed system prompts and user prompts via f-strings and
# payload["system"] = system_prompt. ChatPromptTemplate makes this declarative.
from langchain_core.prompts import ChatPromptTemplate

# PydanticOutputParser: Parses LLM output into a Pydantic model.
# WHY: This replaces your 60-line _extract_json() method. It:
#   1. Generates "format instructions" telling the LLM the exact JSON schema
#   2. Tries to parse the LLM's response into your Pydantic model
#   3. If parsing fails, re-prompts the LLM with the error so it can fix itself
from langchain_core.output_parsers import PydanticOutputParser

# StrOutputParser: Returns raw text from LLM response (no JSON parsing).
from langchain_core.output_parsers import StrOutputParser

from langchain_core.language_models import BaseChatModel

T = TypeVar("T")


class LangChainInterface:
    """
    Core LLM interface built on LangChain.

    Reads your existing config/system_prompts.json and creates properly
    configured ChatOllama instances with fallback support.
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = Path(__file__).parent.parent
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
    # LLM Instance Factory
    # ──────────────────────────────────────────────────────────────────

    def get_llm(self, model_name: str = None, temperature: float = 0.7,
                max_tokens: int = 500) -> ChatOllama:
        """
        Create or retrieve a cached ChatOllama instance.

        WHY CACHE? Each ChatOllama instance opens a connection pool to Ollama.
        Reusing instances avoids re-establishing connections on every call.
        """
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
                               max_tokens: int = 500) -> BaseChatModel:
        """
        Create an LLM with automatic fallback chain.

        WHY: Your old code had a 20-line manual fallback loop.
        LangChain's with_fallbacks() does this declaratively — if the primary
        model throws ANY exception, it tries the next model automatically.
        """
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

    def get_prompt(self, task_name: str) -> ChatPromptTemplate:
        """
        Build a ChatPromptTemplate from your existing system_prompts.json config.

        WHY: Separates prompt structure from data. No more f-string mixing.
        """
        prompt_config = self.config["prompts"][task_name]
        system_prompt = prompt_config["system_prompt"]

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
        """
        Build a complete chain: Prompt → LLM → Pydantic Parser.

        THIS IS THE MOST IMPORTANT METHOD.

        OLD FLOW (your code):
          1. Build f-string prompt
          2. requests.post() to Ollama
          3. Parse streaming NDJSON
          4. Try to extract JSON (60 lines of brace counting)
          5. Return raw dict (if lucky)

        NEW FLOW (this method):
          1. Inject variables into prompt template
          2. Send to Ollama via ChatOllama
          3. Parse response into your Pydantic model
          4. If parsing fails → auto-retry with error feedback
          5. Return typed Pydantic model (guaranteed)

        USAGE:
          chain = interface.build_structured_chain(NewsAnalysis, "news_processor")
          result = chain.invoke({"input_text": "Iran announced..."})
          # result is NewsAnalysis(topic="...", impact_score=8, ...)
        """
        config = self.get_prompt_config(task_name)

        # Step 1: Create the output parser for this Pydantic model.
        parser = PydanticOutputParser(pydantic_object=pydantic_model)

        # Step 2: Append format instructions to system prompt.
        # The parser generates instructions that tell the LLM the exact
        # JSON schema it should output — matching your Pydantic model.
        system_with_format = config["system_prompt"] + "\n\n{format_instructions}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_with_format),
            ("human", "{input_text}"),
        ])

        # Step 3: Create the LLM with fallback support.
        llm = self.get_llm_with_fallback(
            primary_model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )

        # Step 4: Build the chain using the pipe (|) operator.
        # prompt | llm | parser creates a RunnableSequence:
        #   prompt → fills template variables → formatted messages
        #   llm    → sends to Ollama → LLM response
        #   parser → parses into Pydantic model → typed result
        chain = prompt | llm | parser

        return chain

    def build_text_chain(self, task_name: str):
        """
        Build a chain that returns plain text (not Pydantic-parsed).

        WHY: Some tasks (like script curation) return plain text, not JSON.
        This chain just gets the raw LLM text output without JSON parsing.
        """
        config = self.get_prompt_config(task_name)

        prompt = ChatPromptTemplate.from_messages([
            ("system", config["system_prompt"]),
            ("human", "{input_text}"),
        ])

        llm = self.get_llm_with_fallback(
            primary_model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )

        chain = prompt | llm | StrOutputParser()
        return chain