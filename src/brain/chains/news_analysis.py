"""
═══════════════════════════════════════════════════════════════════════════════
News Analysis Chain — Phase 2 LangChain Implementation
═══════════════════════════════════════════════════════════════════════════════

WHY THIS FILE EXISTS:
Your old `llm_interface.py` had a `process_news()` method that:
  1. Built an f-string prompt
  2. Called self.generate() (raw HTTP to Ollama)
  3. Called self._extract_json() (60 lines of brace-counting)
  4. Returned Optional[Dict] — no type safety, no validation

This chain does the SAME task but:
  1. Uses ChatPromptTemplate (clean variable injection)
  2. Uses ChatOllama (handles HTTP + streaming)
  3. Uses PydanticOutputParser (auto-retries on bad JSON)
  4. Returns NewsAnalysis (Pydantic model — type-safe, validated)

COMPARISON:
  OLD: result = llm.process_news(article_text)
       # result is Optional[Dict] — could be None, could have wrong keys
  
  NEW: result = news_chain.analyze(article_text)
       # result is Optional[NewsAnalysis] — guaranteed structure if not None
"""

import re
from typing import Optional

from src.brain.langchain_interface import LangChainInterface
from src.models.schemas import NewsAnalysis

# ── LangChain imports for error handling ──
# OutputParserException: Raised when PydanticOutputParser can't parse LLM output.
# We catch this to provide graceful degradation (return None instead of crashing).
from langchain_core.exceptions import OutputParserException


class NewsAnalysisChain:
    """
    LangChain-powered news analysis chain.
    
    Uses the "news_processor" prompt from config/system_prompts.json
    and returns a validated NewsAnalysis Pydantic model.
    """

    def __init__(self, interface: LangChainInterface = None):
        """
        Args:
            interface: LangChainInterface instance. Created automatically if not provided.
        """
        # WHY SHARED INTERFACE? All chains share the same config reader and
        # LLM cache. This avoids re-reading system_prompts.json on every chain.
        self.interface = interface or LangChainInterface()

        # ── Build the structured chain ──
        # This one line replaces ~40 lines of your old process_news() method:
        #   - Loads the "news_processor" prompt from config
        #   - Creates ChatOllama with task-specific model + fallbacks
        #   - Attaches PydanticOutputParser for NewsAnalysis
        #   - Returns a chain: prompt | llm | parser
        self._chain = self.interface.build_structured_chain(
            pydantic_model=NewsAnalysis,
            task_name="news_processor",
        )

    def analyze(self, article_text: str) -> Optional[NewsAnalysis]:
        """
        Analyze a news article and return structured analysis.
        
        WHY TRY/EXCEPT? PydanticOutputParser works in most cases, but LLMs
        are non-deterministic. If the parser can't extract valid JSON after
        retries, we return None gracefully — same behavior as your old code.
        
        Args:
            article_text: Raw news article text to analyze.
            
        Returns:
            NewsAnalysis pydantic model, or None if parsing fails.
        """
        # ── Strip thinking tokens from input ──
        # WHY: Some models (Gemma 4 Heretic) inject thinking tokens into their
        # output. We strip these from the INPUT article text just in case
        # it was previously processed by another model.
        clean_text = self._strip_thinking_tokens(article_text)

        # ── Build the input prompt ──
        # This matches your old prompt structure but uses template variables
        # instead of f-string interpolation
        input_text = f"Analyze this news article and extract viral-worthy information:\n\n{clean_text}"

        try:
            # ── Invoke the chain ──
            # This single call does:
            #   1. Fills {input_text} in the prompt template
            #   2. Injects {format_instructions} from PydanticOutputParser
            #   3. Sends to ChatOllama (with fallback support)
            #   4. Parses response into NewsAnalysis model
            #   5. If step 4 fails, auto-retries with error feedback
            result = self._chain.invoke({"input_text": input_text})

            print(f"  [NEWS-LC] ✅ Analysis complete: {result.topic} (impact={result.impact_score})")
            return result

        except OutputParserException as e:
            # PydanticOutputParser tried but couldn't parse after retries.
            # This is equivalent to your old _extract_json() returning None.
            print(f"  [NEWS-LC] ❌ Parser failed: {e}")
            return None

        except Exception as e:
            # All models in the fallback chain failed (Ollama down, etc.)
            print(f"  [NEWS-LC] ❌ All models failed: {e}")
            return None

    @staticmethod
    def _strip_thinking_tokens(text: str) -> str:
        """Strip thinking tokens from text (Gemma 4, DeepSeek-R1)."""
        text = re.sub(r'<think\b.*?(?:</think\s*>|$)', '', text, flags=re.DOTALL)
        text = re.sub(
            r'<\|channel\>thought.*?(?:<\|channel\>output<\|channel\>|<\|channel\>)?',
            '', text, flags=re.DOTALL
        )
        text = re.sub(r'<\|[^>]*\>', '', text)
        return text.strip()
