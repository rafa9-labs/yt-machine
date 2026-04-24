"""
═══════════════════════════════════════════════════════════════════════════════
Debate Chain — Phase 2 LangChain Implementation
═══════════════════════════════════════════════════════════════════════════════

WHY: Your old llm_interface.py had debate_skeptic() and debate_explainer()
methods that each: build f-string → call generate() → _extract_json() → return Dict.

This chain does BOTH steps as a multi-step LangChain pipeline:
  1. Feed news analysis → skeptic chain → get critique
  2. Feed news + critique → explainer chain → get explanation
  
Each step returns a typed Pydantic model instead of a raw dict.

MULTI-STEP CHAIN CONCEPT:
  In LangChain, a "multi-step chain" is just calling multiple chains sequentially
  and passing the output of one as input to the next. Your old code did this
  with separate method calls. LangChain makes the data flow explicit and typed.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from src.brain.langchain_interface import LangChainInterface
from langchain_core.exceptions import OutputParserException


# ── Pydantic models for debate outputs ──
# WHY: These replace the untyped dicts your old code returned.
# The LLM is forced to output JSON matching these schemas via PydanticOutputParser.

class SkepticResponse(BaseModel):
    """Structured output from the Skeptic debate step."""
    critique: str = Field(description="Critical analysis questioning the narrative")
    key_question: str = Field(description="The most important unanswered question")


class ExplainerResponse(BaseModel):
    """Structured output from the Explainer debate step."""
    explanation: str = Field(description="Simplified response to the critique")
    analogy: str = Field(description="A relatable comparison or example")


class DebateChain:
    """
    Two-step debate pipeline: Skeptic → Explainer.
    
    WHY A CLASS? Both steps share the same LangChainInterface (config + LLM cache).
    Keeping them in one class ensures they use the same model configuration.
    """

    def __init__(self, interface: LangChainInterface = None):
        self.interface = interface or LangChainInterface()

        # ── Build two separate chains ──
        # Each chain is: prompt_template | llm_with_fallback | pydantic_parser
        # They're independent — can be called separately or together via run_debate()

        # Chain 1: Skeptic questions the narrative
        self._skeptic_chain = self.interface.build_structured_chain(
            pydantic_model=SkepticResponse,
            task_name="debate_skeptic",
        )

        # Chain 2: Explainer simplifies and responds to the critique
        self._explainer_chain = self.interface.build_structured_chain(
            pydantic_model=ExplainerResponse,
            task_name="debate_explainer",
        )

    def run_skeptic(self, news_summary: Dict[str, Any]) -> Optional[SkepticResponse]:
        """
        Run the Skeptic against a news summary.
        
        OLD: llm.debate_skeptic(news_summary) → Optional[Dict]
        NEW: debate.run_skeptic(news_summary) → Optional[SkepticResponse]
        
        Args:
            news_summary: Dict with 'topic', 'key_facts', 'angle' keys.
                         (Will eventually be a NewsAnalysis model after full migration)
        """
        # Build the same prompt content your old code used, but via template
        topic = news_summary.get('topic', '')
        key_facts = ', '.join(news_summary.get('key_facts', []))
        angle = news_summary.get('angle', '')

        input_text = (
            f"Critique this news story:\n\n"
            f"Topic: {topic}\n"
            f"Key Facts: {key_facts}\n"
            f"Angle: {angle}"
        )

        try:
            result = self._skeptic_chain.invoke({"input_text": input_text})
            print(f"  [DEBATE-LC] ✅ Skeptic: {result.key_question[:60]}...")
            return result
        except OutputParserException as e:
            print(f"  [DEBATE-LC] ❌ Skeptic parser failed: {e}")
            return None
        except Exception as e:
            print(f"  [DEBATE-LC] ❌ Skeptic failed: {e}")
            return None

    def run_explainer(self, news_summary: Dict[str, Any],
                       skeptic_response: SkepticResponse) -> Optional[ExplainerResponse]:
        """
        Run the Explainer to respond to the skeptic's critique.
        
        OLD: llm.debate_explainer(news_summary, skeptic_response) → Optional[Dict]
        NEW: debate.run_explainer(news_summary, skeptic) → Optional[ExplainerResponse]
        """
        topic = news_summary.get('topic', '')
        critique = skeptic_response.critique if skeptic_response else ''
        key_question = skeptic_response.key_question if skeptic_response else ''

        input_text = (
            f"Respond to this critique:\n\n"
            f"Topic: {topic}\n"
            f"Skeptic's Critique: {critique}\n"
            f"Key Question: {key_question}"
        )

        try:
            result = self._explainer_chain.invoke({"input_text": input_text})
            print(f"  [DEBATE-LC] ✅ Explainer: {result.analogy[:60]}...")
            return result
        except OutputParserException as e:
            print(f"  [DEBATE-LC] ❌ Explainer parser failed: {e}")
            return None
        except Exception as e:
            print(f"  [DEBATE-LC] ❌ Explainer failed: {e}")
            return None

    def run_debate(self, news_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Run the full 2-step debate: Skeptic → Explainer.
        
        WHY THIS METHOD EXISTS: It orchestrates both steps in sequence,
        passing the skeptic's output into the explainer's input.
        Your old generate_complete_video.py called these methods sequentially —
        this encapsulates that pattern.
        
        Args:
            news_summary: Dict with 'topic', 'key_facts', 'angle' keys.
            
        Returns:
            Dict with 'skeptic' (SkepticResponse) and 'explainer' (ExplainerResponse),
            or None if the skeptic step fails (can't debate without a critique).
        """
        # Step 1: Skeptic critiques the news
        skeptic = self.run_skeptic(news_summary)
        if not skeptic:
            print("  [DEBATE-LC] Cannot run explainer without skeptic response")
            return None

        # Step 2: Explainer responds to the critique
        explainer = self.run_explainer(news_summary, skeptic)
        if not explainer:
            # Explainer failed — return just the skeptic's critique
            print("  [DEBATE-LC] Explainer failed, returning skeptic only")
            return {
                "skeptic": skeptic,
                "explainer": None,
            }

        return {
            "skeptic": skeptic,
            "explainer": explainer,
        }
