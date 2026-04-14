"""
═══════════════════════════════════════════════════════════════════════════════
Curation Chain — Phase 2 LangChain Implementation
═══════════════════════════════════════════════════════════════════════════════

WHY: Your old curate_script() method is unique — it does NOT return JSON.
It sends story narration bodies to the LLM and gets back plain text.
This demonstrates LangChain's StrOutputParser (returns raw text, no JSON parsing).

KEY CONCEPT — build_text_chain() vs build_structured_chain():
  - build_structured_chain() → PydanticOutputParser → returns Pydantic model
  - build_text_chain()       → StrOutputParser     → returns raw string

Your curation step uses the text chain because the LLM outputs curated
story narrations as plain text (with [STORY N] markers), not JSON.
"""

from typing import Optional, List

from brain.langchain_interface import LangChainInterface


class CurationChain:
    """
    LangChain-powered script curation.
    
    Uses the "script_curator" prompt to transform written narration into
    natural spoken language optimized for ElevenLabs TTS.
    """

    def __init__(self, interface: LangChainInterface = None):
        self.interface = interface or LangChainInterface()

        # ── Build a TEXT chain (not structured) ──
        # WHY: Curation returns plain text stories, not JSON objects.
        # StrOutputParser just returns the raw LLM text content.
        self._chain = self.interface.build_text_chain(
            task_name="script_curator",
        )

    def curate_stories(self, story_bodies: List[str]) -> Optional[str]:
        """
        Send story narration bodies to the LLM for curation.
        
        WHY SEPARATE FROM SCRIPT SYNTHESIS? Your old code had a "structural slicing"
        pattern: only story BODIES are sent to the curator. Greeting, segues,
        and closing are NEVER touched by the LLM. This chain only handles
        the story bodies — the caller reassembles the full script.
        
        Args:
            story_bodies: List of story narration texts (usually 3 stories).
            
        Returns:
            Curated story text with [STORY N] markers, or None on failure.
        """
        if len(story_bodies) < 2:
            print(f"  [CURATOR-LC] Not enough stories ({len(story_bodies)}), skipping")
            return None

        # Build body-only text with [STORY N] markers (same as your old code)
        body_text = "\n\n---\n\n".join(
            f"[STORY {i+1}]\n{body}" for i, body in enumerate(story_bodies)
        )

        input_text = (
            f"Transform these 3 story narrations from written text into natural, "
            f"human-sounding spoken language.\n\n"
            f"ORIGINAL STORY NARRATIONS:\n{body_text}\n\n"
            f"Output the 3 curated stories as plain text. "
            f"Keep [STORY N] markers. Separate stories with ---. No JSON."
        )

        try:
            # ── Invoke the text chain ──
            # Returns a plain string — no Pydantic parsing
            result = self._chain.invoke({"input_text": input_text})

            if not result or not result.strip():
                print("  [CURATOR-LC] ❌ Empty response")
                return None

            # Clean any accidental markdown wrapping (same as your old code)
            curated = result.strip()
            if curated.startswith('```'):
                curated = curated.split('\n', 1)[-1]
            if curated.endswith('```'):
                curated = curated.rsplit('```', 1)[0]

            print(f"  [CURATOR-LC] ✅ Curated {len(story_bodies)} stories ({len(curated.split())} words)")
            return curated.strip()

        except Exception as e:
            print(f"  [CURATOR-LC] ❌ Curation failed: {e}")
            return None