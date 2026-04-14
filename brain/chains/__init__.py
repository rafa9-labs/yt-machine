# ── Phase 2: LangChain Chains Package ──
# WHY: Each LLM "task" (news analysis, debate, script synthesis, curation)
# gets its own chain module. This replaces the monolithic llm_interface.py
# where every prompt was a method on a single 1,300-line class.
#
# Each chain is:
#   1. Independently testable
#   2. Bound to a specific model (via config)
#   3. Returns a Pydantic model (not a raw dict)
#   4. Has built-in fallback + retry via LangChain
#
# Import what you need:
#   from brain.chains import NewsAnalysisChain, DebateChain, ScriptChain