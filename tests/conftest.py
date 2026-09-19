"""Collection policy for legacy manual integration scripts.

These files are executable checks, not pytest modules: they perform work at
import time, require external services, or reference a historical generated
project. Keep them runnable directly without letting them abort normal unit
test collection.
"""


collect_ignore = [
    "test_improvements.py",
    "test_langchain_chains.py",
    "test_option_a_layout.py",
    "test_pipeline_models.py",
    "test_vector_memory.py",
    "test_video_rebuild.py",
]
