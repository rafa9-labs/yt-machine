"""Collection policy for tests.

Everything under tests/ is a pytest module now. The former live-service
scripts (test_pipeline_models.py, test_improvements.py, test_vector_memory.py,
test_video_rebuild.py) were removed: they imported deleted module paths
(video_server.*, brain.*, db.*) or asserted hardcoded model names that no
longer exist. The parsing coverage they held is now in
tests/test_llm_parsing.py.
"""

collect_ignore = []
