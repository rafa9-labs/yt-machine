"""
generate_v2.py — DEPRECATED: Redirects to unified pipeline.

The v1/v2 pipelines have been merged into generate_complete_video.py.
This file exists for backward compatibility only.
"""
import sys
import os

print("⚠️  generate_v2.py is deprecated. Running unified pipeline instead.")
print("   → Use: python generate_complete_video.py")

# Re-execute the unified pipeline with the same arguments
os.execvp(sys.executable, [sys.executable, "generate_complete_video.py"] + sys.argv[1:])