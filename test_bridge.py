import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing Viking Bridge imports...")

try:
    import importlib.util
    
    def load_module(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    print("✓ Loading debate_engine...")
    debate_module = load_module("debate_engine", project_root / "redfish" / "debate_engine.py")
    
    print("✓ Loading voiceover_generator...")
    voiceover_module = load_module("voiceover_generator", project_root / "video_server" / "voiceover_generator.py")
    
    print("✓ Loading footage_fetcher...")
    footage_module = load_module("footage_fetcher", project_root / "video_server" / "footage_fetcher.py")
    
    print("✓ Loading video_assembler...")
    assembler_module = load_module("video_assembler", project_root / "video_server" / "video_assembler.py")
    
    print("✓ Loading memory_logger...")
    memory_module = load_module("memory_logger", project_root / "open-viking" / "memory_logger.py")
    
    print("\n✓ All modules loaded successfully!")
    print("\nNow testing FastAPI server...")
    
    from fastapi import FastAPI
    import uvicorn
    
    print("✓ FastAPI imports successful")
    print("\nViking Bridge is ready to run!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
