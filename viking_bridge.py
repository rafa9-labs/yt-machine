from fastapi import FastAPI, HTTPException
from pathlib import Path
import sys
import json
from typing import Dict, Any
import importlib.util

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

debate_module = load_module("debate_engine", project_root / "redfish" / "debate_engine.py")
memory_module = load_module("memory_logger", project_root / "open-viking" / "memory_logger.py")

DebateEngine = debate_module.DebateEngine
MemoryLogger = memory_module.MemoryLogger

app = FastAPI(title="Viking Bridge API", version="1.0.0")

@app.get("/")
async def root():
    return {
        "service": "Viking Bridge API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": [
            "/trigger-video (POST)",
            "/health (GET)",
            "/status (GET)"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "viking-bridge"}

@app.post("/trigger-video")
async def trigger_video_generation():
    try:
        print("\n" + "=" * 60)
        print("VIDEO GENERATION TRIGGERED")
        print("=" * 60)
        
        print("\n[1/3] Scraping news and generating script...")
        debate_engine = DebateEngine()
        results = debate_engine.process_top_articles(max_articles=1)
        
        if not results or len(results) == 0:
            raise HTTPException(status_code=500, detail="No articles could be processed")
        
        script_data = results[0]
        script = script_data["script"]
        analysis = script_data["analysis"]
        
        print(f"  ✓ Script generated: {analysis['topic']}")
        
        print("\n[2/3] Video generation would happen here...")
        print("  Note: Video server components need to be recreated")
        print("  See: video_server/ directory")
        
        video_path = str(project_root / "output" / "videos" / "final_complete_video.mp4")
        
        print("\n[3/3] Logging to memory...")
        logger = MemoryLogger()
        video_id = logger.log_video(
            title=analysis['topic'],
            topic=analysis['topic'],
            keywords=analysis.get('keywords', []),
            duration=40.0,
            file_path=video_path,
            script=script,
            metadata={
                "virality_score": analysis.get('virality_score', 0),
                "source_feed": script_data.get('article', {}).get('feed_name', 'unknown'),
                "generation_timestamp": script_data.get('timestamp', '')
            }
        )
        
        print(f"  ✓ Logged to memory: {video_id}")
        
        print("\n" + "=" * 60)
        print("✓ SCRIPT GENERATION COMPLETE")
        print("=" * 60)
        
        return {
            "success": True,
            "video_id": video_id,
            "video_path": video_path,
            "title": analysis['topic'],
            "duration": 40.0,
            "virality_score": analysis.get('virality_score', 0),
            "keywords": analysis.get('keywords', []),
            "note": "Video generation components need recreation - see video_server/"
        }
    
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    base_dir = Path(__file__).parent
    
    memory_path = base_dir / "open-viking" / "history" / "videos.json"
    scripts_path = base_dir / "open-viking" / "resources" / "generated_scripts.json"
    videos_dir = base_dir / "output" / "videos"
    
    video_count = 0
    if memory_path.exists():
        with open(memory_path, 'r') as f:
            data = json.load(f)
            video_count = data.get('metadata', {}).get('total_count', 0)
    
    script_count = 0
    if scripts_path.exists():
        with open(scripts_path, 'r') as f:
            scripts = json.load(f)
            script_count = len(scripts)
    
    output_videos = list(videos_dir.glob("*.mp4")) if videos_dir.exists() else []
    
    return {
        "status": "operational",
        "statistics": {
            "total_videos_logged": video_count,
            "scripts_generated": script_count,
            "output_videos": len(output_videos)
        },
        "paths": {
            "memory": str(memory_path),
            "scripts": str(scripts_path),
            "videos": str(videos_dir)
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("VIKING BRIDGE API SERVER")
    print("=" * 60)
    print("\nStarting server on http://localhost:5000")
    print("Endpoints:")
    print("  - GET  /         : API info")
    print("  - GET  /health   : Health check")
    print("  - GET  /status   : System status")
    print("  - POST /trigger-video : Generate script from news")
    print("\nNote: Video generation requires video_server/ components")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
