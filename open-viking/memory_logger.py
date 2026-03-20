import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class MemoryLogger:
    def __init__(self, history_path: str = None):
        if history_path is None:
            base_dir = Path(__file__).parent
            history_path = base_dir / "history" / "videos.json"
        
        self.history_path = Path(history_path)
        self._ensure_history_file()
    
    def _ensure_history_file(self):
        if not self.history_path.exists():
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            initial_data = {
                "videos": [],
                "metadata": {
                    "total_count": 0,
                    "last_updated": None,
                    "version": "1.0"
                }
            }
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2)
    
    def log_video(self, metadata: Dict[str, Any]) -> bool:
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            video_entry = {
                "id": data["metadata"]["total_count"] + 1,
                "timestamp": datetime.utcnow().isoformat(),
                "topic": metadata.get("topic", ""),
                "script": metadata.get("script", {}),
                "keywords": metadata.get("keywords", []),
                "video_path": metadata.get("video_path", ""),
                "duration": metadata.get("duration", 0),
                "source_url": metadata.get("source_url", ""),
                "performance_metrics": metadata.get("performance_metrics", {
                    "views": 0,
                    "likes": 0,
                    "shares": 0,
                    "engagement_rate": 0.0
                }),
                "platforms": metadata.get("platforms", []),
                "status": metadata.get("status", "generated")
            }
            
            data["videos"].append(video_entry)
            data["metadata"]["total_count"] += 1
            data["metadata"]["last_updated"] = datetime.utcnow().isoformat()
            
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        
        except Exception as e:
            print(f"Error logging video: {e}")
            return False
    
    def update_performance(self, video_id: int, metrics: Dict[str, Any]) -> bool:
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for video in data["videos"]:
                if video["id"] == video_id:
                    video["performance_metrics"].update(metrics)
                    video["last_performance_update"] = datetime.utcnow().isoformat()
                    break
            else:
                print(f"Video ID {video_id} not found")
                return False
            
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        
        except Exception as e:
            print(f"Error updating performance: {e}")
            return False
    
    def get_all_videos(self) -> list:
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data["videos"]
        except Exception as e:
            print(f"Error reading videos: {e}")
            return []
