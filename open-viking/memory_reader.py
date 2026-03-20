import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import Counter

class MemoryReader:
    def __init__(self, history_path: str = None):
        if history_path is None:
            base_dir = Path(__file__).parent
            history_path = base_dir / "history" / "videos.json"
        
        self.history_path = Path(history_path)
    
    def _load_data(self) -> Dict[str, Any]:
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading data: {e}")
            return {"videos": [], "metadata": {"total_count": 0}}
    
    def get_recent_videos(self, n: int = 10) -> List[Dict[str, Any]]:
        data = self._load_data()
        videos = data.get("videos", [])
        return sorted(videos, key=lambda x: x.get("timestamp", ""), reverse=True)[:n]
    
    def check_topic_coverage(self, topic: str, days: int = 7) -> Optional[Dict[str, Any]]:
        data = self._load_data()
        videos = data.get("videos", [])
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        topic_lower = topic.lower()
        
        for video in reversed(videos):
            video_date = datetime.fromisoformat(video.get("timestamp", ""))
            if video_date < cutoff_date:
                continue
            
            video_topic = video.get("topic", "").lower()
            if topic_lower in video_topic or video_topic in topic_lower:
                return {
                    "duplicate_found": True,
                    "video_id": video.get("id"),
                    "topic": video.get("topic"),
                    "timestamp": video.get("timestamp"),
                    "days_ago": (datetime.utcnow() - video_date).days
                }
        
        return {"duplicate_found": False}
    
    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        data = self._load_data()
        videos = data.get("videos", [])
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_videos = [
            v for v in videos 
            if datetime.fromisoformat(v.get("timestamp", "")) >= cutoff_date
        ]
        
        if not recent_videos:
            return {
                "total_videos": 0,
                "avg_views": 0,
                "avg_engagement": 0.0,
                "top_topics": [],
                "top_keywords": [],
                "best_performing": None
            }
        
        total_views = sum(v.get("performance_metrics", {}).get("views", 0) for v in recent_videos)
        total_engagement = sum(v.get("performance_metrics", {}).get("engagement_rate", 0.0) for v in recent_videos)
        
        all_keywords = []
        for v in recent_videos:
            all_keywords.extend(v.get("keywords", []))
        
        keyword_counts = Counter(all_keywords)
        
        best_video = max(
            recent_videos, 
            key=lambda x: x.get("performance_metrics", {}).get("views", 0),
            default=None
        )
        
        return {
            "total_videos": len(recent_videos),
            "avg_views": total_views / len(recent_videos) if recent_videos else 0,
            "avg_engagement": total_engagement / len(recent_videos) if recent_videos else 0.0,
            "top_topics": [v.get("topic") for v in sorted(
                recent_videos, 
                key=lambda x: x.get("performance_metrics", {}).get("views", 0),
                reverse=True
            )[:5]],
            "top_keywords": [kw for kw, count in keyword_counts.most_common(10)],
            "best_performing": {
                "id": best_video.get("id"),
                "topic": best_video.get("topic"),
                "views": best_video.get("performance_metrics", {}).get("views", 0),
                "engagement_rate": best_video.get("performance_metrics", {}).get("engagement_rate", 0.0)
            } if best_video else None
        }
    
    def get_video_by_id(self, video_id: int) -> Optional[Dict[str, Any]]:
        data = self._load_data()
        videos = data.get("videos", [])
        
        for video in videos:
            if video.get("id") == video_id:
                return video
        
        return None
    
    def search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        data = self._load_data()
        videos = data.get("videos", [])
        
        keyword_lower = keyword.lower()
        
        matching_videos = [
            v for v in videos
            if keyword_lower in v.get("topic", "").lower() or
               keyword_lower in [k.lower() for k in v.get("keywords", [])]
        ]
        
        return sorted(matching_videos, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    def get_total_count(self) -> int:
        data = self._load_data()
        return data.get("metadata", {}).get("total_count", 0)
