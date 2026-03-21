"""
Category Rotation System - Implements 4-day geopolitical topic rotation
Ensures diverse coverage: Middle East → Great Power → Economic → Regional
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


class CategoryRotation:
    """
    Manages 4-day rotation of geopolitical categories to ensure topic diversity.
    Tracks video history and applies rotation boosts to article scoring.
    """
    
    def __init__(self):
        self.rotation_file = Path("output/category_rotation.json")
        self.history_file = Path("output/video_history.json")
        self.categories = [
            "middle_east_conflict",
            "great_power_competition", 
            "economic_warfare",
            "regional_flashpoints"
        ]
        self._ensure_rotation_file()
        self._ensure_history_file()
    
    def _ensure_rotation_file(self):
        """Create rotation tracking file if it doesn't exist"""
        if not self.rotation_file.exists():
            self.rotation_file.parent.mkdir(parents=True, exist_ok=True)
            rotation_data = {
                "current_cycle": {
                    "day_index": 0,
                    "category": "middle_east_conflict",
                    "date": datetime.now().strftime("%Y-%m-%d")
                },
                "history": []
            }
            with open(self.rotation_file, 'w') as f:
                json.dump(rotation_data, f, indent=2)
    
    def _ensure_history_file(self):
        """Create video history tracking file if it doesn't exist"""
        if not self.history_file.exists():
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            history_data = {
                "videos": [],
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
    
    def get_today_category(self) -> str:
        """
        Get today's category based on day of year % 4 rotation.
        
        Returns:
            Today's geopolitical category string
        """
        # Load current rotation state
        with open(self.rotation_file, 'r') as f:
            rotation_data = json.load(f)
        
        today = datetime.now()
        current_date = today.strftime("%Y-%m-%d")
        
        # Check if we need to advance to next day
        last_date = rotation_data["current_cycle"]["date"]
        if current_date != last_date:
            # Advance to next category
            day_index = (rotation_data["current_cycle"]["day_index"] + 1) % 4
            category = self.categories[day_index]
            
            # Update rotation data
            rotation_data["current_cycle"] = {
                "day_index": day_index,
                "category": category,
                "date": current_date
            }
            rotation_data["history"].append({
                "date": last_date,
                "category": rotation_data["current_cycle"]["category"]
            })
            
            # Keep only last 30 days of history
            if len(rotation_data["history"]) > 30:
                rotation_data["history"] = rotation_data["history"][-30:]
            
            with open(self.rotation_file, 'w') as f:
                json.dump(rotation_data, f, indent=2)
        else:
            category = rotation_data["current_cycle"]["category"]
        
        return category
    
    def boost_category_score(self, article: Dict[str, Any], matched_categories: List[str]) -> int:
        """
        Add +10 points if article matches today's category.
        
        Args:
            article: Article dictionary with title/summary
            matched_categories: List of categories the article matches
            
        Returns:
            Score boost amount (typically 0 or 10)
        """
        today_category = self.get_today_category()
        
        if today_category in matched_categories:
            return 10  # Boost today's category by 10 points
        
        return 0
    
    def detect_article_categories(self, article: Dict[str, Any]) -> List[str]:
        """
        Detect which geopolitical categories an article belongs to.
        
        Args:
            article: Article dictionary with title and summary
            
        Returns:
            List of matching category names
        """
        from .scraper_config import GEOPOLITICAL_KEYWORDS
        
        title_lower = article.get("title", "").lower()
        summary_lower = article.get("summary", "").lower()
        combined = title_lower + " " + summary_lower
        
        matched_categories = []
        
        for category, keywords in GEOPOLITICAL_KEYWORDS.items():
            # Check if any keywords from this category appear
            for keyword in keywords:
                if keyword.lower() in combined:
                    matched_categories.append(category)
                    break  # Only need one match per category
        
        return matched_categories
    
    def track_video_generated(self, topic: str, category: str, region: str, article_title: str):
        """
        Track video generation to enforce diversity rules.
        
        Args:
            topic: Video topic
            category: Geopolitical category
            region: Geographic region
            article_title: Original article title
        """
        with open(self.history_file, 'r') as f:
            history_data = json.load(f)
        
        video_entry = {
            "project_id": int(datetime.now().timestamp()),
            "topic": topic,
            "category": category,
            "region": region,
            "article_title": article_title,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        history_data["videos"].append(video_entry)
        history_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Keep only last 20 videos
        if len(history_data["videos"]) > 20:
            history_data["videos"] = history_data["videos"][-20:]
        
        with open(self.history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def is_topic_overused(self, article_title: str) -> bool:
        """
        Check if a topic is overused in recent videos.
        
        Args:
            article_title: New article title to check
            
        Returns:
            True if topic should be penalized for repetition
        """
        with open(self.history_file, 'r') as f:
            history_data = json.load(f)
        
        recent_videos = history_data["videos"][-10:]  # Last 10 videos
        
        # Count Iran/Middle East videos
        iran_count = 0
        for video in recent_videos:
            if any(keyword in video["article_title"].lower() 
                   for keyword in ["iran", "israel", "gaza", "hezbollah"]):
                iran_count += 1
        
        # Penalize if more than 3 Iran videos in last 10
        if iran_count >= 3 and any(keyword in article_title.lower() 
                                  for keyword in ["iran", "israel", "gaza", "hezbollah"]):
            return True
        
        # Check for consecutive videos from same region
        if len(recent_videos) >= 2:
            last_region = recent_videos[-1]["region"]
            second_last_region = recent_videos[-2]["region"]
            
            # Detect region from current article
            current_region = self._detect_region_from_title(article_title)
            
            if current_region == last_region == second_last_region:
                return True
        
        return False
    
    def _detect_region_from_title(self, title: str) -> str:
        """Detect geographic region from article title"""
        title_lower = title.lower()
        
        if any(keyword in title_lower for keyword in ["iran", "israel", "gaza", "syria", "iraq", "saudi", "uae"]):
            return "middle_east"
        elif any(keyword in title_lower for keyword in ["china", "taiwan", "russia", "ukraine", "nato"]):
            return "great_power"
        elif any(keyword in title_lower for keyword in ["sanctions", "tariff", "trade", "oil", "energy"]):
            return "economic"
        elif any(keyword in title_lower for keyword in ["africa", "venezuela", "mexico", "korea", "india"]):
            return "regional"
        else:
            return "other"
    
    def get_rotation_status(self) -> Dict[str, Any]:
        """
        Get current rotation status for debugging.
        
        Returns:
            Dictionary with rotation information
        """
        with open(self.rotation_file, 'r') as f:
            rotation_data = json.load(f)
        
        return {
            "today_category": self.get_today_category(),
            "current_cycle": rotation_data["current_cycle"],
            "categories": self.categories,
            "rotation_sequence": [
                "Day 0: middle_east_conflict",
                "Day 1: great_power_competition", 
                "Day 2: economic_warfare",
                "Day 3: regional_flashpoints"
            ]
        }
