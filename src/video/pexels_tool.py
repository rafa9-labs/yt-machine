import os
import json
from pathlib import Path
from typing import Optional
import requests
from mcp.server.fastmcp import FastMCP

server = FastMCP("pexels-tool")

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PEXELS_API_URL = "https://api.pexels.com/videos/search"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "footage"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@server.tool()
def fetch_vertical_footage(keywords: list, min_duration: int = 5) -> dict:
    """
    Fetch vertical (9:16) HD video clips from Pexels API.
    
    Args:
        keywords: List of search keywords (e.g., ["technology", "innovation"])
        min_duration: Minimum video duration in seconds (default: 5)
    
    Returns:
        dict with status, downloaded files, and metadata
    """
    
    if not PEXELS_API_KEY:
        return {
            "success": False,
            "error": "PEXELS_API_KEY not set in .env file"
        }
    
    headers = {
        "Authorization": PEXELS_API_KEY
    }
    
    downloaded_files = []
    errors = []
    
    for keyword in keywords:
        try:
            params = {
                "query": keyword,
                "per_page": 3,
                "orientation": "portrait",
                "size": "hd"
            }
            
            response = requests.get(
                PEXELS_API_URL,
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            videos = data.get("videos", [])
            
            if not videos:
                errors.append(f"No vertical videos found for keyword: {keyword}")
                continue
            
            for idx, video in enumerate(videos):
                video_files = video.get("video_files", [])
                
                hd_file = None
                for vf in video_files:
                    if vf.get("quality") == "hd" and vf.get("width") == 720:
                        hd_file = vf
                        break
                
                if not hd_file:
                    hd_file = video_files[0] if video_files else None
                
                if not hd_file:
                    continue
                
                duration = video.get("duration", 0)
                if duration < min_duration:
                    continue
                
                video_url = hd_file.get("link")
                if not video_url:
                    continue
                
                filename = f"{keyword}_{video.get('id')}_{idx}.mp4"
                filepath = OUTPUT_DIR / filename
                
                try:
                    video_response = requests.get(video_url, timeout=30, stream=True)
                    video_response.raise_for_status()
                    
                    with open(filepath, 'wb') as f:
                        for chunk in video_response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    downloaded_files.append({
                        "filename": filename,
                        "path": str(filepath),
                        "keyword": keyword,
                        "duration": duration,
                        "resolution": f"{hd_file.get('width')}x{hd_file.get('height')}",
                        "pexels_id": video.get("id")
                    })
                    
                except Exception as e:
                    errors.append(f"Failed to download {filename}: {str(e)}")
        
        except requests.exceptions.RequestException as e:
            errors.append(f"API error for keyword '{keyword}': {str(e)}")
        except Exception as e:
            errors.append(f"Unexpected error for keyword '{keyword}': {str(e)}")
    
    return {
        "success": len(downloaded_files) > 0,
        "downloaded_count": len(downloaded_files),
        "files": downloaded_files,
        "errors": errors if errors else None,
        "output_directory": str(OUTPUT_DIR)
    }
