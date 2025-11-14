# utils/video_utils.py
import os
import logging
import asyncio
from typing import List, Dict, Optional
from urllib.parse import quote_plus
import aiohttp
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

logger = logging.getLogger(__name__)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # Set API key in environment
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# -------------------------
# YouTube API Integration
# -------------------------
def get_youtube_service():
    if not YOUTUBE_API_KEY:
        logger.warning("YouTube API key not found. Falling back to web scraping.")
        return None
    # Import googleapiclient lazily so the module can be imported even when
    # the google-api-python-client package isn't installed (we can fallback
    # to web scraping in that case).
    try:
        from googleapiclient.discovery import build
    except Exception:
        logger.warning("googleapiclient not available; falling back to web scraping.")
        return None

    try:
        return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize YouTube API client: {e}")
        return None

def fetch_videos_from_api(query: str, max_results: int = 5) -> List[Dict]:
    """Fetch videos using official YouTube API."""
    service = get_youtube_service()
    if not service:
        return []

    try:
        response = service.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results,
            order="relevance",
            safeSearch="strict"
        ).execute()

        videos = []
        for item in response.get("items", []):
            vid_id = item["id"]["videoId"]
            snippet = item["snippet"]
            videos.append({
                "video_id": vid_id,
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", "")
            })
        return videos

    except Exception as e:
        logger.error(f"YouTube API fetch failed: {e}")
        return []

# -------------------------
# Web Scraping Fallback
# -------------------------
async def fetch_videos_from_web(query: str, max_results: int = 5) -> List[Dict]:
    """Fallback using web scraping if API fails."""
    search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    videos = []
    if BeautifulSoup is None:
        logger.warning("bs4 (BeautifulSoup) not installed; cannot perform web scraping fallback for YouTube.")
        return []

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as resp:
                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")
                for video_tag in soup.find_all("a", href=True):
                    href = video_tag['href']
                    if href.startswith("/watch?v=") and len(videos) < max_results:
                        video_id = href.split("v=")[-1]
                        title = video_tag.get("title") or "Untitled"
                        videos.append({
                            "video_id": video_id,
                            "url": f"https://www.youtube.com/watch?v={video_id}",
                            "title": title,
                            "description": "",
                            "thumbnail": f"https://img.youtube.com/vi/{video_id}/0.jpg"
                        })
        return videos
    except Exception as e:
        logger.error(f"YouTube web fetch failed: {e}")
        return []

# -------------------------
# Dynamic Video Selector
# -------------------------
async def select_video_for_emotion(
    emotion: str,
    topic: Optional[str] = None,
    max_results: int = 3,
    prefer_api: bool = True
) -> Optional[Dict]:
    """
    Dynamically select a motivational video based on emotion, topic, or story context.
    Tries API first, then web scraping fallback.
    """
    search_terms = [emotion]
    if topic:
        search_terms.append(topic)
    query = " ".join(search_terms) + " motivation inspiration"

    videos = []
    if prefer_api:
        videos = fetch_videos_from_api(query, max_results=max_results)

    if not videos:
        videos = await fetch_videos_from_web(query, max_results=max_results)

    if videos:
        # Optionally, prioritize videos with higher relevance, thumbnails, or keyword matches
        return videos[0]  # return the top ranked video
    return None

# -------------------------
# Example Async Usage
# -------------------------
if __name__ == "__main__":
    import asyncio
    async def test():
        video = await select_video_for_emotion("grief", "overcoming loss")
        if video:
            print(f"Title: {video['title']}")
            print(f"URL: {video['url']}")
            print(f"Thumbnail: {video['thumbnail']}")
        else:
            print("No video found.")

    asyncio.run(test())
