# utils/youtube_recommender.py
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import yt_dlp; if missing, we'll fall back to youtube-search-python
try:
    import yt_dlp as _yt_dlp
    _HAS_YT_DLP = True
except Exception:
    _yt_dlp = None
    _HAS_YT_DLP = False

def _fallback_search(query: str) -> Optional[dict]:
    """Fallback search using youtube-search-python (VideosSearch)"""
    try:
        from youtubesearchpython import VideosSearch
        vs = VideosSearch(query, limit=1)
        res = vs.result()
        items = res.get('result') or []
        if not items:
            return None
        item = items[0]
        return {
            'url': item.get('link'),
            'thumbnail': (item.get('thumbnails') or [{}])[0].get('url') if item.get('thumbnails') else None,
            'title': item.get('title')
        }
    except Exception as e:
        logger.debug(f"Fallback YouTube search failed: {e}")
        return None


def recommend_youtube_video(query: str) -> dict:
    """Return a small dict with url, thumbnail, and title for a YouTube video matching `query`.

    Tries to use yt_dlp if available for reliable extraction; otherwise falls back to
    youtube-search-python. If both fail, returns a default motivational video.
    """
    default = {
        "url": "https://www.youtube.com/watch?v=mgmVOuLgFB0",
        "thumbnail": "https://img.youtube.com/vi/mgmVOuLgFB0/maxresdefault.jpg",
        "title": "Eye of the Tiger - Survivor"
    }

    if _HAS_YT_DLP and _yt_dlp is not None:
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'format': 'best',
                'skip_download': True,
                'extract_flat': True
            }
            with _yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(f"ytsearch1:{query}", download=False)
                if search_results and 'entries' in search_results and search_results['entries']:
                    video = search_results['entries'][0]
                    url = video.get('webpage_url') or video.get('url')
                    thumbnail = video.get('thumbnail') or default['thumbnail']
                    title = video.get('title') or None
                    return {'url': url, 'thumbnail': thumbnail, 'title': title}
        except Exception as e:
            logger.debug(f"yt_dlp search failed: {e}")

    # Fallback to youtube-search-python
    fb = _fallback_search(query)
    if fb:
        return { 'url': fb.get('url') or default['url'], 'thumbnail': fb.get('thumbnail') or default['thumbnail'], 'title': fb.get('title') }

    return default