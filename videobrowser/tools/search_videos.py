import os
import json
import requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults, YouTubeSearchTool,  TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from youtube_search import YoutubeSearch as YTScraper

load_dotenv()

def duckduckgo_search(query: str) -> str:
    wrapper = DuckDuckGoSearchAPIWrapper(
        region="us-en",    # cn-zh (china), wt-wt (world), us-en (us)
        time="y",
        max_results=5,
        source="text"  # text | news | images
    )

    search_tool_json = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="json")
    result_json = search_tool_json.invoke(query)

    return result_json


def youtube_search(query: str) -> list:
    """
    Uses youtube-search library to find videos with rich metadata.
    Returns a list of dicts aligned with serper_search structure.
    """
    try:
        results = YTScraper(query, max_results=10).to_dict()
        
        candidates = []
        for v in results:
            # Construct full URL
            suffix = v.get('url_suffix', '')
            link = f"https://www.youtube.com{suffix}" if suffix else v.get('link', '')
            
            # Get thumbnail (usually a list in this library, take first)
            thumbnails = v.get("thumbnails", [])
            image_url = thumbnails[0] if isinstance(thumbnails, list) and thumbnails else ""

            candidates.append({
                "title": v.get("title"),
                "link": link,
                "snippet": v.get("long_desc") or v.get("title"), # long_desc is often the snippet
                "duration": v.get("duration", "unknown"),
                "imageurl": image_url,
                "videourl": link,
                "source": "YouTube",
                "channel": v.get("channel", "unknown"),
                "date": v.get("publish_time", "unknown"),
                "position": "unknown" 
            })
        return candidates

    except Exception as e:
        print(f"❌ [YouTube] Request failed: {e}")
        return []


def serper_search(query: str) -> list:
    """
    Uses Serper.dev API (official HTTP endpoint) to find videos.
    Returns a list of dicts: [{'title': ..., 'link': ..., 'snippet': ...}]
    """
    url = "https://google.serper.dev/videos"
    
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("⚠️ [Serper] Missing SERPER_API_KEY. Returning empty list.")
        return []

    payload = json.dumps({
        "q": query,
        "num": 10  # Request up to 10 videos
    })
    
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    candidates = []
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
        
        # Parse 'videos' key from response
        if 'videos' in data:
            for v in data['videos']:
                candidates.append({
                    "title": v.get("title"),
                    "link": v.get("link"),
                    "snippet": v.get("snippet", "") or v.get("title"),
                    "duration": v.get("duration", "unknown"),
                    "imageurl": v.get("imageUrl", ""),
                    "videourl": v.get("videoUrl", ""),
                    "source": v.get("source", "unknown"),
                    "channel": v.get("channel", "unknown"),
                    "date": v.get("date", "unknown"),
                    "position": v.get("position", "unknown"),
                })
        else:
            print(f"⚠️ [Serper] No 'videos' key in response for query: {query}")

    except Exception as e:
        print(f"❌ [Serper] Request failed: {e}")
            
    return candidates

def serper_web_search(query: str) -> list:
    """
    Uses Serper.dev API (official HTTP endpoint) to find web.
    Returns a list of dicts: [{'title': ..., 'link': ..., 'snippet': ...}]
    """
    url = "https://google.serper.dev/search"
    
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("⚠️ [Serper] Missing SERPER_API_KEY. Returning empty list.")
        return []

    payload = json.dumps({
        "q": query
    })
    
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    candidates = []
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
        
        # Parse 'organic' key from response
        if 'organic' in data:
            for v in data['organic']:
                candidates.append({
                    "title": v.get("title"),
                    "link": v.get("link"),
                    "snippet": v.get("snippet", "") or v.get("title"),
                    "position": v.get("position", "unknown")
                })
        else:
            print(f"⚠️ [Serper] No 'organic' key in response for query: {query}")

    except Exception as e:
        print(f"❌ [Serper] Request failed: {e}")
            
    return candidates



def tavily_search(query: str) -> list:
    """
    Uses Tavily API to perform a text search.
    Returns a list of dicts: [{'url': ..., 'content': ...}]
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("⚠️ [Tavily] Missing TAVILY_API_KEY. Returning empty list.")
        return []

    tavily_tool = TavilySearchResults(max_results=5) # Default to 5 results
    results = tavily_tool.invoke(query)
    
    # TavilySearchResults.invoke returns a list
    try:
        # results = json.loads(result_json)
        # Normalize to a list of dicts with 'title', 'link', 'snippet' for consistency
        normalized_results = []
        for r in results:
            normalized_results.append({
                "title": r.get("title", "No Title"),
                "link": r.get("url", "#"),
                "snippet": r.get("content", "No content snippet available.")
            })
        return normalized_results
    except Exception as e:
        print(f"❌ [Tavily] Error parsing results: {e}")
        return []


if __name__ == "__main__":
    query = "how to cook pasta"
    print(f"Testing Serper with query: {query}")
    results = tavily_search(query)
    print(f"\nFound {len(results)} results:")
    for res in results:
        # print(f"- {res['title']} ({res['duration']})")
        print(f"- {res['title']} ({res['link']})")
