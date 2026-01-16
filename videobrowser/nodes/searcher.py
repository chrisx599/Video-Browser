from videobrowser.core.state import AgentState
from videobrowser.tools.search_videos import serper_search, youtube_search, serper_web_search, tavily_search
from videobrowser.utils.parser import extract_json_from_text, extract_youtube_id
from videobrowser.config import get_config
from videobrowser.utils.logger import get_logger

logger = get_logger()

def searcher_node(state: AgentState):
    """
    Searcher Node:
    Receives queries from Planner.
    Executes searches using configured providers for text and video.
    Updates 'raw_candidates' in the state.
    """
    logger.log("Searcher", "start", {"loop_step": state.get("loop_step", 0)})
    
    queries = state.get("current_search_queries", [])
    if not queries:
        print("⚠️ [Searcher] No queries provided. Skipping.")
        logger.log("Searcher", "skip", {"reason": "no_queries"})
        return {"raw_candidates": []}

    print(f"🕵️ [Searcher] executing {len(queries)} queries: {queries}")
    logger.log("Searcher", "queries", {"queries": queries})
    
    config = get_config()
    text_provider = config.search.text_search_provider
    video_provider = config.search.video_search_provider
    
    all_video_candidates = []
    text_results_list = []
    
    for q in queries:
        # Sanitize query: Remove quotes which can break the youtube-search scraper
        clean_q = q.replace('"', '')
        
        # --- Video Search ---
        if video_provider == "youtube":
            try:
                yt_results = youtube_search(clean_q)
                print(f"   -> YouTube found {len(yt_results)} results for '{clean_q}'")
                all_video_candidates.extend(yt_results)
            except Exception as e:
                print(f"   -> YouTube tool error: {e}")
                logger.log("Searcher", "error", {"provider": "youtube", "query": clean_q, "error": str(e)}, level="ERROR")
        
        elif video_provider == "serper":
            try:
                # We append "video" to ensure we get video-heavy results if the query is generic
                # Or rely on Serper's video tab logic in our tool wrapper if implemented
                serper_results = serper_search(f"{clean_q} video")
                print(f"   -> Serper found {len(serper_results)} video results for '{clean_q}'")
                all_video_candidates.extend(serper_results)
            except Exception as e:
                print(f"   -> Serper video search error: {e}")
                logger.log("Searcher", "error", {"provider": "serper_video", "query": clean_q, "error": str(e)}, level="ERROR")
                
        # --- Text Web Search (Optional) ---
        if text_provider == "tavily":
            try:
                tavily_results = tavily_search(q)
                print(f"   -> Tavily found {len(tavily_results)} web results for '{q}'")
                # Format: "Title: <title>\nURL: <url>\nContent: <content>"
                for res in tavily_results:
                     text_results_list.append(
                         f"Source: {res.get('title', 'Unknown')} ({res.get('url', 'No URL')})\nContent: {res.get('content', '')}"
                     )
            except Exception as e:
                print(f"   -> Tavily search error: {e}")
                logger.log("Searcher", "error", {"provider": "tavily", "query": q, "error": str(e)}, level="ERROR")
                
        elif text_provider == "serper":
            try:
                serper_web_results = serper_web_search(q)
                print(f"   -> Serper found {len(serper_web_results)} web results for '{q}'")
                for res in serper_web_results:
                     # Adapt fields based on serper_web_search output structure (assumed similar to tavily or standard serper)
                     # Serper usually returns 'snippet'
                     text_results_list.append(
                         f"Source: {res.get('title', 'Unknown')} ({res.get('link', 'No URL')})\nContent: {res.get('snippet', '')}"
                     )
            except Exception as e:
                print(f"   -> Serper web search error: {e}")
                logger.log("Searcher", "error", {"provider": "serper_web", "query": q, "error": str(e)}, level="ERROR")

    # Deduplicate Video Candidates by Link
    unique_candidates = {}
    for cand in all_video_candidates:
        link = cand.get("link")
        if link:
            # Use video ID for deduplication if possible
            # This handles cases where different query params (e.g. &t=, &pp=) make links unique but they point to the same video
            video_id = extract_youtube_id(link)
            
            # Store the extracted ID
            cand["id"] = video_id
            
            if video_id not in unique_candidates:
                unique_candidates[video_id] = cand
            
    final_video_list = list(unique_candidates.values())
    print(f"✅ [Searcher] Found {len(final_video_list)} unique video candidates.")
    
    # Deduplicate Text Results (simple string dedupe)
    final_text_context = list(set(text_results_list))
    if final_text_context:
        print(f"✅ [Searcher] Found {len(final_text_context)} text search snippets.")

    logger.log("Searcher", "end", {
        "unique_videos_found": len(final_video_list),
        "text_snippets_found": len(final_text_context)
    })

    return {
        "raw_candidates": final_video_list,
        "text_search_context": final_text_context,
        # Append queries to history
        "tried_queries": queries 
    }
