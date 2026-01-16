from langchain_core.messages import HumanMessage
from videobrowser.core.state import AgentState, VideoResource
from videobrowser.utils.prompt_manager import load_prompt
from videobrowser.utils.parser import extract_json_from_text, extract_youtube_id
from videobrowser.utils.metrics import update_token_metrics
from videobrowser.utils.llm_factory import get_llm
from videobrowser.config import get_config
import json

def selector_node(state: AgentState):
    """
    JIT Selector:
    Selects top K videos using LLM based on metadata from raw_candidates.
    """
    print("🧠 [Selector] filtering raw candidates...")
    
    config = get_config()
    metrics = state.get("metrics", {})
    video_store = state.get("video_store", {})
    raw_candidates = state.get("raw_candidates", [])
    user_query = state.get("user_query", "")
    
    if not raw_candidates:
        print("   -> No candidates to select from.")
        return {"video_store": video_store}

    llm = get_llm(node_name="selector")
    top_k = config.selector.top_k
    
    # Format candidates for prompt
    candidates_info = ""
    valid_candidates = []
    
    for i, raw in enumerate(raw_candidates):
        url = raw.get("link", "") or raw.get("videourl", "")
        if url:
            title = raw.get("title", "Unknown Title")
            desc = raw.get("snippet", "") or raw.get("description", "")
            candidates_info += f"[{i}] Title: {title}\n    Description: {desc}\n    URL: {url}\n\n"
            
            # Create a normalized candidate object
            candidate = raw.copy()
            candidate["url"] = url
            valid_candidates.append(candidate)
            
    if not valid_candidates:
        return {"video_store": video_store}

    # Load Prompt
    prompt_text = load_prompt(
        "jit_selector.j2",
        user_query=user_query,
        top_k=top_k,
        candidates_info=candidates_info
    )
    
    target_videos = []
    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        metrics = update_token_metrics(metrics, response, category="jit_selector")
        selected_indices = extract_json_from_text(response.content)

        print(f"   -> LLM selected video indices: {selected_indices}")
        
        if isinstance(selected_indices, list):
            for idx in selected_indices:
                if isinstance(idx, int) and 0 <= idx < len(valid_candidates):
                    target_videos.append(valid_candidates[idx])
                    if len(target_videos) >= top_k:
                        break
        elif isinstance(selected_indices, dict):
             # Handle case where LLM returns a dict (rare but possible), directly process value
            values = selected_indices.values()
            for idx in values:
                if isinstance(idx, int) and 0 <= idx < len(valid_candidates):
                    target_videos.append(valid_candidates[idx])
                    if len(target_videos) >= top_k:
                        break
        else:
             print(f"      ⚠️ Invalid selection format, falling back to top {top_k}.")
             target_videos = valid_candidates[:top_k]
             
    except Exception as e:
        print(f"      ⚠️ Selection error: {e}, falling back to top {top_k}.")
        target_videos = valid_candidates[:top_k]

    if not target_videos:
         target_videos = valid_candidates[:top_k]

    print(f"   -> Selected {len(target_videos)} videos for processing.")
    
    # Update Video Store with selected candidates
    # We mark them as "candidate" so the Watcher knows what to process
    updated_store = video_store.copy()
    
    for i, video in enumerate(target_videos):
        video_url = video["url"]
        video_id = extract_youtube_id(video_url) or f"vid_{i}"
        
        if video_id not in updated_store:
            resource = VideoResource(
                video_id=video_id,
                title=video.get('title', 'Unknown'),
                url=video_url,
                duration=video.get('duration', 'Unknown'),
                status="candidate", # Ready for watcher
                summary="",
                transcript=""
            )
            updated_store[video_id] = resource
        else:
            # If already exists, ensure status allows re-processing if needed? 
            # Or just leave it. For JIT, we might want to ensure it's marked as candidate if not already verified.
            if updated_store[video_id].status not in ["verified", "watched"]:
                updated_store[video_id].status = "candidate"

    return {
        "video_store": updated_store,
        "metrics": metrics
    }