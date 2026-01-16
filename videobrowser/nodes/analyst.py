from videobrowser.core.state import AgentState
from videobrowser.tools.vision import extract_frames_from_window
from videobrowser.tools.fetch_video import download_video_file
from videobrowser.utils.prompt_manager import load_prompt
from videobrowser.utils.metrics import update_token_metrics
from videobrowser.utils.logger import get_logger
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from videobrowser.utils.llm_factory import get_llm
from videobrowser.config import get_config
import json
import time

load_dotenv()

llm = get_llm(node_name="analyst")
logger = get_logger()

def analyst_node(state: AgentState):
    """
    JIT Analyst:
    1. Reads the identified temporal windows from video_store (status="verified").
    2. Extracts 1 FPS frames for those specific windows.
    3. Feeds relevant frames and transcript segments to LLM for final answer using 'jit_analyst.j2'.
    """
    logger = get_logger()
    logger.log("JITAnalyst", "start")
    print("🧠 [Analyst] Extracting relevant clips and synthesizing final answer...")
    
    video_store = state.get("video_store", {})
    user_query = state.get("user_query", "")
    metrics = state.get("metrics", {})
    config = get_config()
    
    if not video_store:
        return {"final_answer": "No videos were successfully processed."}

    content_parts = []
    
    # 1. Gather Context (Videos & Transcript)
    has_relevant_content = False
    
    # Iterate over verified videos
    for i, (vid, res) in enumerate(video_store.items()):
        # We focus on verified videos that have window analysis
        if res.status != "verified" or not res.summary:
            continue

        # Parse the window analysis JSON
        try:
            analysis = json.loads(res.summary)
        except:
            print(f"   -> Skipping Video {i+1}: Could not parse window analysis.")
            continue
            
        if not analysis.get("relevant", False):
            continue
            
        windows = analysis.get("windows", [])
        if not windows:
            # Fallback for old format if 'windows' key missing but relevant is true
            if "start_time_seconds" in analysis:
                windows = [analysis]
            else:
                continue
        
        print(f"   -> Processing Video {i+1}: {res.title} ({len(windows)} windows)")
        
        # We re-call download_video_file which hits cache.
        video_path = download_video_file(res.url)
        if not video_path:
            print(f"      ⚠️ Could not retrieve video file for {res.title}")
            continue

        full_transcript_lines = res.transcript.splitlines() if res.transcript else []
        
        for win in windows:
            start = win.get("start_time_seconds", 0.0)
            end = win.get("end_time_seconds", 0.0)
            reason = win.get("reasoning", "")
            
            if end <= start:
                continue

            has_relevant_content = True
            
            # 1. Extract Frames at 1 FPS
            print(f"      -> Extracting clip {start:.1f}s - {end:.1f}s (1 FPS)...")
            
            # Use shared vision tool
            frames = extract_frames_from_window(video_path, start, end, fps_sample=1.0)
            
            if frames:
                content_parts.append({
                    "type": "text", 
                    "text": f"\n=== Video: {res.title} [Clip: {start:.1f}s - {end:.1f}s] ===\nReasoning: {reason}\n"
                })
                
                for f in frames:
                    content_parts.append({
                        "type": "text",
                        "text": f"[Frame at {f['timestamp']:.1f}s]"
                    })
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{f['image']}"}
                    })

            # 2. Extract Relevant Transcript
            relevant_lines = []
            for line in full_transcript_lines:
                try:
                    import re
                    match = re.search(r"((\d+\.?\d*)s\s*-\s*(\d+\.?\d*)s)", line)
                    if match:
                        t_start = float(match.group(2))
                        t_end = float(match.group(3))
                        
                        # Check overlap
                        if t_end >= start and t_start <= end:
                            relevant_lines.append(line)
                except:
                    pass
            
            if relevant_lines:
                content_parts.append({
                    "type": "text",
                    "text": "Transcript Segment:\n" + "\n".join(relevant_lines)
                })
                print(f"      -> Added {len(relevant_lines)} transcript lines.")

    if not has_relevant_content:
        print("   -> No specific visual windows identified. Falling back to full transcript analysis...")
        
        # Fallback: Use full transcripts if available
        for i, (vid, res) in enumerate(video_store.items()):
            if res.transcript:
                print(f"      -> Adding transcript for Video {i+1}: {res.title}")
                content_parts.append({
                    "type": "text", 
                    "text": f"\n=== Video Transcript: {res.title} ===\n{res.transcript[:25000]}..." # Limit to avoid context overflow if huge
                })
                has_relevant_content = True
        
        if not has_relevant_content:
            msg = "No relevant video content or transcripts found to answer the query."
            print(f"   -> {msg}")
            return {"final_answer": msg}

    # 2. Prepare Final Prompt (System/Instruction)
    # The 'jit_analyst.j2' is just the instruction wrapper.
    # We prepend the instruction to the context parts.
    
    # Actually, we can just load the prompt string and append it as a text message at the END.
    # The 'video_context' param in the j2 was my idea, but since we built content_parts dynamically above (with images),
    # we just need the instruction text at the end.
    
    instruction_text = load_prompt(
        "analyst_report.j2",
        user_query=user_query,
        video_context=[] # We passed context via content_parts
    )
    
    # We need to prepend "User Query: ..." context if not already
    # The JIT implementation added a text block at the start:
    content_parts.insert(0, {
        "type": "text", 
        "text": f"User Query: {user_query}\n\nAnalyze the following video clips and transcripts to answer the query."
    })
    
    # Add the formatted instruction at the end
    content_parts.append({
        "type": "text",
        "text": instruction_text
    })

    print("   -> Invoking Analyst LLM with video context...")
    try:
        response = llm.invoke([HumanMessage(content=content_parts)])
        metrics = update_token_metrics(metrics, response, category="jit_analyst")
        final_answer = response.content
    except Exception as e:
        final_answer = f"Error: {e}"

    return {
        "final_answer": final_answer,
        "metrics": metrics
    }