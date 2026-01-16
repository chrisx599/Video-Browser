import json
import base64
import os
from langchain_core.messages import HumanMessage
from videobrowser.core.state import AgentState, VideoResource
from videobrowser.utils.prompt_manager import load_prompt
from videobrowser.utils.parser import extract_youtube_id, extract_json_from_text
from videobrowser.utils.llm_factory import get_llm
from videobrowser.tools.fetch_video import fetch_transcript_with_timestamps, download_video_file
from videobrowser.tools.vision import extract_frames_with_timestamps
from videobrowser.config import get_config
from videobrowser.utils.metrics import update_token_metrics
from videobrowser.utils.logger import get_logger

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None

def watcher_node(state: AgentState):
    """
    JIT Watcher:
    1. Iterates over videos with status="candidate".
    2. For each video:
       - Downloads video & transcript (with timestamps).
       - Extracts sparse frames (e.g. 16) with timestamps.
       - Uses VLM to identify the RELEVANT TEMPORAL WINDOWS using 'jit_watcher_window.j2'.
    3. Stores the window info in video_store (status="verified").
    """
    logger = get_logger()
    logger.log("JITWatcher", "start")
    print("🎥 [Watcher] Starting sparse sampling and window identification...")
    
    config = get_config()
    metrics = state.get("metrics", {})
    video_store = state.get("video_store", {})
    user_query = state.get("user_query", "")
    
    # 1. Identify Candidates (set by Selector)
    candidates = [
        v for v in video_store.values() 
        if v.status == "candidate"
    ]
    
    if not candidates:
        print("   -> No candidates to watch.")
        return {"video_store": video_store}

    llm = get_llm(node_name="watcher") 
    
    for i, video in enumerate(candidates):
        print(f"   -> Processing Video {i+1}/{len(candidates)}: {video.title}")
        video_url = video.url
        
        # 1. Download & Vision (Sparse Sampling)
        frames_data = []
        video_path = None
        try:
            video_path = download_video_file(video_url)
            if video_path:
                num_frames = config.watcher.num_frames  # e.g., 16
                frames_data = extract_frames_with_timestamps(video_path, num_frames=num_frames)
                print(f"      -> Extracted {len(frames_data)} frames with timestamps.")
        except Exception as e:
            print(f"      ⚠️ Vision error: {e}")

        # 2. Transcript with Timestamps
        transcript_segments = []
        transcript_text_with_timestamps = ""
        try:
            transcript_segments = fetch_transcript_with_timestamps(video_url)
            if transcript_segments:
                # Track Whisper Usage (JIT specific approximation using video duration)
                if config.transcript.provider == "whisper" and VideoReader and video_path and os.path.exists(video_path):
                     try:
                         vr = VideoReader(video_path, ctx=cpu(0))
                         # Calculate duration in seconds
                         duration = len(vr) / vr.get_avg_fps()
                         metrics["whisper_audio_seconds"] = metrics.get("whisper_audio_seconds", 0.0) + duration
                     except Exception as e:
                         print(f"      ⚠️ Could not calculate duration for Whisper metric: {e}")
                
                # Format for VLM prompt: [00:00 - 00:05] Text...
                lines = []
                for seg in transcript_segments:
                    start = seg.get('start', 0)
                    end = seg.get('end', 0)
                    text = seg.get('text', '')
                    lines.append(f"[{start:.1f}s - {end:.1f}s] {text}")
                
                transcript_text_with_timestamps = "\n".join(lines)
                print(f"      -> Fetched transcript ({len(transcript_segments)} segments).")
            else:
                print("      -> No transcript found.")
        except Exception as e:
            print(f"      ⚠️ Transcript error: {e}")

        # 3. Identify Temporal Window using VLM
        # Prepare frame descriptions
        frame_descriptions = "\n".join([f"Frame {idx+1}: Timestamp {f['timestamp']:.2f}s" for idx, f in enumerate(frames_data)])
        
        # Truncate transcript if too long (approx 20k chars)
        truncated_transcript = transcript_text_with_timestamps[:25000]
        
        # Load Prompt
        prompt_text = load_prompt(
            "jit_watcher_window.j2",
            user_query=user_query,
            video_title=video.title,
            num_frames=len(frames_data),
            frame_descriptions=frame_descriptions,
            transcript=truncated_transcript
        )
        
        content_parts = [{"type": "text", "text": prompt_text}]
        for f in frames_data:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f['image']}"}
            })
            
        print(f"      -> Identifying relevant window with VLM...")
        analysis_result = {}
        summary_text = "Analysis failed."
        
        try:
            response = llm.invoke([HumanMessage(content=content_parts)])
            metrics = update_token_metrics(metrics, response, category="jit_watcher")
            raw_response = response.content
            
            # Robust JSON extraction
            try:
                analysis_result = extract_json_from_text(raw_response)
                summary_text = json.dumps(analysis_result, indent=2)
            except Exception as e:
                print(f"      ⚠️ JSON parsing error: {e}")
                summary_text = raw_response

            print(f"      -> Window identified: {summary_text[:100]}...")
            
        except Exception as e:
            print(f"      ⚠️ LLM error: {e}")
            summary_text = f"Error: {e}"

        # Update VideoResource
        video.status = "verified"
        video.summary = summary_text
        video.transcript = transcript_text_with_timestamps
        
        video_store[video.video_id] = video

    return {
        "video_store": video_store,
        "metrics": metrics
    }