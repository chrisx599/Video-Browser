import operator
from typing import List, Dict, Optional, Any, Annotated
from pydantic import BaseModel, Field

from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage



class EvidenceFragment(BaseModel):
    """Single piece of evidence from a video"""
    source: str = Field(..., description="source: 'transcript' | 'visual'")
    content: str = Field(..., description="evidence content")
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    confidence: float = 0.0

class VideoResource(BaseModel):
    """Video resource with metadata and evidence"""
    video_id: str
    title: str
    url: str
    duration: str
    status: str = "candidate"  # candidate | analyzing | verified | rejected
    relevance_reason: str = ""
    evidence: List[EvidenceFragment] = []
    summary: Optional[str] = None
    transcript: Optional[str] = None

# ==========================================
# 2. AgentState 定义 (使用 MessagesState 模板)
# ==========================================

class InputState(MessagesState):
    """
    定义图的输入参数。
    User Query 会被包装进 messages[0] (HumanMessage)
    """
    # 也可以显式定义 query，方便提取，不完全依赖 messages
    user_query: str 

class AgentState(MessagesState):
    """
    继承自 MessagesState。
    它自动包含: messages: Annotated[list[AnyMessage], add_messages]
    """
    
    # --- Sector 1: Global Context ---
    user_query: str
    constraints: List[str] = []
    
    # --- Sector 2: The Blackboard (Context Engine) ---
    video_store: Dict[str, VideoResource] = {}
    
    # --- Sector 3: Scratchpad (short) ---
    current_search_queries: List[str] = []
    visual_hypothesis: Optional[str] = None
    raw_candidates: List[Dict[str, Any]] = []
    ambiguity_note: Optional[str] = None
    
    # --- Sector 4: Execution Log (history) ---
    # 这里我们仍然需要 operator.add 来保留 Planner 的尝试历史
    tried_queries: Annotated[List[str], operator.add] = []
    visited_video_ids: Annotated[List[str], operator.add] = []
    text_search_context: Annotated[List[str], operator.add] = []
    
    # Planner 的思考轨迹 (Text Trace)
    plan_trace: Annotated[List[str], operator.add] = []
    
    # --- Sector 5: Control ---
    loop_step: int = 0
    final_answer: Optional[str] = None
    routing_signal: str = "planner" # 默认为 planner

    # --- Sector 6: Metrics ---
    metrics: Annotated[Dict[str, Any], operator.ior] = {}

# ==========================================
# 3. 辅助函数 (View Helpers)
#    适配 LangGraph 的 State 访问方式
# ==========================================

def get_latest_human_message(state: AgentState) -> str:
    """从标准 messages 列表中提取最后一条用户消息"""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            return msg.content
    return state.get("user_query", "")

def format_planner_view(state: AgentState) -> str:
    """
    Context Engineering: Render the view for the Planner.
    Includes:
    - User Goal & Constraints
    - Search History (to avoid loops)
    - Current Knowledge (summarized from video evidence)
    - Text Context (from web searches)
    """
    # Extract Blackboard Summary
    blackboard_summary = ""
    
    # Organize videos by status for a more structured view
    verified_videos = []
    watched_videos = [] # Watched but not yet verified or rejected
    candidate_videos = []
    rejected_videos = []

    if state.get("video_store"):
        for vid, res in state["video_store"].items():
            if res.status == "verified":
                verified_videos.append(res)
            elif res.status == "watched":
                watched_videos.append(res)
            elif res.status == "candidate":
                candidate_videos.append(res)
            elif res.status == "rejected":
                rejected_videos.append(res)

    if not state.get("video_store") or (not verified_videos and not watched_videos and not candidate_videos and not rejected_videos):
        blackboard_summary += "No videos analyzed yet.\n\n"
    else:
        # Verified Videos - This is what we know
        if verified_videos:
            blackboard_summary += "=== ✅ Verified Knowledge ===\n"
            for res in verified_videos:
                blackboard_summary += f"- Title: {res.title} (ID: {res.video_id})\n"
                if res.summary: # Prioritize video summary if available
                    summary_preview = res.summary
                    blackboard_summary += f"  Summary: {summary_preview}\n"
                elif res.evidence:
                    blackboard_summary += "  Key Evidence Snippets:\n"
                    for i, ev in enumerate(res.evidence[:3]): # Show top 3 snippets
                        content_preview = ev.content[:150].replace('\n', ' ') + "..." if len(ev.content) > 150 else ev.content.replace('\n', ' ')
                        blackboard_summary += f"    - {content_preview}\n"
                blackboard_summary += "\n"
        
        # Watched Videos - Information extracted, but not yet verified
        if watched_videos:
            blackboard_summary += "=== ⏹️ Watched Videos (Awaiting Verification) ===\n"
            for res in watched_videos:
                blackboard_summary += f"- Title: {res.title} (ID: {res.video_id})\n"
                if res.summary:
                    summary_preview = res.summary[:200] + "..." if len(res.summary) > 200 else res.summary
                    blackboard_summary += f"  Summary: {summary_preview}\n"
                elif res.evidence:
                    blackboard_summary += "  Extracted Evidence Snippets:\n"
                    for i, ev in enumerate(res.evidence[:2]): # Show top 2 snippets
                        content_preview = ev.content[:150].replace('\n', ' ') + "..." if len(ev.content) > 150 else ev.content.replace('\n', ' ')
                        blackboard_summary += f"    - {content_preview}\n"
                blackboard_summary += "\n"

        # Candidate Videos - Search results not yet processed
        if candidate_videos:
            blackboard_summary += "=== ⏳ Candidate Videos (From Search Results) ===\n"
            for res in candidate_videos:
                blackboard_summary += f"- Title: {res.title} (ID: {res.video_id})\n"
                blackboard_summary += "\n"

        # Rejected Videos - What we've discarded and why
        if rejected_videos:
            blackboard_summary += "=== ❌ Rejected Videos (Reasons for Discard) ===\n"
            for res in rejected_videos:
                blackboard_summary += f"- Title: {res.title} (ID: {res.video_id})\n"
                blackboard_summary += f"  Reason: {res.relevance_reason if res.relevance_reason else 'No specific reason provided.'}\n"
                blackboard_summary += "\n"
    
    # Extract History
    history_str = "None"
    if state.get("tried_queries"):
        history_str = "\n".join([f"- {q}" for q in state["tried_queries"]])
        
    # Extract Text Context
    text_context_str = "None"
    if state.get("text_search_context"):
        # We'll just show the first few lines of each text snippet to save space
        text_context_str = "\n".join([f"- {s[:150]}..." for s in state["text_search_context"]])
    
    return f"""
    Current Task: Plan the next search queries to fulfill the user's request.

    User Goal: {state.get('user_query')}
    Constraints: {state.get('constraints')}

    Search History (Previously attempted queries):
    {history_str}
    
    Text Context (Web Search Results):
    {text_context_str}

    Current Knowledge Status:
    {blackboard_summary}
    """