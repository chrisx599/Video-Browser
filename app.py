import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from typing import Dict, Any, List

from videobrowser.graph.builder import build_graph
from videobrowser.core.state import AgentState, VideoResource

# ==============================================================================
# 1. Initialization (Chat Start)
# ==============================================================================

@cl.on_chat_start
async def start():
    """
    Initialize the LangGraph agent when a new chat session starts.
    """
    # 1. Build the graph
    graph = build_graph()
    
    # 2. Store graph and config in the user session
    # We use the session ID as the thread ID for conversation persistence/memory
    cl.user_session.set("graph", graph)
    cl.user_session.set("config", {"configurable": {"thread_id": cl.user_session.get("id")}})
    
    # 3. Welcome message
    await cl.Message(
        content="""👋 **Welcome to VideoBrowser Agent!**

I can research video content from YouTube to answer your questions. 

**Example queries:**
- *Find a video explaining how a Transformer model works and summarize the key mechanism.*
- *Find a recipe for braised pork and list the ingredients.*""",
        author="System"
    ).send()

# ==============================================================================
# 2. Message Handling (Main Loop)
# ==============================================================================

@cl.on_message
async def main(message: cl.Message):
    """
    Handle user input, run the graph, and stream updates to the UI.
    """
    graph = cl.user_session.get("graph")
    config = cl.user_session.get("config")
    
    # Input for the graph
    inputs = {"user_query": message.content}
    
    # We'll use a dictionary to keep track of active steps for each node
    # so we can update them or close them properly.
    active_steps: Dict[str, cl.Step] = {}
    
    # Stream the graph execution
    # astream returns an async iterator yielding dictionaries: {node_name: node_output_state}
    async for output in graph.astream(inputs, config=config):
        for node_name, node_state in output.items():
            
            # --- 2.1 Planner Node ---
            if node_name == "planner":
                await handle_planner(node_state)

            # --- 2.2 Searcher Node ---
            elif node_name == "searcher":
                await handle_searcher(node_state)

            # --- 2.3 Selector Node ---
            elif node_name == "selector":
                await handle_selector(node_state)

            # --- 2.4 Watcher Node ---
            elif node_name == "watcher":
                await handle_watcher(node_state)

            # --- 2.5 Checker Node ---
            elif node_name == "checker":
                await handle_checker(node_state)

            # --- 2.6 Analyst Node (Final Answer) ---
            elif node_name == "analyst":
                await handle_analyst(node_state)


# ==============================================================================
# 3. Node Handlers (Visualization Logic)
# ==============================================================================

async def handle_planner(state: dict):
    """Visualize the Planner's thought process."""
    if "plan_trace" in state and state["plan_trace"]:
        latest_plan = state["plan_trace"][-1]
        async with cl.Step(name="Planner", type="run") as step:
            step.input = "Analyzing request and planning actions..."
            step.output = latest_plan
    
    if "current_search_queries" in state and state["current_search_queries"]:
         async with cl.Step(name="Generated Queries", type="tool") as step:
            step.output = "\n".join([f"- {q}" for q in state["current_search_queries"]])


async def handle_searcher(state: dict):
    """Visualize search results."""
    # The searcher updates 'raw_candidates' or 'text_search_context'
    # We can show a summary of what was found.
    
    # In the Searcher node output, 'tried_queries' contains the list of queries just executed (delta)
    queries = state.get("tried_queries", [])
    query_str = queries[-1] if queries else "Unknown Query"

    async with cl.Step(name="Searcher", type="tool") as step:
        step.input = f"Searching for: {query_str}"
        
        # Count found videos (candidates)
        # Note: 'raw_candidates' might be overwritten or appended, 
        # but here we just check if we have any candidates in the store roughly.
        # A better way is checking 'raw_candidates' in the partial state update if available,
        # but 'state' here is the node output.
        
        candidates = state.get("raw_candidates", [])
        text_context = state.get("text_search_context", [])
        
        output_msg = f"Found {len(candidates)} video candidates and {len(text_context)} web results."
        step.output = output_msg


async def handle_selector(state: dict):
    """Visualize the selection process."""
    # Selector updates video_store status to 'candidate', 'analyzing', or 'rejected'
    
    # Let's count how many are selected for watching
    video_store = state.get("video_store", {})
    selected_count = sum(1 for v in video_store.values() if v.status == "candidate" or v.status == "analyzing")
    
    if selected_count > 0:
        async with cl.Step(name="Selector", type="llm") as step:
            step.output = f"Selected {selected_count} videos for further processing."


async def handle_watcher(state: dict):
    """Visualize the watching/downloading process."""
    # Watcher updates video_store with 'summary' or 'evidence' and sets status to 'watched'/'verified'
    
    video_store = state.get("video_store", {})
    # Find recently processed videos (status='watched' or 'verified')
    # Since we don't have a strict 'last_processed' field, we iterate and show "Watched".
    
    # To avoid spamming, we could just say "Processed videos".
    # Or iterate and find which ones have content now.
    
    processed_titles = [v.title for v in video_store.values() if v.status in ["watched", "verified"]]
    
    if processed_titles:
        async with cl.Step(name="Watcher", type="tool") as step:
            list_str = "\n".join([f"- {t}" for t in processed_titles])
            step.output = f"Processed {len(processed_titles)} videos:\n{list_str}"


async def handle_checker(state: dict):
    """Visualize the self-reflection."""
    # Checker outputs a routing signal
    signal = state.get("routing_signal", "planner")
    
    async with cl.Step(name="Checker", type="run") as step:
        if signal == "analyst":
            step.output = "Sufficient information gathered. Proceeding to analysis."
        elif signal == "planner":
            step.output = "Information insufficient. Re-planning..."
        elif signal == "ask_user":
            step.output = "Ambiguity detected. Need user clarification."


async def handle_analyst(state: dict):
    """Render the final answer."""
    final_answer = state.get("final_answer", "")
    
    if final_answer:
        # 1. Send the main text response
        await cl.Message(content=final_answer, author="VideoBrowser Agent").send()
        
        # 2. (Optional) Show sources explicitly if we can parse them from video_store
        # The analyst report usually contains references, but we can add a dedicated UI element.
        video_store = state.get("video_store", {})
        verified_videos = [v for v in video_store.values() if v.status == "verified"]
        
        if verified_videos:
            elements = []
            for v in verified_videos:
                # Create a link button or text for each source
                # Chainlit doesn't have a direct "Link Card" but we can use Markdown in a separate message or Expandable
                pass
            
            # We could append a "Sources" expander
            sources_text = "\n".join([f"- [{v.title}]({v.url})" for v in verified_videos])
            if sources_text:
                await cl.Message(
                    content=f"**Sources Used:**\n{sources_text}", 
                    author="System",
                    collapse_open=False
                ).send()
