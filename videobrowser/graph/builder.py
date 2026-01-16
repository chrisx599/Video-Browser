from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from videobrowser.core.state import AgentState

from videobrowser.nodes.planner import planner_node
from videobrowser.nodes.searcher import searcher_node
from videobrowser.nodes.selector import selector_node
from videobrowser.nodes.watcher import watcher_node
from videobrowser.nodes.checker import checker_node
from videobrowser.nodes.analyst import analyst_node



def route_checker_output(state: AgentState):
    """
    Decide where to go after Checker finishes:
    - Ambiguity found -> UserProxy (Ask user)
    - Confidence high -> Analyst (Write summary)
    - Not finished/Need new loop -> Planner (Re-plan)
    """
    signal = state.get("routing_signal", "planner")
    
    if signal == "ask_user":
        return "user_proxy"
    elif signal == "analyst":
        return "analyst"
    else:
        # Default: Go back to Planner for the next loop
        return "planner"

def route_selector_output(state: AgentState):
    """
    Selector routing logic:
    - If video_id is selected -> Watcher (Watch closely)
    - If nothing useful found -> Checker (Checker will find insufficiency and send back to Planner)
    """
    # Check if there are videos with status 'analyzing' or 'candidate' on the blackboard
    candidates = [
        v for v in state["video_store"].values() 
        if v.status in ["candidate", "analyzing"]
    ]
    
    if len(candidates) > 0:
        return "watcher"  # Videos found, go watch
    else:
        return "checker"  # No videos, go check (will likely fail, but follows process)


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("searcher", searcher_node)
    workflow.add_node("selector", selector_node)
    workflow.add_node("watcher", watcher_node)
    workflow.add_node("checker", checker_node)
    workflow.add_node("analyst", analyst_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "selector")
    workflow.add_edge("watcher", "checker")
    workflow.add_edge("analyst", END)

    workflow.add_conditional_edges(
        "selector",
        route_selector_output,
        {
            "watcher": "watcher",
            "checker": "checker"
        }
    )
    
    workflow.add_conditional_edges(
        "checker",
        route_checker_output,
        {
            "planner": "planner",
            "analyst": "analyst",
        }
    )
    
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)



if __name__ == "__main__":
    app = build_graph()
    
    # Use Thread ID to isolate different sessions
    config = {"configurable": {"thread_id": "research_demo_1"}}
    
    print("🚀 Agent Graph Started...")
    
    # First run
    inputs = {"user_query": "Find me a video explaining the principle of sugar coloring in braised pork"}
    
    for update in app.stream(inputs, config=config):
        # Print output of each step in real-time for debugging
        for node_name, node_output in update.items():
            print(f"--- Step: {node_name} ---")
            
            # Specifically print the final answer from the analyst node
            if node_name == "analyst":
                print("\n✅ FINAL ANSWER:\n")
                print(node_output.get("final_answer", "No answer generated."))
                print("\n" + "="*50 + "\n")
            
            # print(node_output) # Print detailed State changes here