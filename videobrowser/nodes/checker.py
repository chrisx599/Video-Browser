from videobrowser.core.state import AgentState
from videobrowser.utils.logger import get_logger
from videobrowser.config import get_config
from dotenv import load_dotenv

load_dotenv()

logger = get_logger()

def checker_node(state: AgentState):
    """
    JIT Checker:
    Strictly controls the search rounds based on max_loop_steps.
    It will force the agent to loop 'max_loop_steps' times to gather information
    before proceeding to the analyst.
    """
    config = get_config()
    max_loop_steps = config.checker.max_loop_steps
    current_step = state.get("loop_step", 0) + 1
    
    # Verify state accumulation
    video_store_size = len(state.get("video_store", {}))
    history_size = len(state.get("tried_queries", []))
    
    print(f"🧐 [Checker] Step {current_step}/{max_loop_steps}")
    print(f"   -> Knowledge Accumulation: {video_store_size} videos in store, {history_size} queries tried.")
    
    logger.log("Checker", "check", {
        "step": current_step,
        "max_steps": max_loop_steps,
        "videos_found": video_store_size
    })

    # Strictly control rounds: Continue if below limit, else finish.
    # Note: JIT implementation forces loops. 
    # If you want early stopping if relevant info found, you'd check video_store here.
    # But strictly following JIT:
    if current_step < max_loop_steps:
         print("   -> Round complete. Continuing search loop (strict mode)...")
         routing_signal = "planner"
    else:
         print("   -> Max loops reached. Proceeding to Analyst.")
         routing_signal = "analyst"

    return {
        "loop_step": current_step,
        "routing_signal": routing_signal
    }