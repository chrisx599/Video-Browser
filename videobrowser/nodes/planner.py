from langchain_core.messages import SystemMessage, HumanMessage
from videobrowser.core.state import AgentState, format_planner_view
from videobrowser.utils.prompt_manager import load_prompt
from videobrowser.utils.parser import extract_json_from_text
from videobrowser.utils.metrics import update_token_metrics
from videobrowser.utils.logger import get_logger
from videobrowser.config import get_config
from dotenv import load_dotenv
from videobrowser.utils.llm_factory import get_llm
import json

load_dotenv()

llm = get_llm(node_name="planner")
logger = get_logger()

def planner_node(state: AgentState):
    logger.log("Planner", "start", {"loop_step": state.get("loop_step", 0)})
    config = get_config()
    
    context_view = format_planner_view(state)
    
    system_prompt = load_prompt("planner.j2", max_queries=config.planner.max_queries)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context_view)
    ]
    response = llm.invoke(messages)
    
    metrics = update_token_metrics(state.get("metrics", {}), response)
    
    try:
        plan = extract_json_from_text(response.content)
        # Enforce max_queries limit
        if plan.get("search_queries"):
             plan["search_queries"] = plan["search_queries"][:config.planner.max_queries]
    except Exception as e:
        print(f"⚠️ [Planner] JSON parsing failed: {e}. Fallback to user query.")
        logger.log("Planner", "error", {"error": str(e), "content": response.content}, level="ERROR")
        plan = {
            "thought": "Parsing error, fallback to original query.",
            "search_queries": [state["user_query"]]
        }

    print(f"🧠 [Planner] Thought: {plan.get('thought')}")
    print(f"🔍 [Planner] Queries: {plan.get('search_queries')}")

    logger.log("Planner", "end", {
        "thought": plan.get("thought"),
        "search_queries": plan.get("search_queries")
    })

    return {
        "plan_trace": [f"Thought: {plan.get('thought')}"],
        "current_search_queries": plan.get("search_queries", []),
        "metrics": metrics
    }