from typing import Dict, Any

def update_token_metrics(current_metrics: Dict[str, Any], response: Any, category: str = None) -> Dict[str, Any]:
    """
    Extracts token usage from an LLM response and updates the metrics dictionary.
    
    Args:
        current_metrics: The existing metrics dictionary from the state.
        response: The LLM response object (expected to have .response_metadata['token_usage']).
        category: Optional category name to track specific metrics (e.g., 'watcher').
        
    Returns:
        A new metrics dictionary with updated counts.
    """
    if not current_metrics:
        current_metrics = {}
        
    token_usage = response.response_metadata.get("token_usage", {})
    new_metrics = current_metrics.copy()
    
    # Global metrics
    new_metrics["input_tokens"] = new_metrics.get("input_tokens", 0) + token_usage.get("prompt_tokens", 0)
    new_metrics["output_tokens"] = new_metrics.get("output_tokens", 0) + token_usage.get("completion_tokens", 0)
    new_metrics["total_tokens"] = new_metrics.get("total_tokens", 0) + token_usage.get("total_tokens", 0)
    
    # Category specific metrics
    if category:
        cat_metrics = new_metrics.get(category, {})
        if not isinstance(cat_metrics, dict):
            cat_metrics = {}
            
        cat_metrics["input_tokens"] = cat_metrics.get("input_tokens", 0) + token_usage.get("prompt_tokens", 0)
        cat_metrics["output_tokens"] = cat_metrics.get("output_tokens", 0) + token_usage.get("completion_tokens", 0)
        cat_metrics["total_tokens"] = cat_metrics.get("total_tokens", 0) + token_usage.get("total_tokens", 0)
        
        new_metrics[category] = cat_metrics
    
    return new_metrics
