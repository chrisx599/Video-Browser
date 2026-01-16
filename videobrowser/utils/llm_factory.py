import os
from langchain_openai import ChatOpenAI
from videobrowser.config import get_config

def get_llm(node_name: str = None) -> ChatOpenAI:
    """
    Factory function to get an LLM instance based on configuration.
    
    Args:
        node_name: The name of the node (e.g., 'planner', 'selector') to apply overrides.
    """
    config = get_config()
    
    # Start with default config values
    llm_params = config.llm.default.model_dump()
    
    # Apply overrides if they exist for this node
    if node_name and node_name in config.llm.overrides:
        # Merge dictionary updates
        override_params = config.llm.overrides[node_name]
        llm_params.update(override_params)
    
    # Extract params needed for instantiation
    api_key = llm_params.get("api_key")
    
    if not api_key:
        raise ValueError("API Key not found in configuration or environment variable.")
    
    # Construct arguments for ChatOpenAI
    # We filter out internal config keys (like 'provider') that ChatOpenAI doesn't accept,
    # unless we want to map them specifically.
    
    init_kwargs = {
        "model": llm_params.get("model"),
        "temperature": llm_params.get("temperature"),
        "api_key": api_key,
    }
    
    if llm_params.get("base_url"):
        init_kwargs["base_url"] = llm_params.get("base_url")

    if llm_params.get("max_tokens"):
        init_kwargs["max_tokens"] = llm_params.get("max_tokens")
        
    # Create the instance
    # Future: Switch on llm_params['provider'] to support other classes
    return ChatOpenAI(**init_kwargs)
