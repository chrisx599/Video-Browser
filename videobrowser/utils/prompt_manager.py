import os
from pathlib import Path
from typing import Any
from jinja2 import Template

# Define the directory relative to this file: src/utils/../prompts -> src/prompts
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def load_prompt(template_name: str, **kwargs: Any) -> str:
    """
    Loads a prompt template from the src/prompts directory and renders it.
    
    Args:
        template_name: The filename of the template (e.g., 'planner.j2').
        **kwargs: Key-value pairs to substitute into the template.
        
    Returns:
        The rendered string.
        
    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    template_path = PROMPTS_DIR / template_name
    
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template file not found at: {template_path}")
        
    with open(template_path, "r", encoding="utf-8") as f:
        raw_content = f.read()
        
    if kwargs:
        template = Template(raw_content)
        return template.render(**kwargs)
    

    return Template(raw_content).render()
