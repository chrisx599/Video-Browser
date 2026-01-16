import json
import re

def extract_json_from_text(text: str) -> dict:
    """
    Robustly extracts and parses JSON object or array from a string.
    Handles Markdown code blocks (```json ... ```) and surrounding text.
    
    Args:
        text: The raw string output from an LLM.
        
    Returns:
        dict or list: The parsed JSON object or array.
        
    Raises:
        json.JSONDecodeError: If valid JSON cannot be found or parsed.
    """
    try:
        # 1. Try extracting from Markdown code blocks first
        # Match either { ... } or [ ... ]
        match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
            
        # 2. Try finding the first '{' and last '}' OR first '[' and last ']'
        # This handles cases where the LLM outputs text before/after the JSON without code blocks
        
        # Check for object
        match_obj = re.search(r"(\{.*\})", text, re.DOTALL)
        # Check for array
        match_arr = re.search(r"(\[.*\])", text, re.DOTALL)
        
        # Heuristic: pick the one that looks "larger" or valid, or just try both
        candidate_strs = []
        if match_obj: candidate_strs.append(match_obj.group(1))
        if match_arr: candidate_strs.append(match_arr.group(1))
        
        # Sort candidates by length (longest match likely the full JSON)
        candidate_strs.sort(key=len, reverse=True)
        
        for s in candidate_strs:
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                continue
            
        # 3. Fallback: Try parsing the whole text (in case it's pure JSON)
        return json.loads(text)
        
    except json.JSONDecodeError as e:
        # Re-raise with context if needed, or let caller handle
        raise json.JSONDecodeError(f"Failed to parse JSON from text: {text[:100]}...", e.doc, e.pos)

def clean_vtt_text(text: str) -> str:
    """
    Cleans raw VTT transcript text by removing timestamps, tags, header info,
    and deduplicating roll-up caption lines.
    
    Args:
        text: Raw VTT text content (possibly with headers).
        
    Returns:
        str: Cleaned, continuous text.
    """
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        # 1. Skip header/metadata lines
        if line.startswith(("Kind:", "Language:", "WEBVTT")):
            continue
        if "-->" in line: # Skip timestamp lines if they exist independently
            continue
            
        # 2. Remove tags like <00:00:00.480>, <c>, </c>
        # Pattern: <[^>]+> matches anything between < and >
        line_no_tags = re.sub(r'<[^>]+>', '', line)
        
        # 3. Normalize whitespace
        line_clean = " ".join(line_no_tags.split())
        
        if not line_clean:
            continue
            
        cleaned_lines.append(line_clean)
        
    # 4. Deduplicate roll-up captions
    # In roll-up, a line often repeats the end of the previous line or contains it entirely.
    # Simple heuristic: If line A is a prefix of line B, keep B.
    # Or more commonly in YouTube auto-subs:
    # Line 1: "Hello world"
    # Line 2: "Hello world this is"
    # Line 3: "this is a test"
    
    final_lines = []
    if not cleaned_lines:
        return ""
        
    current_buffer = cleaned_lines[0]
    
    for i in range(1, len(cleaned_lines)):
        next_line = cleaned_lines[i]
        
        # Exact duplicate
        if next_line == current_buffer:
            continue
            
        # Roll-up: Next line contains current buffer (e.g. "Hello" -> "Hello World")
        if current_buffer in next_line:
            current_buffer = next_line
            continue
            
        # Roll-up overlap: End of current matches start of next?
        # This is harder to do perfectly without LCS, but for simple VTT:
        # If we didn't fully merge, append current and start new.
        final_lines.append(current_buffer)
        current_buffer = next_line
        
    final_lines.append(current_buffer)
    
    return " ".join(final_lines)

def extract_youtube_id(url: str) -> str:
    """
    Extracts the video ID from various YouTube URL formats.
    Supports:
    - Standard: https://www.youtube.com/watch?v=VIDEO_ID
    - Shorts: https://www.youtube.com/shorts/VIDEO_ID
    - Shortened: https://youtu.be/VIDEO_ID
    
    Returns the original string if no known pattern is matched (fallback).
    """
    if not url:
        return ""
        
    # Standard: v=VIDEO_ID
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
        
    # Shorts: /shorts/VIDEO_ID
    if "/shorts/" in url:
        return url.split("/shorts/")[-1].split("?")[0].split("&")[0]
        
    # Shortened: youtu.be/VIDEO_ID
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
        
    return url
