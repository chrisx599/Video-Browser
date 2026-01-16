import asyncio
import json
import os
import time
import concurrent.futures
from typing import Dict, Any
from tqdm import tqdm

# We need to import necessary modules to build the graph inside the worker process
from videobrowser.graph.builder import build_graph
from videobrowser.utils.llm_factory import get_llm
from videobrowser.utils.parser import extract_json_from_text
from langchain_core.messages import SystemMessage, HumanMessage

# --- Configuration ---
INPUT_FILE = "data/benchmark/videobrowsecomp/subset_2.jsonl"
MAX_WORKERS = 8  # Adjust based on CPU cores or desired parallelism

# --- Worker Function (Runs in a separate process) ---
def worker_task(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Independent worker function that runs in a separate process.
    It constructs its own graph and judge to avoid pickling issues.
    """
    row_id = row.get("row_id", "unknown")
    question = row["question"]
    ground_truth = row["answer"]
    
    # 1. Initialize local Graph and LLM
    # We use asyncio.run() to execute the async graph within this synchronous worker process
    try:
        app = build_graph()
        llm = get_llm(node_name="analyst")
    except Exception as e:
        return {
            "row_id": row_id,
            "error": f"Init error: {str(e)}",
            "is_correct": False
        }

    # 2. Define Execution Logic
    async def run_agent():
        # The formatting instructions are now handled by the config and Analyst node directly.
        inputs = {"user_query": question}
        
        thread_id = f"eval_proc_{row_id}"
        config = {"configurable": {"thread_id": thread_id}}
        
        final_answer = "No answer generated."
        metrics = {}
        
        # Run Graph
        try:
            async for update in app.astream(inputs, config=config):
                for node_name, node_output in update.items():
                    if node_name == "analyst":
                        final_answer = node_output.get("final_answer", final_answer)
                        metrics = node_output.get("metrics", {})
        except Exception as e:
            final_answer = f"Error during execution: {str(e)}"
            
        return final_answer, metrics

    async def run_judge(prediction: str):
        # Extract the actual answer from the agent's JSON response if possible
        clean_prediction = prediction
        try:
            pred_json = extract_json_from_text(prediction)
            if isinstance(pred_json, dict) and "Answer" in pred_json:
                clean_prediction = pred_json["Answer"]
        except Exception:
            # If parsing fails, use the full raw output
            pass

        prompt = f"""
        Question: {question}

        Ground Truth: {ground_truth}

        Model Prediction: {clean_prediction}

        Evaluate if the Prediction matches the Ground Truth.
        Return JSON: {{"is_correct": true}} or {{"is_correct": false}}
        """
        try:
            response = await llm.ainvoke([
                SystemMessage(content="You are an expert evaluator."),
                HumanMessage(content=prompt)
            ])
            # Use robust parser
            return extract_json_from_text(response.content)
        except Exception:
            return {"is_correct": False}

    # 3. Execute
    print(f"[{row_id}] Processing in PID {os.getpid()}...")
    start_time = time.time()
    
    metrics = {}
    try:
        # Run the async agent loop
        prediction, metrics = asyncio.run(run_agent())
        
        # Run the async judge
        eval_result = asyncio.run(run_judge(prediction))
        
        is_correct = eval_result.get("is_correct", False)
        
    except Exception as e:
        prediction = f"Critical Worker Error: {e}"
        is_correct = False
    
    duration = time.time() - start_time
    print(f"[{row_id}] Finished. Result: {'Correct' if is_correct else 'Incorrect'} (Time: {duration:.2f}s)")

    return {
        "row_id": row_id,
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "is_correct": is_correct,
        "duration": duration,
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "watcher_input_tokens": metrics.get("watcher", {}).get("input_tokens", 0),
        "watcher_output_tokens": metrics.get("watcher", {}).get("output_tokens", 0),
        "watcher_total_tokens": metrics.get("watcher", {}).get("total_tokens", 0),
    }

# --- Main ---
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    # Generate timestamped output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Ensure directory exists
    os.makedirs("data/benchmark/results", exist_ok=True)
    output_file = f"data/benchmark/results/evaluation_results_{timestamp}.jsonl"
    print(f"Results will be saved to: {output_file}")

    # Load Data
    data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} test cases. Starting pool with {MAX_WORKERS} workers.")

    results = []
    
    # Use ProcessPoolExecutor for true parallelism
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_row = {executor.submit(worker_task, row): row for row in data}
        
        # Process as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(data)):
            try:
                result = future.result()
                results.append(result)
                
                # Append to JSONL immediately
                with open(output_file, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    # Final Stats
    correct_count = sum(1 for r in results if r.get("is_correct"))
    total = len(results)
    accuracy = (correct_count / total * 100) if total > 0 else 0
    
    total_duration = sum(r.get("duration", 0) for r in results)
    total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in results)
    total_all_tokens = sum(r.get("total_tokens", 0) for r in results)
    watcher_total_input_tokens = sum(r.get("watcher_input_tokens", 0) for r in results)
    watcher_total_output_tokens = sum(r.get("watcher_output_tokens", 0) for r in results)
    watcher_total_all_tokens = sum(r.get("watcher_total_tokens", 0) for r in results)

    print(f"\nEvaluation Complete.")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total})")
    print(f"Total Inference Time (sum of all threads): {total_duration:.2f}s")
    print(f"Total Token Usage: {total_all_tokens} (In: {total_input_tokens}, Out: {total_output_tokens})")
    print(f"Watcher Token Usage: {watcher_total_all_tokens} (In: {watcher_total_input_tokens}, Out: {watcher_total_output_tokens})")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
