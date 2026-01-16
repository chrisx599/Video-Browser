from videobrowser.graph.builder import build_graph
import time

if __name__ == "__main__":
    app = build_graph()
    
    # Use Thread ID to isolate different sessions
    config = {"configurable": {"thread_id": "research_demo_1"}}
    
    print("🚀 Agent Graph Started...")
    
    # First run
    inputs = {
        "user_query": "A legendary power forward, after switching careers to become a commentator, once bet with his co-host on a popular American basketball analysis show that a No. 1 draft pick center from Asia could not score 19 points in a single game. Subsequently, the center proved himself in a game, forcing the commentator to fulfill the bet \u2014 kissing a donkey's butt on a subsequent live broadcast. What was the center's final score in that game?",
        "metrics": {"start_time": time.time(), "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    }

    
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