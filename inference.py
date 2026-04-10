import os
import json
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "https://harshavardhanJ-data-cleaning-env.hf.space")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def run_task(task_name):
    try:
        # Reset environment
        reset_resp = requests.post(
            f"{API_BASE_URL}/reset",
            json={"task_name": task_name},
            timeout=30
        )
        print(f"[START] task={task_name}")
        
        # Check response is valid JSON
        try:
            obs = reset_resp.json()
        except json.JSONDecodeError:
            print(f"Reset response not JSON: {reset_resp.text}")
            return 0.0

        # Step through the task
        action_map = {
            "remove_duplicates": "remove_duplicates",
            "fill_missing": "fill_missing", 
            "fix_outliers": "fix_outliers"
        }
        
        action = action_map.get(task_name, "remove_duplicates")
        step_resp = requests.post(
            f"{API_BASE_URL}/step",
            json={"action_type": action},
            timeout=30
        )
        
        try:
            result = step_resp.json()
        except json.JSONDecodeError:
            print(f"Step response not JSON: {step_resp.text}")
            return 0.0

        reward = result.get("reward", 0.0)
        print(f"[STEP] action={action} reward={reward}")
        print(f"[END] task={task_name} final_reward={reward}")
        return reward

    except Exception as e:
        print(f"[END] task={task_name} error={e} final_reward=0.0")
        return 0.0

if __name__ == "__main__":
    tasks = ["remove_duplicates", "fill_missing", "fix_outliers"]
    scores = {}
    for task in tasks:
        scores[task] = run_task(task)
    print(f"\nFinal scores: {scores}")