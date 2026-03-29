import os
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
TASKS = ["remove_duplicates", "fill_missing", "fix_outliers"]
BASE = "https://HarshavardhanJ-data-cleaning-env.hf.space"

def run_task(task_name):
    obs = requests.post(f"{BASE}/reset",
          json={"task_name": task_name}).json()
    total_reward = 0.0
    for step in range(10):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content":
                 "Reply with ONLY one of: remove_duplicates, fill_missing, fix_outliers"},
                {"role": "user", "content": str(obs)}
            ],
            max_tokens=20
        )
        action = response.choices[0].message.content.strip()
        result = requests.post(f"{BASE}/step",
                 json={"action_type": action}).json()
        total_reward += result.get("reward", 0)
        obs = result.get("observation", obs)
        print(f"  Step {step+1}: {action} → {result.get('reward',0):.2f}")
        if result.get("done"):
            break
    return total_reward

if __name__ == "__main__":
    print("=== Data Cleaning Baseline ===\n")
    for task in TASKS:
        print(f"\nTask: {task}")
        print(f"Score: {run_task(task):.2f}")