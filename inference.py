import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://harshavardhanJ-data-cleaning-env.hf.space")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
HF_TOKEN = os.environ.get("HF_TOKEN")
API_KEY = os.environ.get("API_KEY", "dummy")

# ✅ Must use API_BASE_URL and API_KEY from environment
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ.get("API_KEY", "dummy")
)

def get_action_from_llm(task_name, observation):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a data cleaning agent. Respond with only one of: remove_duplicates, fill_missing, fix_outliers"},
            {"role": "user", "content": f"Task: {task_name}. Observation: {observation}. What action should be taken?"}
        ],
        max_tokens=20
    )
    return response.choices[0].message.content.strip()

def run_task(task_name):
    try:
        reset_resp = requests.post(
            f"{API_BASE_URL}/reset",
            json={"task_name": task_name},
            timeout=30
        )
        try:
            obs = reset_resp.json()
        except json.JSONDecodeError:
            print(f"[END] task={task_name} final_reward=0.0")
            return 0.0

        print(f"[START] task={task_name} obs={json.dumps(obs)}")

        # ✅ Use LLM to decide action
        action = get_action_from_llm(task_name, str(obs))

        step_resp = requests.post(
            f"{API_BASE_URL}/step",
            json={"action_type": task_name},
            timeout=30
        )
        try:
            result = step_resp.json()
        except json.JSONDecodeError:
            print(f"[END] task={task_name} final_reward=0.0")
            return 0.0

        reward = result.get("reward", 0.0)
        print(f"[STEP] task={task_name} action={action} reward={reward}")
        print(f"[END] task={task_name} final_reward={reward}")
        return reward

    except Exception as e:
        print(f"[END] task={task_name} final_reward=0.0 error={str(e)}")
        return 0.0

if __name__ == "__main__":
    tasks = ["remove_duplicates", "fill_missing", "fix_outliers"]
    scores = {}
    for task in tasks:
        scores[task] = run_task(task)
    print(f"\nFinal scores: {scores}")