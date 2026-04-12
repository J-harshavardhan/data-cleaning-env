---
title: Data Cleaning RL Environment
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - data-cleaning
---

# 🧹 Data Cleaning RL Environment

An **OpenAI Gym-compatible Reinforcement Learning environment** for training agents to clean tabular datasets. Built for the **OpenEnv Hackathon × Scaler (2026)**.

An agent interacts with a dirty DataFrame episode by episode — removing duplicates, filling missing values, or fixing outliers — and receives rewards based on how clean the resulting dataset becomes.

---

## 🎯 What It Does

| Feature | Details |
|---|---|
| **Task** | Clean a tabular dataset through sequential actions |
| **Observation** | Dataset preview, row count, duplicate count, missing values |
| **Actions** | `remove_duplicates`, `fill_missing`, `fix_outliers` |
| **Reward** | Based on data quality improvement per action |
| **Episodes** | Each episode = one dirty dataset instance |

---

## ⚡ Quick Start

### Connect via Python

```python
from data_cleaning_env import DataCleaningAction, DataCleaningEnv
import asyncio

async def main():
    with DataCleaningEnv.from_env("HarshavardhanJ/data-cleaning-env") as env:
        state = await env.reset()
        result = await env.step(DataCleaningAction(
            action_type="fill_missing"
        ))
        print(result)

asyncio.run(main())
```

### Connect to a Local Server

```python
env = DataCleaningEnv(base_url="http://localhost:8000")
```

---

## 🕹️ Web Interface

The deployed Space exposes:

| Endpoint | Description |
|---|---|
| `/web` | Interactive UI for exploring the environment |
| `/docs` | Full OpenAPI / Swagger documentation |
| `/health` | Container health check |
| `/ws` | WebSocket endpoint for low-latency sessions |

---

## 🤖 Action Types

| Action | Difficulty | Description |
|---|---|---|
| `remove_duplicates` | Easy | Remove all duplicate rows |
| `fill_missing` | Medium | Fill nulls using mean/median per column |
| `fix_outliers` | Hard | Clip outliers using IQR method |

---

## 🔁 Contribute

Fork and submit improvements via Pull Request:

```bash
openenv fork HarshavardhanJ/data-cleaning-env --repo-id <your-username>/<your-repo-name>
cd <forked-repo>
openenv push HarshavardhanJ/data-cleaning-env --create-pr
```

---

## 🏗️ Built With

- [OpenEnv](https://openenv.dev) — RL environment framework
- [FastAPI](https://fastapi.tiangolo.com) — API server
- [HuggingFace Spaces](https://huggingface.co/spaces) — Hosting (Docker)

---

## 👤 Author

**Jagannati Harshavardhan**  
B.Tech @ GIST Nellore | AI & ML @ IIT Patna × Masai School  
[GitHub](https://github.com/J-harshavardhan) · [HuggingFace](https://huggingface.co/HarshavardhanJ) · [LinkedIn](https://linkedin.com/in/jagannati-harsha-vardhan-38117637b)
