---
title: Sentinel Env — AI Agent Safety
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - safety
  - security
  - jailbreak-detection
---

# 🛡️ Sentinel Environment — AI Agent Safety & Jailbreak Detection

Deployed as a Docker-based Hugging Face Space for the Meta OpenENV RL Challenge 2026.

## Endpoints

- `POST /reset?task_name=basic-injection&seed=42` — Start new episode
- `POST /step` — Execute one step (JSON body with action)
- `GET /state` — Get current episode state
- `GET /health` — Health check
- `GET /grade` — Grade current episode
- `GET /resilience-profile` — Get resilience profile

## Tasks

1. **basic-injection** (Easy) — Detect obvious prompt injections
2. **social-engineering** (Medium) — Detect sophisticated social engineering
3. **stealth-exfiltration** (Hard) — Detect covert data exfiltration

## Source

Built for the Meta OpenENV RL Challenge 2026.
Full source: https://github.com/meta-pytorch/OpenEnv
