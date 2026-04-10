---
title: PrivRL
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---


# PrivRL — Multi-Step Privacy Risk Classification Environment

A rigorous, OpenEnv-compliant reinforcement learning environment designed to evaluate AI agents on privacy risk assessment and decision-making under partial observability. 

## Problem Statement

Modern websites deploy diverse, often opaque data collection practices. Identifying high-risk privacy violations—such as aggressive third-party trackers, insecure connections, and deceptive data policies—requires aggregating multiple disparate signals. Automating this synthesis is critical for scalable web oversight and user protection.

Current approaches often rely on brittle, rules-based keyword blocking. PrivRL transforms privacy classification into an RL challenge, pushing agents to navigate ambiguity, weigh contradictory terms, and optimize for long-term interpretability and accuracy.

## Solution Overview

PrivRL is a structured reinforcement learning benchmark where agents classify simulated website environments into discrete risk categories ("safe", "risky", or "dangerous"). By framing privacy analysis as an RL problem, PrivRL allows researchers to train agents capable of navigating complex or deceptive privacy architectures, moving beyond simple keyword blocking to true semantic risk evaluation.

## Key Features

*   **OpenEnv Compliant API**: Fully supports standardized REST endpoints for seamless hackathon validation and infrastructure integration.
*   **Procedural Site Generation**: Generates varied privacy policies, tracker distributions, and cookie counts programmatically to prevent memorization and enforce generalized learning.
*   **Deception Model v2**: Features a contradiction-aware algorithmic scorer using weighted phrasing and density normalization to identify when "safe" terminology masks dangerous data-sharing clauses.
*   **Vector Observations**: Serves a cleanly normalized 13-dimensional float array natively supporting advanced DQN, PPO, and standard RL policy algorithms.
*   **Live LLM Inference Insights**: Augments baseline decision steps with an integrated semantic advisor powered by OpenRouter for dynamic, real-time context.

## Architecture

The system segregates responsibilities across clean layer boundaries:
*   **API Layer**: A FastAPI + Uvicorn server providing the `/reset`, `/step`, `/state`, and `/health` interfaces.
*   **Environment Layer**: The `PrivRLEnv` engine governing state transitions, rewards, dataset generation, and observation masking.
*   **Model Layer**: Pydantic dataclasses (`models.py`) strictly enforcing the data contract across boundaries.
*   **Inference Layer**: The client application executing decisions against the active environment state and interfacing with external LLMs for advisory reporting.

## RL Formulation

*   **State / Observation**: A Partially Observable Markov Decision Process (POMDP). The visible state includes HTTPS toggles, integer cookie loads, text snippets from privacy policies, and arrays of third-party domains.
*   **Actions**: The action space accepts discrete risk classifications:
    *   `mark_safe`
    *   `mark_risky`
    *   `mark_dangerous`
*   **Reward Design**: Rewards are shaped based purely on correctness versus underlying algorithmic truth, penalizing over/under-reactions strictly while maintaining boundary thresholds that allow continuous learning.

## LLM Integration

The provided baseline inference agent features a non-blocking OpenAI-compatible client routed through OpenRouter (`openai/gpt-4o-mini`). 
*   **Role**: Acts as a "Privacy Advisor AI" executing on the exact observation matrix consumed by the RL agent. 
*   **Execution**: Generates short, 2-sentence rationale advisories concurrent with the discrete action decisions. 
*   **Safety**: Runs strictly functionally separate from the reward loop. If the API fails, rates-limits, or times out, the system natively catches the exception and outputs static, contextual fallbacks corresponding to the chosen baseline action.

## API Documentation

The environment is built on FastAPI and exposes the following OpenEnv endpoints:

*   `POST /reset` — Accepts `task_id` (easy, medium, hard) and `seed` to initialize a new episode and generate the first observation.
*   `POST /step` — Receives the agent's classification action and returns the subsequent observation, reward, and terminal state flag.
*   `GET /state` — Retrieves the protected latent state of the current active episode for orchestrator monitoring.
*   `GET /health` — Provides immediate system readiness validation for CI/CD or Docker health checks.

## Setup Instructions

### Local Setup

Ensure Python 3.10 is installed.

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker Setup

The system includes a production-ready container definition with built-in health monitoring.

```bash
docker build -t privrl .
docker run -p 7860:7860 privrl
```

## Example Output

The baseline inference script (`python inference.py`) strictly adheres to standard validation formats while providing LLM augments:

1.Output for stdout
```text
[START] task=easy env=PrivRL model=openai/gpt-4o-mini
[STEP] step=1 action=mark_safe reward=0.80 done=false error=null
[STEP] step=2 action=mark_risky reward=0.57 done=false error=null
[STEP] step=3 action=mark_safe reward=0.48 done=false error=null
[STEP] step=4 action=mark_safe reward=0.48 done=false error=null
[STEP] step=5 action=mark_risky reward=0.52 done=false error=null
[STEP] step=6 action=mark_safe reward=0.80 done=false error=null
[STEP] step=7 action=mark_risky reward=0.57 done=false error=null
[STEP] step=8 action=mark_risky reward=0.57 done=false error=null
[STEP] step=9 action=mark_safe reward=0.48 done=false error=null
[STEP] step=10 action=mark_safe reward=0.80 done=false error=null
[STEP] step=11 action=mark_safe reward=0.80 done=false error=null
[STEP] step=12 action=mark_safe reward=0.80 done=true error=null
[END] success=true steps=12 score=0.6518 rewards=0.80,0.57,0.48,0.48,0.52,0.80,0.57,0.57,0.48,0.80,0.80,0.80
[START] task=medium env=PrivRL model=openai/gpt-4o-mini
[STEP] step=1 action=mark_safe reward=0.50 done=false error=null
[STEP] step=2 action=mark_safe reward=0.48 done=false error=null
[STEP] step=3 action=mark_safe reward=0.80 done=false error=null
```

2. Output for stdout with stderr
```text
[START] task=easy env=PrivRL model=openai/gpt-4o-mini
[STEP] step=1 action=mark_safe reward=0.80 done=false error=null
[ADVICE] It's important to enable HTTPS on your website to ensure secure data transmission and protect user privacy. Additionally, consider reviewing and updating your cookie and tracker policies to enhance user trust and compliance with privacy regulations.
[STEP] step=2 action=mark_risky reward=0.57 done=false error=null
[ADVICE] Since you have a relatively high number of cookies and no trackers, it's important to regularly review and manage your cookie settings to ensure your privacy is not compromised. Additionally, make sure to verify the hidden policy for any potential risks associated with the cookies you are accepting.
[STEP] step=3 action=mark_safe reward=0.48 done=false error=null
[ADVICE] This website appears safe. Normal browsing is fine.
[STEP] step=4 action=mark_safe reward=0.48 done=false error=null
[ADVICE] Ensure that your website uses HTTPS to encrypt data transmitted between users and your server, enhancing security and privacy. Additionally, consider implementing a clear cookie policy to inform users about cookie usage and obtain their consent.
[STEP] step=5 action=mark_risky reward=0.52 done=false error=null
[ADVICE] Ensure that your privacy policy is transparent and accessible to users, as it is currently hidden. Additionally, regularly review and update your cookie and tracker usage to minimize potential risks.
[STEP] step=6 action=mark_safe reward=0.80 done=false error=null
[ADVICE] It's important to enable HTTPS on your website to ensure secure data transmission and protect user privacy. Additionally, consider implementing a clear cookie and tracker policy to inform users about data collection practices.
[STEP] step=7 action=mark_risky reward=0.57 done=false error=null
[ADVICE] Your website has a high number of cookies and is not using HTTPS, which poses significant privacy and security risks for users. It is crucial to implement HTTPS and evaluate the necessity of the cookies in use to enhance user trust and protect their data.
[STEP] step=8 action=mark_risky reward=0.57 done=false error=null
[ADVICE] Since your site uses cookies but has no trackers and is secured with HTTPS, ensure that your cookie policy is transparent and compliant with privacy regulations. Regularly review and update your cookie consent mechanisms to maintain user trust and legal compliance.
[STEP] step=9 action=mark_safe reward=0.48 done=false error=null
[ADVICE] Ensure that your website maintains a clear and transparent privacy policy, even if it is currently hidden, to inform users about data handling practices and their rights. Regularly review and update your privacy practices to stay compliant with evolving regulations.
[STEP] step=10 action=mark_safe reward=0.80 done=false error=null
[ADVICE] Your website is secure with HTTPS and has no cookies or trackers, which is great for user privacy. However, ensure that your privacy policy is transparent and accessible to users for full compliance and trust.
[STEP] step=11 action=mark_safe reward=0.80 done=false error=null
[ADVICE] Ensure that your website maintains a clear and transparent privacy policy, even if it's currently hidden, to inform users about data collection practices. Regularly review and update your privacy measures to comply with regulations and enhance user trust.
[STEP] step=12 action=mark_safe reward=0.80 done=true error=null
[ADVICE] Ensure you regularly review and update your privacy settings, especially for cookies and trackers, to maintain your online security. Always prioritize using websites that implement HTTPS for secure data transmission.
[END] success=true steps=12 score=0.6518 rewards=0.80,0.57,0.48,0.48,0.52,0.80,0.57,0.57,0.48,0.80,0.80,0.80

```

## Project Structure

```text
PrivRL/
├── README.md                 # Documentation
├── openenv.yaml              # OpenEnv platform configuration metadata
├── Dockerfile                # Production container parameters
├── requirements.txt          # Frozen dependency graph
├── app.py                    # Core FastAPI backend router
├── inference.py              # Baseline heuristic agent + OpenRouter LLM 
└── env/                      
    ├── environment.py        # Core algorithm logic and reward shaping
    ├── models.py             # Strict Pydantic type schemas (API Contract)
    └── dataset.py            # Static baseline truth datasets
```

## Why This Project Stands Out

PrivRL bridges a critical gap between security research and reinforcement learning orchestration through rigorous engineering:
*   **Production Robustness**: The API is 100% type-safe, bounding inputs to prevent degenerate edge testing from crashing container processes during evaluation.
*   **Procedural Integrity**: Procedural generation avoids trivial memorization architectures allowing genuine testing of generalization.
*   **Extensible Integrations**: The system unifies a deterministic, heuristic-driven baseline agent with stochastic, real-world LLM semantic evaluations without blocking core RL training paths.

## Future Improvements

*   **Custom Environment Wrappers**: Adding direct integration pipelines via StableBaselines3 vectorized wrappers for multi-agent training scenarios.
*   **Semantic Observation Enhancements**: Passing full DOM trees into observations alongside metadata instead of localized policy snippets.
*   **Continuous Domain Crawling**: Dynamically hooking the procedural generation target pool to a live crawler indexing the Tranco Top 1M domains.
