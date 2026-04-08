"""
PrivRL Inference Script — Heuristic Agent + LLM Privacy Advisor

Strict OpenEnv output format:
  [START] task=<task> env=PrivRL model=<model>
  [STEP] step=N action=<name> reward=<float> done=<bool> error=null
  [END] success=<bool> steps=<n> rewards=<csv>
"""
import os
import sys
from openai import OpenAI
from env.environment import PrivRLEnv, classify_tracker
from env.models import PrivRLAction

# =============================================================================
# CONFIG
# =============================================================================

SEED = 42
MAX_STEPS_SAFETY = 100
USE_LLM = True

API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# =============================================================================
# CLIENT
# =============================================================================

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# =============================================================================
# FALLBACK ADVICE
# =============================================================================

FALLBACK_ADVICE = {
    "mark_safe": "This website appears safe. Normal browsing is fine.",
    "mark_risky": "This site has moderate tracking. Be cautious with personal data.",
    "mark_dangerous": "High privacy risk detected. Avoid sharing personal information.",
}

# =============================================================================
# LLM ADVISOR
# =============================================================================

def generate_privacy_advice(obs, action_name):
    if not USE_LLM:
        return FALLBACK_ADVICE.get(action_name, "")

    try:
        prompt = f"""You are a privacy advisor.

Cookies: {obs.cookies}
Trackers: {obs.trackers}
HTTPS: {obs.https}
Policy: {obs.privacy_policy[:200]}

Classification: {action_name}

Give short 1-2 sentence advice."""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.4,
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return FALLBACK_ADVICE.get(action_name, "")

# =============================================================================
# HEURISTIC
# =============================================================================

def heuristic_classify(obs):
    risk_score = 0
    trackers = obs.trackers or []

    malicious_count = 0

    for t in trackers:
        cat = classify_tracker(t)
        if cat == "malicious":
            malicious_count += 1
            risk_score += 3
        elif cat in ("social", "opaque"):
            risk_score += 2
        else:
            risk_score += 1

    if len(trackers) >= 5:
        risk_score += 2

    if not obs.https:
        risk_score += 2

    if obs.cookies > 30:
        risk_score += 2
    elif obs.cookies > 15:
        risk_score += 1

    if malicious_count > 0 or risk_score >= 6:
        return "mark_dangerous"
    elif risk_score >= 3:
        return "mark_risky"
    else:
        return "mark_safe"

# =============================================================================
# RUN
# =============================================================================

def run_episode(task_id="easy", seed=SEED):
    env = PrivRLEnv()
    obs = env.reset(task_id=task_id, seed=seed)

    print(f"[START] task={task_id} env=PrivRL model={MODEL_NAME}")

    done = False
    step_count = 0
    all_rewards = []

    while not done and step_count < MAX_STEPS_SAFETY:
        action_name = heuristic_classify(obs)
        action = PrivRLAction(classification=action_name)
        obs, reward, done, info = env.step(action)
        step_count += 1
        all_rewards.append(reward)

        print(f"[STEP] step={step_count} action={action_name} reward={reward:.2f} done={str(done).lower()} error=null")

        # Generate and print advice to STDERR so it doesn't break OpenEnv STDOUT parsing
        if USE_LLM:
            advice = generate_privacy_advice(obs, action_name)
            print(f"[ADVICE] {advice}", file=sys.stderr)

    grade = env.grade()
    score = grade["score"]
    success = score > 0.0  # True if episode completes basically
    rewards_csv = ",".join(f"{r:.2f}" for r in all_rewards)

    print(f"[END] success={str(success).lower()} steps={step_count} score={score:.4f} rewards={rewards_csv}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    for tid in ["easy", "medium", "hard"]:
        run_episode(task_id=tid, seed=SEED)