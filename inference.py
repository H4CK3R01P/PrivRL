"""
PrivRL Inference Script — Heuristic Agent + LLM Privacy Advisor (No ENV)

Uses OpenRouter via OpenAI client.
"""
import os
from openai import OpenAI
from env.environment import PrivRLEnv, classify_tracker
from env.models import PrivRLAction

# =============================================================================
# CONFIG (HARDCODED — CHANGE ONLY HERE)
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
        prompt = f"""
You are a privacy advisor.

Cookies: {obs.cookies}
Trackers: {obs.trackers}
HTTPS: {obs.https}
Policy: {obs.privacy_policy[:200]}

Classification: {action_name}

Give short 1–2 sentence advice.
"""

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

def run_episode():
    env = PrivRLEnv()
    obs = env.reset(task_id="easy", seed=SEED)

    done = False

    while not done:
        action_name = heuristic_classify(obs)

        action = PrivRLAction(classification=action_name)
        obs, reward, done, _ = env.step(action)

        print(f"[STEP] action={action_name} reward={reward:.4f}")

        if USE_LLM:
            advice = generate_privacy_advice(obs, action_name)
            print(f"[ADVICE] {advice}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("[START]")
    run_episode()
    print("[END]")