"""
PrivRL Inference Script — Smart Baseline Agent (v4)

Demonstrates multi-step POMDP reasoning:
  1. Inspect trackers (if hidden)
  2. Inspect privacy policy (if hidden)
  3. Classify using heuristic based on revealed signals

Output format (strict):
  [START]
  [STEP] action=<name> reward=<float>
  [END]
"""

from env.environment import PrivRLEnv, classify_tracker
from env.models import PrivRLAction

# ── Fixed seed for reproducibility ──
SEED = 42
MAX_STEPS_SAFETY = 100  # absolute fallback to prevent infinite loops


def heuristic_classify(obs) -> str:
    """
    Deterministic heuristic classifier based on revealed observation signals.

    Decision logic:
      1. Count trackers by risk category (malicious > ads > social > analytics)
      2. Check HTTPS status
      3. Check cookie count
      4. Scan policy text for danger signals (if visible)

    Returns one of: "mark_safe", "mark_risky", "mark_dangerous"
    """
    risk_score = 0

    # ── Tracker signal ──
    trackers = obs.trackers if obs.trackers else []
    n_trackers = len(trackers)

    malicious_count = 0
    ads_count = 0
    for t in trackers:
        cat = classify_tracker(t)
        if cat == "malicious":
            malicious_count += 1
            risk_score += 3
        elif cat == "ads":
            ads_count += 1
            risk_score += 1
        elif cat in ("social", "opaque"):
            risk_score += 2
        elif cat == "attribution":
            risk_score += 1

    if n_trackers >= 5:
        risk_score += 2
    elif n_trackers >= 3:
        risk_score += 1

    # ── HTTPS signal ──
    if not obs.https:
        risk_score += 2

    # ── Cookie signal ──
    if obs.cookies > 30:
        risk_score += 2
    elif obs.cookies > 15:
        risk_score += 1

    # ── Policy deception signal (if visible) ──
    policy = obs.privacy_policy if obs.privacy_policy else ""
    if "[HIDDEN]" not in policy and policy.strip():
        policy_lower = policy.lower()
        danger_keywords = [
            "becomes our property", "any purpose we see fit",
            "sold to third parties", "stored indefinitely",
            "without warrant", "financial data",
            "ad targeting", "detailed user profiles",
            "voice recordings", "unencrypted",
        ]
        safe_keywords = [
            "do not collect", "end-to-end encrypted",
            "no data sold", "gdpr compliant", "anonymized",
        ]
        danger_hits = sum(1 for kw in danger_keywords if kw in policy_lower)
        safe_hits = sum(1 for kw in safe_keywords if kw in policy_lower)

        if danger_hits >= 2:
            risk_score += 3
        elif danger_hits >= 1:
            risk_score += 1

        if safe_hits >= 2:
            risk_score -= 2
        elif safe_hits >= 1:
            risk_score -= 1

    # ── Decision ──
    if malicious_count > 0 or risk_score >= 6:
        return "mark_dangerous"
    elif risk_score >= 3:
        return "mark_risky"
    else:
        return "mark_safe"


def run_episode(task_id: str, seed: int) -> None:
    """Run a single episode with the smart baseline agent."""
    env = PrivRLEnv()
    obs = env.reset(task_id=task_id, seed=seed)

    done = False
    step_count = 0

    while not done and step_count < MAX_STEPS_SAFETY:
        action_name = heuristic_classify(obs)

        action = PrivRLAction(classification=action_name)
        obs, reward, done, info = env.step(action)
        step_count += 1

        print(f"[STEP] action={action_name} reward={reward:.4f}")


if __name__ == "__main__":
    print("[START]")
    run_episode(task_id="easy", seed=SEED)
    print("[END]")