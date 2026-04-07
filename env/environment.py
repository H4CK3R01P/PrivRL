"""
PrivRL Environment — OpenEnv-compatible RL environment for website privacy risk classification.

An agent is presented with simulated website attributes (cookies, trackers, HTTPS, privacy policy)
and must classify the website as "mark_safe", "mark_risky", or "mark_dangerous".

Implements the OpenEnv interface:
    - reset(task_id)  → PrivRLObservation  (randomized website to classify)
    - step(action)    → PrivRLObservation  (shaped reward + next website or done)
    - state           → PrivRLState        (internal episode metadata)

Improved reward shaping (v2):
    Base classification reward:
        - Exact match:         +1.0
        - Off by one level:    +0.3
        - Completely wrong:    -1.0

    Bonus signals (up to +0.5 additional):
        - Tracker awareness:   +0.15 if action aligns with tracker risk profile
        - HTTPS awareness:     +0.10 if action correctly accounts for HTTPS/HTTP
        - Policy deception:    +0.25 if agent detects deceptive privacy policy text

    Total reward per step is capped to [−1.0, +1.5] and normalized to [0.0, 1.0] for grading.
"""

import random
import uuid
from typing import Optional, List, Dict

from env.models import PrivRLAction, PrivRLObservation, PrivRLState


# =============================================================================
# Tracker Taxonomy — categorized by privacy impact
# =============================================================================

TRACKER_CATEGORIES: Dict[str, str] = {
    # --- Advertising / Retargeting ---
    "doubleclick.net":          "ads",
    "criteo.com":               "ads",
    "taboola.com":              "ads",
    "outbrain.com":             "ads",
    "adserver.biz":             "ads",
    "ad-for-kids.net":          "ads",
    "moat.com":                 "ads",
    "rubiconproject.com":       "ads",
    "pubmatic.com":             "ads",
    # --- Analytics / Measurement ---
    "google-analytics.com":     "analytics",
    "hotjar.com":               "analytics",
    "mixpanel.com":             "analytics",
    "segment.io":               "analytics",
    "comscore.com":             "analytics",
    "nielsen.com":              "analytics",
    "chartbeat.com":            "analytics",
    "amplitude.com":            "analytics",
    "heap.io":                  "analytics",
    "kid-analytics.com":        "analytics",
    "telemetry.devtools.io":    "analytics",
    # --- Social Tracking ---
    "facebook.com/tr":          "social",
    "linkedin.com/tracking":    "social",
    "twitter.com/i/adsct":      "social",
    "pinterest.com/ct":         "social",
    "tiktok.com/pixel":         "social",
    # --- Attribution / Mobile ---
    "appsflyer.com":            "attribution",
    "adjust.com":               "attribution",
    "branch.io":                "attribution",
    "unity3d.com/tracking":     "attribution",
    # --- Malicious / High-Risk ---
    "malware-tracker.com":      "malicious",
    "crypto-miner.js":          "malicious",
    "keylogger-cdn.net":        "malicious",
    "tracking.xyz":             "malicious",
    "pixel-spy.com":            "malicious",
    "data-harvest.net":         "malicious",
    # --- IoT / Device Cloud ---
    "cloud-iot.smarthome.com":  "iot",
    # --- Proprietary / Opaque ---
    "proprietary-pixel.dataguard.io": "opaque",
}

# Risk weight per tracker category
TRACKER_RISK_WEIGHT: Dict[str, float] = {
    "ads":          0.7,
    "analytics":    0.3,
    "social":       0.6,
    "attribution":  0.5,
    "malicious":    1.0,
    "iot":          0.6,
    "opaque":       0.8,
}


def classify_tracker(domain: str) -> str:
    """Return the category of a tracker domain, or 'unknown'."""
    return TRACKER_CATEGORIES.get(domain, "unknown")


def compute_tracker_risk(trackers: List[str]) -> float:
    """
    Compute an aggregate tracker risk score in [0.0, 1.0].
    Considers the count, diversity, and risk weight of each tracker category.
    """
    if not trackers:
        return 0.0

    total_weight = 0.0
    categories_seen = set()
    for t in trackers:
        cat = classify_tracker(t)
        categories_seen.add(cat)
        total_weight += TRACKER_RISK_WEIGHT.get(cat, 0.5)

    # Blend: raw weight + diversity bonus (more categories = higher risk)
    raw = min(total_weight / 5.0, 1.0)           # cap at 5 weighted trackers
    diversity = min(len(categories_seen) / 4.0, 1.0)  # cap at 4 distinct types
    return 0.7 * raw + 0.3 * diversity


# =============================================================================
# Privacy Policy Deception Detection (v2 — Weighted + Contradiction-Aware)
# =============================================================================

# Weighted danger signals: higher weight = more severe indicator
# Weights reflect real-world privacy harm severity
DECEPTIVE_PHRASES: Dict[str, float] = {
    # Severe (weight 1.0) — unambiguous data abuse
    "becomes our property":                  1.0,
    "any purpose we see fit":                1.0,
    "financial data for any purpose":        1.0,
    "without warrant":                       1.0,
    "read email contents":                   1.0,
    "unencrypted on our servers":            1.0,
    # High (weight 0.8) — significant risk
    "sold to third parties":                 0.8,
    "disclosed to law enforcement":          0.8,
    "stored indefinitely":                   0.8,
    "voice recordings stored":               0.8,
    "detailed user profiles":                0.8,
    "shared with trusted protection partners": 0.8,
    "friend lists":                          0.8,
    "contacts, photos, and location":        0.8,
    # Medium (weight 0.5) — concerning but context-dependent
    "ad targeting":                          0.5,
    "school name":                           0.5,
    "behavioral profiles":                   0.5,
    "real-time bidding":                     0.5,
    "cross-site":                            0.5,
    "device fingerprint":                    0.5,
}

# Weighted safe signals: higher weight = stronger transparency guarantee
SAFE_PHRASES: Dict[str, float] = {
    # Strong (weight 1.0) — concrete data protection claims
    "end-to-end encrypted":                  1.0,
    "no data sold":                          1.0,
    "do not collect":                        1.0,
    "no third-party tracking":               1.0,
    "publicly auditable":                    1.0,
    # High (weight 0.8) — regulatory/audit compliance
    "gdpr compliant":                        0.8,
    "ferpa":                                 0.8,
    "soc2":                                  0.8,
    "hipaa":                                 0.8,
    "pci-dss":                               0.8,
    # Medium (weight 0.5) — partial/procedural protections
    "no tracking":                           0.5,
    "open source":                           0.5,
    "anonymized":                            0.5,
    "opt out":                               0.5,
    "can disable":                           0.5,
    "essential cookies":                     0.5,
    "no accounts required":                  0.5,
}


def detect_policy_deception(privacy_policy: str) -> float:
    """
    Score how deceptive a privacy policy is (v2 — weighted + contradiction-aware).

    Returns a value in [-1.0, 1.0]:
        > +0.3  = deceptive  (danger signals, possibly masked by safe language)
        < -0.3  = transparent (strong, credible safety guarantees)
        Near 0  = mixed or neutral

    Improvements over v1:
        - Phrase weights: severe danger words score higher than mild ones
        - Contradiction bonus: safe + danger phrases together = amplified deception
          (mimics real deceptive policies that bury danger in reassuring text)
        - Density normalization: long policies are not over-penalized
    """
    if not privacy_policy or not privacy_policy.strip():
        return 0.0

    text = privacy_policy.lower()
    word_count = max(len(text.split()), 1)

    # Weighted phrase scoring
    danger_score = sum(
        weight for phrase, weight in DECEPTIVE_PHRASES.items() if phrase in text
    )
    safe_score = sum(
        weight for phrase, weight in SAFE_PHRASES.items() if phrase in text
    )

    danger_hits = sum(1 for phrase in DECEPTIVE_PHRASES if phrase in text)
    safe_hits   = sum(1 for phrase in SAFE_PHRASES if phrase in text)

    # Contradiction amplification:
    # Safe + danger phrases together is MORE deceptive than danger alone.
    # Classic deceptive policy pattern: "We protect your data..." + "stored indefinitely".
    if danger_hits > 0 and safe_hits > 0:
        # Overlap amplification: contradiction is the key deception signal
        contradiction_factor = 1.0 + min(safe_score / (danger_score + 1e-6), 0.5)
        raw = (danger_score * contradiction_factor - safe_score * 0.2) / 5.0
    elif danger_hits > 0:
        raw = danger_score / 5.0
    elif safe_hits > 0:
        raw = -safe_score / 5.0
    else:
        raw = 0.0

    # Density normalization: penalize long policies less (longer = more chances for hits)
    density_factor = min(1.0, 100.0 / word_count)  # scales down for very long policies
    normalized = raw * (0.7 + 0.3 * density_factor)  # soft normalization

    return max(-1.0, min(1.0, float(normalized)))


# =============================================================================
# Constants
# =============================================================================

CLASSIFICATION_ACTIONS = {"mark_safe", "mark_risky", "mark_dangerous"}
INVESTIGATION_ACTIONS  = {"inspect_trackers", "inspect_policy"}
VALID_ACTIONS = CLASSIFICATION_ACTIONS | INVESTIGATION_ACTIONS
RISK_LEVELS  = ["mark_safe", "mark_risky", "mark_dangerous"]

# Maximum characters of privacy policy text exposed in observations
MAX_POLICY_LENGTH = 300


def _generate_reasoning_hint(cookies: int, trackers: list, https: bool,
                              trackers_visible: bool, policy_visible: bool) -> str:
    """Generate a short deterministic reasoning hint based on visible signals."""
    parts = []
    if cookies > 30:
        parts.append("Very high cookie count")
    elif cookies > 15:
        parts.append("Moderate cookie count")
    elif cookies <= 3:
        parts.append("Minimal cookies")
    if not https:
        parts.append("No HTTPS")
    if trackers_visible:
        n = len(trackers)
        if n == 0:
            parts.append("No trackers")
        elif n >= 4:
            parts.append(f"High tracker count ({n})")
        else:
            parts.append(f"{n} tracker(s)")
    else:
        parts.append("Trackers not inspected")
    if not policy_visible:
        parts.append("Policy not inspected")
    return " | ".join(parts) if parts else "No notable signals"


# =============================================================================
# Inline Dataset — 36 simulated websites with typed trackers
# =============================================================================

EASY_SITES = [
    {
        "name": "GreenLeaf Blog",
        "cookies": 2, "trackers": [], "https": True,
        "privacy_policy": "We do not collect any personal data. This blog uses only essential cookies for page layout preferences. No data is shared with third parties. No analytics scripts are loaded.",
        "ground_truth": "mark_safe", "difficulty": "easy",
    },
    {
        "name": "OpenRecipes",
        "cookies": 1, "trackers": [], "https": True,
        "privacy_policy": "We use a single session cookie to remember your dark-mode preference. No tracking, no ads, no data collection. Fully static site hosted on GitHub Pages.",
        "ground_truth": "mark_safe", "difficulty": "easy",
    },
    {
        "name": "LocalWeather",
        "cookies": 3, "trackers": [], "https": True,
        "privacy_policy": "Location data is used only to display weather in your browser via the Geolocation API. We store nothing on our servers. All processing is client-side JavaScript.",
        "ground_truth": "mark_safe", "difficulty": "easy",
    },
    {
        "name": "QuickMath Tools",
        "cookies": 0, "trackers": [], "https": True,
        "privacy_policy": "Zero cookies. Zero tracking. Zero data collection. This is a purely static calculator tool with no server-side component whatsoever.",
        "ground_truth": "mark_safe", "difficulty": "easy",
    },
    {
        "name": "AdTracker Central",
        "cookies": 45, "trackers": ["doubleclick.net", "facebook.com/tr", "google-analytics.com", "criteo.com", "taboola.com", "outbrain.com"], "https": False,
        "privacy_policy": "We collect browsing data, device fingerprints, IP addresses, screen resolution, and installed fonts. This information is shared with our 200+ advertising partners for cross-site targeted advertising and real-time bidding.",
        "ground_truth": "mark_dangerous", "difficulty": "easy",
    },
    {
        "name": "FreeGames4All",
        "cookies": 38, "trackers": ["tracking.xyz", "adserver.biz", "pixel-spy.com", "data-harvest.net"], "https": False,
        "privacy_policy": "Your data including contacts, photos, and location is collected and sold to third parties for marketing analysis. We make no guarantees about data security or breach notification.",
        "ground_truth": "mark_dangerous", "difficulty": "easy",
    },
    {
        "name": "ShadyDeals Shop",
        "cookies": 52, "trackers": ["malware-tracker.com", "crypto-miner.js", "keylogger-cdn.net"], "https": False,
        "privacy_policy": "All information entered on this site becomes our property. We may use your financial data for any purpose we see fit, including resale to data brokers and debt collectors.",
        "ground_truth": "mark_dangerous", "difficulty": "easy",
    },
    {
        "name": "NewsDaily",
        "cookies": 12, "trackers": ["google-analytics.com"], "https": True,
        "privacy_policy": "We use Google Analytics to understand readership patterns. Cookies track page views and session duration. Data is anonymized after 30 days. You can opt out via our cookie banner.",
        "ground_truth": "mark_risky", "difficulty": "easy",
    },
    {
        "name": "ShopMart Online",
        "cookies": 15, "trackers": ["facebook.com/tr", "google-analytics.com"], "https": True,
        "privacy_policy": "We collect purchase history, browsing behavior, and email address for personalized marketing campaigns. Data is shared with our payment processor (Stripe) and analytics provider (Google).",
        "ground_truth": "mark_risky", "difficulty": "easy",
    },
    {
        "name": "TravelBuddy",
        "cookies": 18, "trackers": ["hotjar.com", "mixpanel.com"], "https": True,
        "privacy_policy": "We record user sessions via Hotjar for UX research. Location data and search queries are stored for travel recommendations. Behavioral analytics powered by Mixpanel. Data retained for 12 months.",
        "ground_truth": "mark_risky", "difficulty": "easy",
    },
    {
        "name": "PublicLibrary Portal",
        "cookies": 1, "trackers": [], "https": True,
        "privacy_policy": "Library card number and borrowing history are stored locally. No personal data leaves your browser. Government-funded public service with no commercial interests.",
        "ground_truth": "mark_safe", "difficulty": "easy",
    },
    {
        "name": "ViralClickBait",
        "cookies": 40, "trackers": ["doubleclick.net", "taboola.com", "outbrain.com", "facebook.com/tr", "moat.com"], "https": False,
        "privacy_policy": "Pageviews, scroll depth, mouse movements, and click patterns are recorded. Data packaged into behavioral profiles and auctioned via real-time bidding to the highest-paying advertiser.",
        "ground_truth": "mark_dangerous", "difficulty": "easy",
    },
]

MEDIUM_SITES = [
    {
        "name": "HealthTrack Pro",
        "cookies": 8, "trackers": ["google-analytics.com", "amplitude.com"], "https": True,
        "privacy_policy": "Health data (heart rate, steps, sleep patterns) is encrypted with AES-256 and stored on HIPAA-compliant servers. We use analytics for crash reporting and feature usage metrics. Anonymized aggregate data may be shared with university research partners.",
        "ground_truth": "mark_risky", "difficulty": "medium",
    },
    {
        "name": "EduLearn Platform",
        "cookies": 5, "trackers": [], "https": True,
        "privacy_policy": "Student data is protected under FERPA regulations. We collect grades, assignment progress, and login timestamps. Data is shared only with the student's enrolled educational institution via encrypted API. Parents can request data deletion.",
        "ground_truth": "mark_safe", "difficulty": "medium",
    },
    {
        "name": "FinanceWiz",
        "cookies": 20, "trackers": ["segment.io", "hotjar.com", "heap.io"], "https": True,
        "privacy_policy": "Financial transaction data is encrypted with AES-256 in transit and at rest. We use Segment for event tracking and Hotjar session recordings exclusively for fraud detection and security audit trails. No data sold. SOC2 Type II certified.",
        "ground_truth": "mark_risky", "difficulty": "medium",
    },
    {
        "name": "SocialBuzz",
        "cookies": 25, "trackers": ["facebook.com/tr", "doubleclick.net", "linkedin.com/tracking", "tiktok.com/pixel"], "https": True,
        "privacy_policy": "We personalize your social feed using your interests, connection graph, and cross-app browsing history. Behavioral data is shared with advertising partners for lookalike audience targeting. You can manage some ad preferences in settings.",
        "ground_truth": "mark_dangerous", "difficulty": "medium",
    },
    {
        "name": "CloudNotes",
        "cookies": 4, "trackers": [], "https": True,
        "privacy_policy": "All notes are protected with end-to-end encryption using your passphrase. We cannot read your content even if compelled by law. Only minimal metadata (creation timestamp, byte size) is stored for synchronization.",
        "ground_truth": "mark_safe", "difficulty": "medium",
    },
    {
        "name": "StreamFlix",
        "cookies": 22, "trackers": ["google-analytics.com", "comscore.com", "nielsen.com"], "https": True,
        "privacy_policy": "We collect viewing history, device identifiers, playback quality metrics, and content preferences for personalized recommendations. Viewership data is shared with Nielsen and comScore for industry audience measurement reports.",
        "ground_truth": "mark_risky", "difficulty": "medium",
    },
    {
        "name": "JobHunter",
        "cookies": 14, "trackers": ["linkedin.com/tracking", "twitter.com/i/adsct"], "https": True,
        "privacy_policy": "Resume data including work history, skills, and salary expectations is stored for AI-powered job matching. Employers can view your professional profile. Application data shared with recruitment agency partners. Profile visible in LinkedIn-style search.",
        "ground_truth": "mark_risky", "difficulty": "medium",
    },
    {
        "name": "PetCare Community",
        "cookies": 3, "trackers": [], "https": True,
        "privacy_policy": "Forum posts are public and indexed by search engines. We store your chosen username and email (for password resets only). No data is sold or shared commercially. Community moderation by volunteer pet owners.",
        "ground_truth": "mark_safe", "difficulty": "medium",
    },
    {
        "name": "GamingHub",
        "cookies": 30, "trackers": ["unity3d.com/tracking", "facebook.com/tr", "appsflyer.com", "adjust.com"], "https": True,
        "privacy_policy": "We collect gameplay telemetry, device hardware identifiers (IDFA/GAID), social graph connections, and in-game purchase history for matchmaking, ad targeting, and cross-device attribution. Data shared with Unity Ads and Facebook Audience Network.",
        "ground_truth": "mark_dangerous", "difficulty": "medium",
    },
    {
        "name": "FitnessTracker",
        "cookies": 10, "trackers": ["google-analytics.com"], "https": True,
        "privacy_policy": "Workout metrics (distance, pace, calories, heart rate zones) are stored securely. GPS routes tracked during outdoor runs. Data may be shared with health insurance partners for premium discount programs, with your opt-in consent.",
        "ground_truth": "mark_risky", "difficulty": "medium",
    },
    {
        "name": "CryptoExchange Pro",
        "cookies": 18, "trackers": ["segment.io", "mixpanel.com"], "https": True,
        "privacy_policy": "KYC identity documents (passport, driver's license) are stored for regulatory compliance. Transaction history retained indefinitely. We share data with tax authorities as required by law. Two-factor authentication mandatory.",
        "ground_truth": "mark_risky", "difficulty": "medium",
    },
    {
        "name": "DatingMatch",
        "cookies": 28, "trackers": ["facebook.com/tr", "appsflyer.com", "pinterest.com/ct"], "https": True,
        "privacy_policy": "We collect detailed personal preferences including age, body type, ethnicity, religion, and sexual orientation for matching. Location tracked in real-time. Profile photos analyzed with facial recognition. Data shared with advertising partners.",
        "ground_truth": "mark_dangerous", "difficulty": "medium",
    },
]

HARD_SITES = [
    {
        "name": "PrivacyFirst VPN",
        "cookies": 2, "trackers": [], "https": True,
        "privacy_policy": "We are committed to your privacy and do not collect personal data. Your browsing traffic is routed through our zero-knowledge servers for maximum protection. Aggregated bandwidth patterns may be shared with network infrastructure optimization partners to improve service quality. Connection timestamps and session durations are retained for 90 days for service reliability.",
        "ground_truth": "mark_risky", "difficulty": "hard",
    },
    {
        "name": "SecureChat App",
        "cookies": 3, "trackers": [], "https": True,
        "privacy_policy": "Messages are encrypted in transit using industry-standard TLS 1.3. However, communication metadata including sender identity, recipient identity, timestamps, message frequency, IP addresses, and complete contact lists are stored indefinitely on our servers for service reliability. This data may be disclosed to law enforcement agencies without warrant upon formal request.",
        "ground_truth": "mark_dangerous", "difficulty": "hard",
    },
    {
        "name": "BankSecure Online",
        "cookies": 28, "trackers": [], "https": True,
        "privacy_policy": "All 28 cookies are strictly essential for multi-factor authentication challenge-response, CSRF token rotation, session management, and real-time fraud detection scoring. No third-party tracking scripts are loaded. Data encrypted at rest with AES-256-GCM. SOC2 Type II and PCI-DSS Level 1 certified. No data shared externally under any circumstances.",
        "ground_truth": "mark_safe", "difficulty": "hard",
    },
    {
        "name": "DataGuard Analytics",
        "cookies": 8, "trackers": ["proprietary-pixel.dataguard.io"], "https": True,
        "privacy_policy": "Your privacy is our top priority. We protect your data by analyzing it thoroughly using our proprietary privacy-enhancing technology. Our advanced algorithms create detailed user profiles including browsing patterns, purchase intent, and demographic segments — all to guard against fraud and identity theft. These comprehensive profiles are shared with our network of trusted protection partners for enhanced security coverage.",
        "ground_truth": "mark_dangerous", "difficulty": "hard",
    },
    {
        "name": "DevTools IDE",
        "cookies": 6, "trackers": ["telemetry.devtools.io"], "https": True,
        "privacy_policy": "Open source IDE released under MIT license. Optional telemetry (disabled by default) collects crash stack traces and anonymized feature usage frequency. All telemetry data is anonymized with k-anonymity (k≥50). You can disable telemetry in Settings → Privacy. Full source code is publicly auditable on GitHub.",
        "ground_truth": "mark_safe", "difficulty": "hard",
    },
    {
        "name": "FreeMail Express",
        "cookies": 5, "trackers": [], "https": False,
        "privacy_policy": "FreeMail provides free email service supported by advertising revenue. Email content is automatically scanned using NLP algorithms for spam filtering and contextual ad targeting. We may read email contents to provide relevant inline advertisements. All emails are stored unencrypted on our servers in plaintext format. Backup copies retained for 5 years after account deletion.",
        "ground_truth": "mark_dangerous", "difficulty": "hard",
    },
    {
        "name": "MediaCorp News",
        "cookies": 20, "trackers": ["google-analytics.com", "chartbeat.com"], "https": True,
        "privacy_policy": "GDPR and CCPA compliant. All tracking requires explicit opt-in consent via granular cookie banner with reject-all option. Analytics used exclusively for editorial content performance decisions — never advertising. Data automatically purged after 90 days. No data is sold or shared with advertisers. Appointed Data Protection Officer contactable at privacy@mediacorp.com.",
        "ground_truth": "mark_risky", "difficulty": "hard",
    },
    {
        "name": "KiddyFun Games",
        "cookies": 15, "trackers": ["kid-analytics.com", "ad-for-kids.net"], "https": True,
        "privacy_policy": "Welcome to the most fun games for children ages 4-12! To personalize the magical experience, we collect each child's age, school name, grade level, interests, friend lists, and gameplay behavioral patterns. Fun stickers awarded based on engagement time! Parents can email us for COPPA data access requests (response within 90 business days).",
        "ground_truth": "mark_dangerous", "difficulty": "hard",
    },
    {
        "name": "WikiFacts",
        "cookies": 1, "trackers": [], "https": True,
        "privacy_policy": "Community-edited open encyclopedia. We log IP addresses temporarily (72 hours) for anti-vandalism detection. Edit history is permanent and publicly visible by design for accountability. No accounts required to read. No advertising. Operated by a registered nonprofit foundation.",
        "ground_truth": "mark_safe", "difficulty": "hard",
    },
    {
        "name": "SmartHome Hub",
        "cookies": 7, "trackers": ["cloud-iot.smarthome.com"], "https": True,
        "privacy_policy": "SmartHome Hub collects continuous sensor data from all connected devices: temperature, humidity, motion patterns, door lock status, audio from smart speakers, and video from security cameras. All data processed in our cloud infrastructure for home automation rules. Voice recordings stored for 24 months to improve voice recognition accuracy. Video clips stored for 12 months. Data center located in jurisdiction with no data protection laws.",
        "ground_truth": "mark_dangerous", "difficulty": "hard",
    },
    {
        "name": "SpeedTest Global",
        "cookies": 4, "trackers": ["pubmatic.com", "rubiconproject.com"], "https": True,
        "privacy_policy": "Our free internet speed test requires hardware fingerprinting (GPU model, CPU cores, screen resolution, installed fonts, WebGL renderer) to accurately calibrate bandwidth measurements. This calibration fingerprint is shared with measurement verification partners. Test results and your approximate location (city-level) are aggregated into public broadband quality reports.",
        "ground_truth": "mark_risky", "difficulty": "hard",
    },
    {
        "name": "ConsentShield CMP",
        "cookies": 22, "trackers": ["google-analytics.com", "segment.io"], "https": True,
        "privacy_policy": "ConsentShield is a Consent Management Platform. The 22 cookies store your explicit consent choices for each tracker category across partner websites. We use analytics solely to measure consent banner interaction rates. No personal data is collected beyond consent records. IAB TCF 2.2 certified. Consent records stored as legally required audit trail.",
        "ground_truth": "mark_safe", "difficulty": "hard",
    },
]


# =============================================================================
# Task Registry
# =============================================================================

TASKS = {
    "easy":   {"id": "easy",   "description": "Classify websites with obvious privacy indicators",                      "sites": EASY_SITES},
    "medium": {"id": "medium", "description": "Classify websites with mixed privacy signals requiring reasoning",        "sites": MEDIUM_SITES},
    "hard":   {"id": "hard",   "description": "Classify websites with deceptive or contradictory privacy patterns",      "sites": HARD_SITES},
}

ALL_TASK_IDS = list(TASKS.keys())


# =============================================================================
# Procedural Site Generator (prevents dataset memorization)
# =============================================================================
# Difficulty-parameterized distributions for site generation.
# Each episode can optionally draw from generated sites instead of fixed ones.

_PROC_TRACKER_POOL: Dict[str, str] = {
    # (domain, category) pairs drawn from the taxonomy
    "doubleclick.net": "ads",   "criteo.com": "ads",      "taboola.com": "ads",
    "outbrain.com": "ads",      "rubiconproject.com": "ads", "pubmatic.com": "ads",
    "moat.com": "ads",          "adserver.biz": "ads",
    "google-analytics.com": "analytics", "hotjar.com": "analytics",
    "mixpanel.com": "analytics", "segment.io": "analytics", "amplitude.com": "analytics",
    "heap.io": "analytics",     "chartbeat.com": "analytics", "comscore.com": "analytics",
    "facebook.com/tr": "social", "linkedin.com/tracking": "social",
    "twitter.com/i/adsct": "social", "tiktok.com/pixel": "social",
    "appsflyer.com": "attribution", "adjust.com": "attribution",
    "unity3d.com/tracking": "attribution",
    "malware-tracker.com": "malicious", "keylogger-cdn.net": "malicious",
    "tracking.xyz": "malicious",  "pixel-spy.com": "malicious",
    "data-harvest.net": "malicious",
    "cloud-iot.smarthome.com": "iot",
    "proprietary-pixel.dataguard.io": "opaque",
}
_PROC_TRACKER_DOMAINS = list(_PROC_TRACKER_POOL.keys())

_SAFE_CLAUSES = [
    "We do not collect any personal data.",
    "No tracking scripts are loaded on this site.",
    "Data is end-to-end encrypted and cannot be read by our servers.",
    "We are fully GDPR compliant. All tracking requires explicit opt-in.",
    "No data is sold or shared with third parties.",
    "You can opt out of all data collection at any time.",
    "Open source and publicly auditable.",
    "Only essential cookies are used for session management.",
    "We are a registered nonprofit with no commercial advertising interest.",
    "Data is anonymized within 24 hours and never linked to individuals.",
]
_DANGER_CLAUSES = [
    "Your data becomes our property upon registration.",
    "We may use your financial data for any purpose we see fit.",
    "Information is shared with our 200+ advertising partners for real-time bidding.",
    "Contact lists, photos, and location are collected for analytics.",
    "Voice recordings are stored for 24 months to improve recognition accuracy.",
    "We may disclose data to law enforcement without warrant upon formal request.",
    "Detailed user profiles are created including browsing patterns and purchase intent.",
    "All email content is scanned for contextual ad targeting.",
    "Data is stored indefinitely on servers with no data protection laws.",
    "Device fingerprints including GPU, fonts, and screen resolution are collected.",
    "Friend lists and behavioral patterns are shared with trusted protection partners.",
]
_NEUTRAL_CLAUSES = [
    "We use analytics to monitor service reliability.",
    "Session identifiers are stored for authentication purposes.",
    "IP addresses are logged temporarily for anti-abuse detection.",
    "Cookies are used to remember your preferences.",
    "Third-party payment processors handle transaction data.",
]

_DIFFICULTY_PARAMS: Dict[str, dict] = {
    "easy": {
        "cookie_range": (0, 50),      # wide spread — easy to read signal
        "tracker_range": (0, 6),
        "https_probs": {"mark_safe": 0.95, "mark_risky": 0.85, "mark_dangerous": 0.15},
        "safe_clauses":    {"mark_safe": (3, 5), "mark_risky": (1, 2), "mark_dangerous": (0, 1)},
        "danger_clauses":  {"mark_safe": (0, 0), "mark_risky": (1, 2), "mark_dangerous": (3, 5)},
        "neutral_clauses": (1, 2),
    },
    "medium": {
        "cookie_range": (3, 30),      # overlapping ranges — harder to distinguish
        "tracker_range": (0, 4),
        "https_probs": {"mark_safe": 0.90, "mark_risky": 0.75, "mark_dangerous": 0.5},
        "safe_clauses":    {"mark_safe": (2, 4), "mark_risky": (1, 3), "mark_dangerous": (0, 2)},
        "danger_clauses":  {"mark_safe": (0, 1), "mark_risky": (1, 3), "mark_dangerous": (2, 4)},
        "neutral_clauses": (1, 3),
    },
    "hard": {
        "cookie_range": (1, 30),      # surface signals are misleading
        "tracker_range": (0, 3),
        "https_probs": {"mark_safe": 0.90, "mark_risky": 0.80, "mark_dangerous": 0.70},
        "safe_clauses":    {"mark_safe": (3, 5), "mark_risky": (2, 4), "mark_dangerous": (2, 4)},
        "danger_clauses":  {"mark_safe": (0, 0), "mark_risky": (1, 3), "mark_dangerous": (3, 5)},
        "neutral_clauses": (2, 4),
    },
}


def _assign_ground_truth(cookies: int, trackers: list, https: bool,
                         policy: str, difficulty: str, rng: random.Random) -> str:
    """
    Deterministically assign ground truth from site signals.
    Used by the procedural generator to ensure label consistency.
    """
    tracker_risk = compute_tracker_risk(trackers)
    deception = detect_policy_deception(policy)

    if difficulty == "easy":
        # Clear signals dominate
        if cookies > 30 or tracker_risk > 0.6 or not https:
            return "mark_dangerous" if (cookies > 40 or tracker_risk > 0.7) else "mark_risky"
        if cookies <= 3 and tracker_risk <= 0.1 and https:
            return "mark_safe"
        return "mark_risky"

    elif difficulty == "medium":
        score = 0
        if cookies > 20: score += 2
        elif cookies > 8: score += 1
        if tracker_risk > 0.5: score += 2
        elif tracker_risk > 0.2: score += 1
        if not https: score += 2
        if deception > 0.3: score += 2
        if score >= 5: return "mark_dangerous"
        if score >= 2: return "mark_risky"
        return "mark_safe"

    else:  # hard — policy text is the primary signal, surface features are misleading
        if deception > 0.5:
            return "mark_dangerous"
        if deception > 0.2 or tracker_risk > 0.4:
            return "mark_risky"
        if deception < -0.3 and https:
            return "mark_safe"
        # Ambiguous — use cookie count as tiebreaker
        if cookies > 20: return "mark_risky"
        return "mark_safe"


def generate_site(difficulty: str, rng: random.Random, site_index: int = 0) -> dict:
    """
    Procedurally generate a single website with consistent difficulty characteristics.

    Guarantees:
    - Ground truth is derived FROM actual signals (not pre-assigned)
    - Difficulty controls signal clarity, not just label distribution
    - Repeated calls with different RNG states produce structurally varied sites
    - Works as a drop-in replacement for static EASY/MEDIUM/HARD_SITES entries
    """
    params = _DIFFICULTY_PARAMS[difficulty]

    # Sample label first (balanced across 3 classes)
    label = rng.choice(["mark_safe", "mark_risky", "mark_dangerous"])

    # Cookies: difficulty-gated ranges, label-biased
    lo, hi = params["cookie_range"]
    if label == "mark_safe":
        cookies = rng.randint(lo, max(lo, hi // 3))
    elif label == "mark_risky":
        cookies = rng.randint(lo, max(lo, hi * 2 // 3))
    else:  # dangerous
        cookies = rng.randint(hi // 2, hi)
    cookies = max(0, cookies)

    # HTTPS: label-biased probability
    https_prob = params["https_probs"][label]
    https = rng.random() < https_prob

    # Trackers: sample from taxonomy pool
    t_lo, t_hi = params["tracker_range"]
    n_trackers = rng.randint(t_lo, t_hi)
    if label == "mark_safe":
        n_trackers = min(n_trackers, t_hi // 2)
    elif label == "mark_dangerous":
        n_trackers = max(n_trackers, t_hi // 2)
    trackers = rng.sample(_PROC_TRACKER_DOMAINS, min(n_trackers, len(_PROC_TRACKER_DOMAINS)))

    # Privacy policy: compose from clause pools
    s_lo, s_hi = params["safe_clauses"][label]
    d_lo, d_hi = params["danger_clauses"][label]
    n_lo, n_hi = params["neutral_clauses"]

    safe_pool    = rng.sample(_SAFE_CLAUSES,    min(rng.randint(s_lo, s_hi), len(_SAFE_CLAUSES)))
    danger_pool  = rng.sample(_DANGER_CLAUSES,  min(rng.randint(d_lo, d_hi), len(_DANGER_CLAUSES)))
    neutral_pool = rng.sample(_NEUTRAL_CLAUSES, min(rng.randint(n_lo, n_hi), len(_NEUTRAL_CLAUSES)))

    # Shuffle clause order (prevents positional shortcuts)
    all_clauses = safe_pool + danger_pool + neutral_pool
    rng.shuffle(all_clauses)
    policy = " ".join(all_clauses)

    # Derive ground truth FROM the generated signals (not the sampled label)
    # This ensures label consistency — avoids mismatches from stochastic sampling
    ground_truth = _assign_ground_truth(cookies, trackers, https, policy, difficulty, rng)

    return {
        "name": f"proc_site_{difficulty}_{site_index:04d}",
        "cookies": cookies,
        "trackers": trackers,
        "https": https,
        "privacy_policy": policy,
        "ground_truth": ground_truth,
        "difficulty": difficulty,
        "procedural": True,   # flag to distinguish from static sites
    }


# =============================================================================
# Reward Shaping (v4) — Corrected reward design, no degenerate inspection harvest
# =============================================================================

# FIX 1: Inspection now costs a small step penalty rather than granting free reward.
# This prevents the degenerate policy of inspect→inspect→classify_anything for +0.4.
# Agents must decide whether revealing information is worth the step cost.
INSPECT_STEP_COST = -0.02

# FIX 2: Informed decision bonus is now CONDITIONAL on correct classification.
# Inspecting then guessing wrong gives NO bonus — only inspecting AND being right does.
# This forces the agent to actually USE the revealed information.
INFORMED_DECISION_BONUS = 0.2

# Max steps per website before forced termination.
# Hard difficulty gets fewer steps to force tighter tradeoffs.
MAX_STEPS_PER_SITE = 5
MAX_STEPS_HARD = 4  # less margin on hard — agent must inspect selectively

# Per-step cost applied on every action (smooth efficiency incentive).
# Replaces the hard cliff of a single timeout penalty.
STEP_COST = -0.02

# Penalty for exceeding step limit without classifying
TIMEOUT_PENALTY = -0.5

# FIX 6: Cookie noise range to prevent exact-value memorization
COOKIE_NOISE_RANGE = 2  # ±2 cookies added at reset (sim sensor uncertainty)

# Category index for get_vector_obs()
CATEGORY_INDEX: Dict[str, int] = {
    "ads": 0, "analytics": 1, "social": 2, "attribution": 3,
    "malicious": 4, "iot": 5, "opaque": 6, "unknown": 6,
}


def compute_reward(predicted: str, ground_truth: str,
                   trackers: List[str], https: bool,
                   privacy_policy: str, difficulty: str,
                   trackers_inspected: bool = False,
                   policy_inspected: bool = False) -> dict:
    """
    Compute a shaped reward with multiple signal components.

    Now includes an informed-decision bonus: if the agent inspected trackers
    and/or policy before classifying, it gets an extra reward boost.

    Returns a dict with:
        base_reward:             Ordinal distance between predicted and true label
        tracker_bonus:           Reward for correct tracker-awareness
        https_bonus:             Reward for HTTPS-awareness
        deception_bonus:         Reward for detecting deceptive policy language
        informed_decision_bonus: Bonus for inspecting before deciding
        total_reward:            Sum of all components (capped to [-1.0, 1.7])
        breakdown:               Human-readable explanation string
    """
    # ── Base reward: ordinal distance ──
    if predicted == ground_truth:
        base = 1.0
    else:
        pred_idx  = RISK_LEVELS.index(predicted)
        truth_idx = RISK_LEVELS.index(ground_truth)
        distance  = abs(pred_idx - truth_idx)
        base = 0.3 if distance == 1 else -1.0

    # ── Tracker awareness bonus (+0.15 max) ──
    tracker_risk = compute_tracker_risk(trackers)
    tracker_bonus = 0.0

    if tracker_risk >= 0.6 and predicted in ("mark_risky", "mark_dangerous"):
        tracker_bonus = 0.15
    elif tracker_risk <= 0.1 and predicted == "mark_safe":
        tracker_bonus = 0.15
    elif tracker_risk >= 0.6 and predicted == "mark_safe":
        tracker_bonus = -0.10

    # ── HTTPS awareness bonus (+0.10 max) ──
    https_bonus = 0.0
    if not https and predicted in ("mark_risky", "mark_dangerous"):
        https_bonus = 0.10
    elif not https and predicted == "mark_safe":
        https_bonus = -0.05

    # ── Policy deception bonus (+0.25 max, hard tasks only) ──
    deception_bonus = 0.0
    deception_score = detect_policy_deception(privacy_policy)

    if difficulty == "hard":
        if deception_score > 0.3 and predicted in ("mark_risky", "mark_dangerous"):
            deception_bonus = 0.25
        elif deception_score > 0.3 and predicted == "mark_safe":
            deception_bonus = -0.15
        elif deception_score < -0.3 and predicted == "mark_safe":
            deception_bonus = 0.15
    elif difficulty == "medium":
        if deception_score > 0.3 and predicted in ("mark_risky", "mark_dangerous"):
            deception_bonus = 0.10
        elif deception_score < -0.3 and predicted == "mark_safe":
            deception_bonus = 0.05

    # ── Informed decision bonus (+0.2 max, ONLY if classification is correct) ──
    # FIX 2: bonus is now gated on correctness to prevent inspect→guess_anything exploit.
    # Inspecting reveals real information; that information should improve accuracy.
    informed_bonus = 0.0
    correct = (predicted == ground_truth)
    if correct:
        if trackers_inspected and policy_inspected:
            informed_bonus = INFORMED_DECISION_BONUS       # full investigation + correct
        elif trackers_inspected or policy_inspected:
            informed_bonus = INFORMED_DECISION_BONUS / 2  # partial investigation + correct
    # If wrong even after inspection: no informed bonus (agent failed to use the info)

    # ── Sum and clamp ──
    total = base + tracker_bonus + https_bonus + deception_bonus + informed_bonus
    total = max(-1.0, min(1.7, total))

    # ── Build breakdown string ──
    parts = [f"base={base:+.2f}"]
    if tracker_bonus != 0:
        parts.append(f"tracker={tracker_bonus:+.2f}")
    if https_bonus != 0:
        parts.append(f"https={https_bonus:+.2f}")
    if deception_bonus != 0:
        parts.append(f"deception={deception_bonus:+.2f}")
    if informed_bonus != 0:
        parts.append(f"informed={informed_bonus:+.2f}")

    return {
        "base_reward":             float(base),
        "tracker_bonus":           float(tracker_bonus),
        "https_bonus":             float(https_bonus),
        "deception_bonus":         float(deception_bonus),
        "informed_decision_bonus": float(informed_bonus),
        "total_reward":            float(total),
        "breakdown":               " | ".join(parts),
    }


def normalize_score(total_reward: float, num_sites: int) -> float:
    """
    Normalize cumulative reward to [0.0, 1.0] for OpenEnv grading.

    FIX: Bounds updated to reflect v4 reward design:
      - Classification range:  [-1.0, +1.7]
      - Step cost (2 steps):    -0.04 per site at minimum
      - Net max per site:       1.7 - 0.02 = ~1.68 (conservative: use 1.7)
      - Net min per site:      -1.0 - 0.04 = -1.04 (conservative: use -1.1)
    """
    if num_sites == 0:
        return 0.0
    min_possible = -1.1 * num_sites
    max_possible =  1.7 * num_sites
    raw = (total_reward - min_possible) / (max_possible - min_possible)
    return round(max(0.0, min(1.0, raw)), 4)


# =============================================================================
# Environment (v4) — Multi-step POMDP with Partial Observability
# =============================================================================

class PrivRLEnv:
    """
    OpenEnv-compatible multi-step RL environment for privacy risk classification.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  KEY DESIGN: PARTIAL OBSERVABILITY + INVESTIGATION PHASE           ║
    ║                                                                    ║
    ║  On reset(), agents see:  cookies, https, site_name               ║
    ║  HIDDEN on reset():       trackers (empty), privacy_policy ("")   ║
    ║                                                                    ║
    ║  To reveal hidden data, agents must use investigation actions:     ║
    ║    "inspect_trackers"  → reveals real tracker list   (+0.1 reward) ║
    ║    "inspect_policy"    → reveals privacy policy text (+0.1 reward) ║
    ║                                                                    ║
    ║  Then classify with:                                               ║
    ║    "mark_safe" / "mark_risky" / "mark_dangerous"                  ║
    ║                                                                    ║
    ║  Classification ends the current site and advances to the next.    ║
    ║  Max 5 steps per site — if exceeded, episode terminates.          ║
    ╚══════════════════════════════════════════════════════════════════════╝

    This creates a TRUE multi-step RL problem where:
    - State evolves across steps (hidden → revealed)
    - Agent must decide WHEN to stop investigating and classify
    - Investigating costs steps but improves decision quality
    - Classifying without inspection is risky but saves steps

    Usage:
        env = PrivRLEnv()
        obs = env.reset(task_id="easy", seed=42)

        # Investigation phase
        obs, r, done, info = env.step(PrivRLAction(classification="inspect_trackers"))
        obs, r, done, info = env.step(PrivRLAction(classification="inspect_policy"))

        # Decision phase
        obs, r, done, info = env.step(PrivRLAction(classification="mark_dangerous"))
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        """Initialize the environment with empty state."""
        self._state = PrivRLState()
        self._sites: List[dict] = []        # deep copies to allow noise injection
        self._current_idx: int = 0
        self._rng = random.Random()

        # ── Per-site investigation state ──
        self._trackers_visible: bool = False
        self._policy_visible: bool = False
        self._site_steps: int = 0           # steps taken on current site
        self.max_steps: int = MAX_STEPS_PER_SITE
        self.current_step: int = 0          # global step counter
        self._task_id: str = "easy"         # track for difficulty-aware step limits

    def _build_observation(self, site: dict, reward: Optional[float],
                           done: bool, message: str) -> PrivRLObservation:
        """
        Build an observation with partial observability applied.

        Trackers and privacy_policy are hidden until the agent
        uses the corresponding inspect action.
        Privacy policy is truncated to MAX_POLICY_LENGTH for response size.
        """
        # Determine visible policy text (truncated for response size)
        if self._policy_visible:
            full_policy = site["privacy_policy"]
            if len(full_policy) > MAX_POLICY_LENGTH:
                policy_text = full_policy[:MAX_POLICY_LENGTH] + "...[TRUNCATED]"
            else:
                policy_text = full_policy
        else:
            policy_text = "[HIDDEN] Use 'inspect_policy' to reveal."

        return PrivRLObservation(
            cookies=int(site["cookies"]),
            trackers=list(site["trackers"]) if self._trackers_visible else [],
            https=bool(site["https"]),
            privacy_policy=policy_text,
            task_id=str(self._state.task_id),
            # FIX 3: Anonymize site_name to prevent neural net memorization of proper nouns.
            # Agent should learn from signals (cookies, trackers, policy), not site identity.
            site_name=f"site_{self._current_idx:02d}",
            done=bool(done),
            reward=float(reward) if reward is not None else None,
            message=str(message),
        )

    def _reset_site_state(self):
        """Reset per-site investigation state for a new website."""
        self._trackers_visible = False
        self._policy_visible = False
        self._site_steps = 0

    # ─────────────────────────────────────────────────────────────────────────
    # OpenEnv API: reset()
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy", seed: Optional[int] = None,
              episode_id: Optional[str] = None,
              procedural: bool = False, n_sites: Optional[int] = None,
              **kwargs) -> PrivRLObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id:    "easy", "medium", or "hard"
            seed:       Random seed (None = random). Reproducible in both modes.
            episode_id: Optional episode identifier.
            procedural: If True, generate sites procedurally each episode
                        (prevents memorization, recommended for research/training).
                        If False (default), use static curated dataset.
            n_sites:    Number of sites per episode when procedural=True.
                        Defaults to 12 (same as static dataset).

        Returns:
            PrivRLObservation with PARTIAL data (trackers/policy hidden)
        """
        if task_id not in ALL_TASK_IDS:
            raise ValueError(f"Invalid task_id '{task_id}'. Valid: {ALL_TASK_IDS}")

        # Seed the RNG
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        if procedural:
            # Procedural mode: generate fresh sites every episode
            # Different seed each time = no memorization possible
            n = n_sites if n_sites is not None else len(TASKS[task_id]["sites"])
            self._sites = [
                generate_site(difficulty=task_id, rng=self._rng, site_index=i)
                for i in range(n)
            ]
        else:
            # Static mode (default): curated sites + cookie noise
            self._sites = [dict(s) for s in TASKS[task_id]["sites"]]
            for s in self._sites:
                noise = self._rng.randint(-COOKIE_NOISE_RANGE, COOKIE_NOISE_RANGE)
                s["cookies"] = max(0, s["cookies"] + noise)
            self._rng.shuffle(self._sites)

        self._current_idx = 0
        self._task_id = task_id

        # Difficulty-aware step limit
        self.max_steps = MAX_STEPS_HARD if task_id == "hard" else MAX_STEPS_PER_SITE

        # Initialize state
        self._state = PrivRLState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_index=0,
            task_id=task_id,
            ground_truth=self._sites[0]["ground_truth"],
            total_reward=0.0,
            total_sites=len(self._sites),
            sites_done=0,
        )

        # Reset investigation state
        self._reset_site_state()
        self.current_step = 0

        site = self._sites[0]
        return self._build_observation(
            site=site,
            reward=None,
            done=False,
            message=(
                f"Episode started. Classify {len(self._sites)} websites. "
                f"Difficulty: {task_id}. Trackers and policy are HIDDEN — "
                f"use 'inspect_trackers' / 'inspect_policy' to reveal, "
                f"then classify with 'mark_safe' / 'mark_risky' / 'mark_dangerous'. "
                f"Max {self.max_steps} steps per site."
            ),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # OpenEnv API: step()
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action: PrivRLAction, **kwargs) -> tuple:
        """
        Execute one step in the multi-step environment.

        Action types:
            INVESTIGATION (does NOT end the site):
                "inspect_trackers" — reveals tracker list, costs INSPECT_STEP_COST
                "inspect_policy"   — reveals privacy policy text, costs INSPECT_STEP_COST

            CLASSIFICATION (ends the current site, advances to next):
                "mark_safe" / "mark_risky" / "mark_dangerous"
                Reward = shaped multi-component score + informed bonus

        Returns:
            tuple of (observation, reward, done, info)
        """
        classification = action.classification.strip().lower()

        # ── Validate action ──
        if classification not in VALID_ACTIONS:
            obs = PrivRLObservation(
                cookies=0, trackers=[], https=False, privacy_policy="",
                task_id=self._state.task_id, site_name="ERROR",
                done=True, reward=-1.0,
                message=f"Invalid action '{classification}'. Must be one of: {sorted(VALID_ACTIONS)}",
            )
            return obs, -1.0, True, {
                "error": f"Invalid action: {classification}",
                "step_number": int(self._state.step_count),
                "valid_actions": sorted(VALID_ACTIONS),
            }

        current_site = self._sites[self._current_idx]
        self._site_steps += 1
        self.current_step += 1
        self._state.step_count += 1

        # ╔════════════════════════════════════════════╗
        # ║  INVESTIGATION ACTIONS (reveal hidden data) ║
        # ╚════════════════════════════════════════════╝

        if classification in INVESTIGATION_ACTIONS:
            # FIX 1: Inspection now costs INSPECT_STEP_COST (negative) instead of
            # granting free positive reward. Agent must weigh information value
            # against the step cost to decide when inspection is worth it.
            msg_parts = []
            new_reveal = False

            if classification == "inspect_trackers":
                if not self._trackers_visible:
                    self._trackers_visible = True
                    new_reveal = True
                    tracker_count = len(current_site["trackers"])
                    msg_parts.append(
                        f"Trackers revealed: {tracker_count} found. "
                        f"Step cost: {INSPECT_STEP_COST}"
                    )
                else:
                    msg_parts.append("Trackers already visible. No new information.")

            elif classification == "inspect_policy":
                if not self._policy_visible:
                    self._policy_visible = True
                    new_reveal = True
                    msg_parts.append(
                        f"Privacy policy revealed. "
                        f"Step cost: {INSPECT_STEP_COST}"
                    )
                else:
                    msg_parts.append("Policy already visible. No new information.")

            # Step cost applies only when a new reveal happens (re-inspection is free/zero)
            reward = float(INSPECT_STEP_COST) if new_reveal else 0.0
            self._state.total_reward += reward

            # Check step limit — force episode end
            if self._site_steps >= self.max_steps:
                self._state.total_reward += TIMEOUT_PENALTY
                score = normalize_score(self._state.total_reward, self._state.total_sites)
                obs = self._build_observation(
                    site=current_site, reward=TIMEOUT_PENALTY, done=True,
                    message=(
                        f"Step limit ({self.max_steps}) reached without classification! "
                        f"Penalty: {TIMEOUT_PENALTY}. Episode terminated. Score: {score:.4f}"
                    ),
                )
                return obs, float(TIMEOUT_PENALTY), True, {
                    "site_name": str(current_site["name"]),
                    "action_type": "timeout",
                    "reason": "step_limit_exceeded",
                    "step_number": int(self._state.step_count),
                    "site_steps": int(self._site_steps),
                    "max_steps": int(self.max_steps),
                    "final_score": float(score),
                    "total_reward": float(self._state.total_reward),
                }

            # Return updated observation (with revealed data)
            msg_parts.append(
                f"Steps on site: {self._site_steps}/{self.max_steps}."
            )
            obs = self._build_observation(
                site=current_site, reward=float(reward), done=False,
                message=" ".join(msg_parts),
            )
            info = {
                "site_name": current_site["name"],
                "action_type": "investigation",
                "step_number": int(self._state.step_count),
                "trackers_visible": bool(self._trackers_visible),
                "policy_visible": bool(self._policy_visible),
                "site_steps": int(self._site_steps),
                "max_steps": int(self.max_steps),
                "reasoning_hint": _generate_reasoning_hint(
                    cookies=current_site["cookies"],
                    trackers=current_site["trackers"],
                    https=current_site["https"],
                    trackers_visible=self._trackers_visible,
                    policy_visible=self._policy_visible,
                ),
            }
            return obs, float(reward), False, info

        # ╔════════════════════════════════════════════╗
        # ║  CLASSIFICATION ACTIONS (end current site)  ║
        # ╚════════════════════════════════════════════╝

        ground_truth = current_site["ground_truth"]
        difficulty   = current_site["difficulty"]

        reward_info = compute_reward(
            predicted=classification,
            ground_truth=ground_truth,
            trackers=current_site["trackers"],
            https=current_site["https"],
            privacy_policy=current_site["privacy_policy"],
            difficulty=difficulty,
            trackers_inspected=self._trackers_visible,
            policy_inspected=self._policy_visible,
        )
        reward = reward_info["total_reward"]

        self._state.total_reward += reward
        self._state.sites_done += 1
        self._current_idx += 1

        done = self._current_idx >= len(self._sites)

        # ── Build info dict (all values JSON-safe primitives) ──
        info = {
            "site_name": str(current_site["name"]),
            "action_type": "classification",
            "step_number": int(self._state.step_count),
            "ground_truth": str(ground_truth),
            "predicted": str(classification),
            "trackers_inspected": bool(self._trackers_visible),
            "policy_inspected": bool(self._policy_visible),
            "site_steps": int(self._site_steps),
            "max_steps": int(self.max_steps),
            "breakdown": str(reward_info["breakdown"]),
            "base_reward": float(reward_info["base_reward"]),
            "tracker_bonus": float(reward_info["tracker_bonus"]),
            "https_bonus": float(reward_info["https_bonus"]),
            "deception_bonus": float(reward_info["deception_bonus"]),
            "informed_decision_bonus": float(reward_info["informed_decision_bonus"]),
            "reasoning_hint": _generate_reasoning_hint(
                cookies=current_site["cookies"],
                trackers=current_site["trackers"],
                https=current_site["https"],
                trackers_visible=self._trackers_visible,
                policy_visible=self._policy_visible,
            ),
        }

        if done:
            score = normalize_score(self._state.total_reward, self._state.total_sites)
            info["final_score"] = score
            info["total_reward"] = self._state.total_reward
            obs = self._build_observation(
                site=current_site, reward=reward, done=True,
                message=(
                    f"Episode complete! {self._state.sites_done}/{self._state.total_sites} sites. "
                    f"Total reward: {self._state.total_reward:.2f}. Score: {score:.4f}. "
                    f"Last: '{current_site['name']}' predicted={classification} actual={ground_truth} "
                    f"[{reward_info['breakdown']}]"
                ),
            )
        else:
            # Advance to next site — reset investigation state
            next_site = self._sites[self._current_idx]
            self._state.current_index = self._current_idx
            self._state.ground_truth = next_site["ground_truth"]
            self._reset_site_state()  # hide trackers/policy for next site

            obs = self._build_observation(
                site=next_site, reward=reward, done=False,
                message=(
                    f"'{current_site['name']}': predicted={classification} actual={ground_truth} "
                    f"reward={reward:+.2f} [{reward_info['breakdown']}]. "
                    f"Progress: {self._state.sites_done}/{self._state.total_sites}. "
                    f"Next site loaded — trackers/policy hidden."
                ),
            )

        return obs, float(reward), bool(done), info

    # ─────────────────────────────────────────────────────────────────────────
    # OpenEnv API: state property
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> PrivRLState:
        """Return internal episode state (includes ground truth — not for agent)."""
        return self._state

    # ─────────────────────────────────────────────────────────────────────────
    # Grading & Metadata
    # ─────────────────────────────────────────────────────────────────────────

    def get_normalized_score(self) -> float:
        """Get the normalized score [0.0, 1.0] for the current episode."""
        return normalize_score(self._state.total_reward, self._state.total_sites)

    def get_task_info(self) -> dict:
        """Return metadata about the current task."""
        tid = self._state.task_id
        if tid in TASKS:
            return {"task_id": tid, "description": TASKS[tid]["description"],
                    "num_sites": len(TASKS[tid]["sites"])}
        return {"task_id": tid, "description": "Unknown", "num_sites": 0}

    @staticmethod
    def list_tasks() -> list:
        """Return list of all available task configurations."""
        return [
            {"task_id": tid, "description": TASKS[tid]["description"],
             "num_sites": len(TASKS[tid]["sites"])}
            for tid in ALL_TASK_IDS
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 5: Vectorized Observation (Gym / StableBaselines3 compatible)
    # ─────────────────────────────────────────────────────────────────────────

    def get_vector_obs(self) -> List[float]:
        """
        Return a 13-dimensional normalized float vector for the current site.

        Suitable for direct input to MLP-based RL agents (PPO, A2C, DQN).
        Use this instead of the text-based PrivRLObservation for numeric training.

        Dimensions:
          [0]   cookies (normalized 0–1, capped at 60)
          [1]   https   (0.0 or 1.0)
          [2]   trackers_visible (0.0 or 1.0)
          [3]   policy_visible   (0.0 or 1.0)
          [4]   site_steps / max_steps  (step budget consumed)
          [5]   deception_score  (only if policy visible, else 0.0)
          [6]   tracker_category_ads          (count, if visible)
          [7]   tracker_category_analytics    (count, if visible)
          [8]   tracker_category_social       (count, if visible)
          [9]   tracker_category_attribution  (count, if visible)
          [10]  tracker_category_malicious    (count, if visible)
          [11]  tracker_category_iot          (count, if visible)
          [12]  tracker_category_opaque       (count, if visible)
        """
        if not self._sites:
            return [0.0] * 13

        site = self._sites[self._current_idx]

        # Tracker category counts (only if trackers have been revealed)
        cat_counts = [0.0] * 7
        if self._trackers_visible:
            for t in site["trackers"]:
                cat = classify_tracker(t)
                idx = CATEGORY_INDEX.get(cat, 6)
                cat_counts[idx] += 1.0

        # Deception score (only if policy has been revealed)
        deception = float(
            detect_policy_deception(site["privacy_policy"])
        ) if self._policy_visible else 0.0

        return [
            float(min(site["cookies"], 60)) / 60.0,     # [0] normalized cookies
            float(site["https"]),                        # [1] https flag
            float(self._trackers_visible),               # [2] trackers revealed?
            float(self._policy_visible),                 # [3] policy revealed?
            float(self._site_steps) / float(self.max_steps),  # [4] step budget
            deception,                                   # [5] policy deception score
            *cat_counts,                                 # [6–12] tracker categories
        ]

