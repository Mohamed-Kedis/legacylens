"""
Automatic Policy Suggestions: rule-based, explainable, and the user always confirms.

Output maps directly onto the Legacy Policy fields of the product:
    who        -> which trusted contacts can receive it
    when       -> trigger (legacy_activation | emergency | never)
    action     -> deliver | deliver_redacted | delete | destroy (cryptographic erasure)
    access     -> one_time | expires_after_days
    safeguards -> extra security required before release
"""
from __future__ import annotations

from typing import List

RULES = {
    "id_card": {"who": "executor", "when": "legacy_activation", "action": "deliver_redacted",
                "access": {"one_time": True, "expires_after_days": 7},
                "why": "Identity documents are needed for paperwork after death, but a full copy enables identity fraud"},
    "passport": {"who": "executor", "when": "legacy_activation", "action": "deliver_redacted",
                 "access": {"one_time": True, "expires_after_days": 7},
                 "why": "Same as ID: share a redacted copy only with the person handling official procedures"},
    "financial": {"who": "designated_heirs", "when": "legacy_activation", "action": "deliver",
                  "access": {"one_time": False, "expires_after_days": 30},
                  "why": "Heirs need account details to claim assets; limit the access window"},
    "contract": {"who": "executor_and_lawyer", "when": "legacy_activation", "action": "deliver",
                 "access": {"one_time": False, "expires_after_days": 90},
                 "why": "Contracts have legal effect after death and may be needed for a longer period"},
    "photo": {"who": "family", "when": "legacy_activation", "action": "deliver",
              "access": {"one_time": False, "expires_after_days": None},
              "why": "Personal memories are usually meant to be kept by family"},
    "other": {"who": "user_choice", "when": "legacy_activation", "action": "delete",
              "access": {"one_time": True, "expires_after_days": 7},
              "why": "No clear owner after death: default to privacy (delete) unless the user decides otherwise"},
}


def suggest_policy(doc_type: str, risk: dict, pii: List[dict], faces: int) -> dict:
    s = {k: (v.copy() if isinstance(v, dict) else v) for k, v in RULES.get(doc_type, RULES["other"]).items()}
    safeguards: List[str] = []
    if risk["level"] in ("high", "critical"):
        safeguards += ["mfa_before_release", "two_of_three_contacts_approval"]
    if risk["level"] == "critical":
        safeguards.append("cooling_period_48h_on_policy_change")
    if any(p["entity"] in ("CREDIT_CARD",) for p in pii):
        s["action"] = "destroy"
        s["why"] = "Card numbers are useless to heirs and risky if leaked: destroy (cryptographic erasure)"
    if doc_type == "photo" and faces == 0 and risk["score"] < 10:
        safeguards = []
    s["safeguards"] = safeguards
    s["requires_user_confirmation"] = True
    return s
