#!/usr/bin/env python3
"""Validate ground_truth strings across all attack catalogs.

This script ensures that:
1. Every ground_truth value in attack catalogs is a valid ThreatCategory enum value (or "safe")
2. The grader's SUPERCLASS_MAP covers all attack categories used in catalogs
3. No ground_truth collides with superclass names in unexpected ways

Exit codes:
    0 - All checks passed
    1 - One or more validation failures (errors printed to stderr)
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import ThreatCategory
from server.attacks import (
    BASIC_INJECTION_ATTACKS,
    SOCIAL_ENGINEERING_ATTACKS,
    STEALTH_EXFILTRATION_ATTACKS,
)
from server.grader import SUPERCLASS_MAP


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    # ── 1. Build the set of valid ground_truth values ──────────────────────
    valid_ground_truths: set[str] = {"safe"} | {c.value for c in ThreatCategory}

    # ── 2. Collect all attack catalogs ─────────────────────────────────────
    catalogs = {
        "basic_injections": BASIC_INJECTION_ATTACKS,
        "social_engineering": SOCIAL_ENGINEERING_ATTACKS,
        "stealth_exfiltration": STEALTH_EXFILTRATION_ATTACKS,
    }

    # Track which ground_truth values are actually used
    used_ground_truths: set[str] = set()

    for catalog_name, attacks in catalogs.items():
        for idx, (attack_text, ground_truth, attack_type) in enumerate(attacks):
            used_ground_truths.add(ground_truth)
            if ground_truth not in valid_ground_truths:
                errors.append(
                    f"[{catalog_name}] Attack #{idx} (type='{attack_type}'): "
                    f"invalid ground_truth='{ground_truth}' — "
                    f"not a valid ThreatCategory value"
                )

    # ── 3. Check SUPERCLASS_MAP coverage ───────────────────────────────────
    # All non-safe ground_truth values used in catalogs must appear in SUPERCLASS_MAP
    all_superclass_categories: set[str] = set()
    for sc_name, categories in SUPERCLASS_MAP.items():
        for cat in categories:
            all_superclass_categories.add(cat.value)

    for gt in used_ground_truths:
        if gt == "safe":
            continue  # safe is not expected in SUPERCLASS_MAP
        if gt not in all_superclass_categories:
            errors.append(
                f"SUPERCLASS_MAP coverage: ground_truth='{gt}' is used in attack catalogs "
                f"but not covered by any superclass in grader.SUPERCLASS_MAP"
            )

    # ── 4. Check for unexpected collisions ─────────────────────────────────
    superclass_names = set(SUPERCLASS_MAP.keys())

    # Build reverse map: category value -> which superclass it belongs to
    category_to_superclass: dict[str, str] = {}
    for sc_name, categories in SUPERCLASS_MAP.items():
        for cat in categories:
            category_to_superclass[cat.value] = sc_name

    for gt in used_ground_truths:
        if gt in superclass_names:
            # Collision is OK if the ground_truth is a member of its own superclass
            # (e.g., "injection" is both a ThreatCategory and a superclass name,
            #  and ThreatCategory.INJECTION is in the "injection" superclass set).
            # But flag it as a warning for awareness.
            own_sc = category_to_superclass.get(gt)
            if own_sc == gt:
                warnings.append(
                    f"Name collision (benign): ground_truth='{gt}' is also a superclass name, "
                    f"and ThreatCategory.{gt.upper()} belongs to the '{gt}' superclass. "
                    f"This is intentional but worth noting."
                )
            else:
                errors.append(
                    f"Name collision (unexpected): ground_truth='{gt}' is also a superclass name "
                    f"in SUPERCLASS_MAP, but does NOT belong to the '{gt}' superclass "
                    f"(belongs to '{own_sc}' instead). This may cause confusion."
                )

    # ── 5. Report results ──────────────────────────────────────────────────
    if errors:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"VALIDATION FAILED — {len(errors)} error(s), {len(warnings)} warning(s)", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)
        for i, err in enumerate(errors, 1):
            print(f"  ERROR {i}: {err}", file=sys.stderr)
        if warnings:
            print(file=sys.stderr)
            for i, warn in enumerate(warnings, 1):
                print(f"  WARNING {i}: {warn}", file=sys.stderr)
        print(file=sys.stderr)
        return 1

    # Success — print summary
    print("✓ All ground_truth values are valid ThreatCategory entries")
    print(f"  Valid values: {sorted(valid_ground_truths)}")
    print(f"  Used in catalogs: {sorted(used_ground_truths)}")
    print(f"  SUPERCLASS_MAP covers: {sorted(all_superclass_categories)}")
    print(f"  Superclass names: {sorted(superclass_names)}")
    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s) (non-blocking):")
        for w in warnings:
            print(f"    - {w}")
    print()
    print("✓ All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
