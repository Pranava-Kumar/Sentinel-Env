"""Load real-world jailbreak prompts from the jailbreak-prompts directory.

Parses structured jailbreak text files and converts them into attack dictionaries
compatible with the attack engine. Supports multiple formats:
- JSON-structured prompts (MISC.txt, etc.)
- Raw text prompts (1337.txt, AAA.txt, etc.)
- Multi-prompt files (MOTHERLOAD.txt, OPENAI.txt, SYSPROMPTS.txt)
- Directory-based frameworks (G0DM0D3/, L1B3RT4S/)

These real-world jailbreaks significantly expand the attack surface beyond
synthetic attacks, providing production-grade adversarial testing.
"""

import json
import re
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

# Path to jailbreak prompts directory (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
JAILBREAK_DIR = PROJECT_ROOT / "jailbreak-prompts"

# Attack type mapping: file/pattern -> attack classification
ATTACK_TYPE_MAP = {
    "1337": "encoded_payload",  # Leetspeak encoding
    "AAA": "jailbreak",  # Generic jailbreak
    "MOTHERLOAD": "jailbreak",  # Composite jailbreak
    "OPENAI": "system_override",  # System prompt attacks
    "SYSPROMPTS": "system_override",  # System prompt extraction
    "MISC": "context_reframe",  # Various reframing attacks
    "G0DM0D3": "system_override",  # God mode bypass
    "L1B3RT4S": "context_manipulation",  # Libertas framework
    "SHORTCUTS": "split_query",  # Distributed attacks
}

# Difficulty assignment by source
DIFFICULTY_MAP = {
    "1337": "medium",  # Encoding requires detection
    "AAA": "easy",
    "MOTHERLOAD": "hard",  # Complex composite attacks
    "OPENAI": "hard",  # Sophisticated system overrides
    "SYSPROMPTS": "hard",
    "MISC": "medium",
    "G0DM0D3": "hard",  # Advanced framework
    "L1B3RT4S": "hard",  # Structured manipulation
    "SHORTCUTS": "medium",
}


def _classify_jailbreak(text: str, source: str) -> str:
    """Classify a jailbreak prompt into a threat category.

    Uses keyword heuristics to determine the most appropriate threat category.
    """
    text_lower = text.lower()

    # Check for encoding attacks
    if any(term in text_lower for term in ["1337", "l33t", "leetspeak", "base64", "hex", "encoded"]):
        return "encoded_payload"

    # Check for system override attempts
    if any(
        term in text_lower for term in ["system", "override", "ignore previous", "developer mode", "g0d", "godmode"]
    ):
        return "system_override"

    # Check for context manipulation
    if any(term in text_lower for term in ["research", "academic", "educational", "hypothetical", "fiction", "story"]):
        return "context_manipulation"

    # Check for authority/roleplay
    if any(term in text_lower for term in ["acting as", "role", "simulate", "pretend", "you are now"]):
        return "roleplay_attack"

    # Check for emotional manipulation
    if any(term in text_lower for term in ["please", "help", "urgent", "emergency", "important"]):
        return "emotional_manipulation"

    # Check for urgency
    if any(term in text_lower for term in ["immediately", "urgent", "asap", "right now", "without asking"]):
        return "urgency_manipulation"

    # Default to jailbreak category
    return "jailbreak"


def _parse_json_prompts(text: str, source: str) -> list[dict[str, Any]]:
    """Parse JSON-structured prompts from files like MISC.txt."""
    attacks = []

    # Find JSON blocks
    json_pattern = re.compile(r'\{[^{}]*(?:"prompt"[^{}]*\})', re.DOTALL)
    matches = json_pattern.findall(text)

    for match in matches:
        try:
            data = json.loads(match)
            if "prompt" in data:
                attack_type = ATTACK_TYPE_MAP.get(source, "jailbreak")
                if "attack_type" in data:
                    attack_type = data["attack_type"]
                else:
                    attack_type = _classify_jailbreak(data["prompt"], source)

                attacks.append(
                    {
                        "text": data["prompt"],
                        "ground_truth": attack_type,
                        "is_attack": True,
                        "attack_type": attack_type,
                        "difficulty": DIFFICULTY_MAP.get(source, "medium"),
                        "source": f"jailbreak:{source}",
                    }
                )
        except json.JSONDecodeError:
            continue

    return attacks


def _parse_raw_prompts(text: str, source: str) -> list[dict[str, Any]]:
    """Parse raw text prompts (non-JSON format)."""
    attacks = []

    # Split on common delimiters
    sections = re.split(r"\n{3,}|#{3,}|[-=*]{5,}", text)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 20:
            continue

        # Skip if it looks like a header only
        if section.startswith("#") and len(section) < 100:
            continue

        attack_type = _classify_jailbreak(section, source)
        attacks.append(
            {
                "text": section[:2000],  # Limit length
                "ground_truth": attack_type,
                "is_attack": True,
                "attack_type": attack_type,
                "difficulty": DIFFICULTY_MAP.get(source, "medium"),
                "source": f"jailbreak:{source}",
            }
        )

    return attacks


def _parse_shortcuts_json(text: str) -> list[dict[str, Any]]:
    """Parse SHORTCUTS.json file which contains structured attack variants."""
    attacks = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "text" in item:
                    attack_type = item.get("attack_type", "split_query")
                    attacks.append(
                        {
                            "text": item["text"],
                            "ground_truth": attack_type,
                            "is_attack": True,
                            "attack_type": attack_type,
                            "difficulty": item.get("difficulty", "medium"),
                            "source": "jailbreak:shortcuts",
                        }
                    )
        elif isinstance(data, dict):
            # Single shortcut
            attack_type = data.get("attack_type", "split_query")
            attacks.append(
                {
                    "text": data.get("text", ""),
                    "ground_truth": attack_type,
                    "is_attack": True,
                    "attack_type": attack_type,
                    "difficulty": data.get("difficulty", "medium"),
                    "source": "jailbreak:shortcuts",
                }
            )
    except json.JSONDecodeError:
        pass

    return attacks


def _load_directory(dir_path: Path) -> list[dict[str, Any]]:
    """Load jailbreak prompts from a directory (e.g., G0DM0D3/)."""
    attacks = []
    source = dir_path.name.upper()

    for file_path in dir_path.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip non-text files
        if file_path.suffix not in (".txt", ".json", ".md", ".js", ".ts", ".py"):
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if not content.strip():
            continue

        if file_path.suffix == ".json":
            attacks.extend(_parse_raw_prompts(content, source))
        else:
            # For code/markdown files, extract relevant sections
            attacks.extend(_parse_raw_prompts(content, source))

    return attacks


def load_jailbreak_prompts() -> list[dict[str, Any]]:
    """Load all jailbreak prompts from the jailbreak-prompts directory.

    Returns:
        List of attack dictionaries compatible with the attack engine.
    """
    if not JAILBREAK_DIR.exists():
        # Silently return empty list - directory not required for basic operation
        return []

    # Check if directory is empty (has only .gitkeep)
    files = [f for f in JAILBREAK_DIR.iterdir() if f.is_file() and f.name != ".gitkeep"]
    if not files:
        # Directory exists but is empty - that's OK
        return []

    all_attacks = []

    # Load individual text files
    for file_path in JAILBREAK_DIR.iterdir():
        if not file_path.is_file():
            continue

        source = file_path.stem.upper()

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning("Failed to read jailbreak file", file=str(file_path), error=str(e))
            continue

        if not content.strip():
            continue

        # Parse based on file type
        if file_path.name == "SHORTCUTS.json":
            attacks = _parse_shortcuts_json(content)
        elif file_path.suffix == ".json":
            attacks = _parse_json_prompts(content, source)
        else:
            attacks = _parse_raw_prompts(content, source)

        all_attacks.extend(attacks)

    # Load directory-based frameworks (G0DM0D3/, L1B3RT4S/)
    for dir_path in JAILBREAK_DIR.iterdir():
        if dir_path.is_dir():
            all_attacks.extend(_load_directory(dir_path))

    logger.info("Loaded jailbreak prompts", count=len(all_attacks))
    return all_attacks


def get_jailbreak_by_type(attack_type: str) -> list[dict[str, Any]]:
    """Get jailbreak prompts filtered by attack type.

    Args:
        attack_type: The attack type to filter by (e.g., "jailbreak", "system_override")

    Returns:
        List of attack dictionaries matching the specified type.
    """
    all_prompts = load_jailbreak_prompts()
    return [p for p in all_prompts if p.get("attack_type") == attack_type]
