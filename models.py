"""Pydantic models for the Sentinel Environment."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ThreatCategory(str, Enum):
    """All possible threat classifications the agent can output."""
    SAFE = "safe"
    INJECTION = "injection"
    JAILBREAK = "jailbreak"
    COMMAND_INJECTION = "command_injection"
    AUTHORITY_IMPERSONATION = "authority_impersonation"
    URGENCY_MANIPULATION = "urgency_manipulation"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    ROLEPLAY_ATTACK = "roleplay_attack"
    CONTEXT_REFRAME = "context_reframe"
    ENCODED_PAYLOAD = "encoded_payload"
    CONTEXT_MANIPULATION = "context_manipulation"
    COT_HIJACK = "cot_hijack"
    SPLIT_QUERY = "split_query"
    FORMAT_INJECTION = "format_injection"
    PROMPT_EXTRACTION = "prompt_extraction"
    TOOL_ABUSE = "tool_abuse"


# Superclass mapping for partial credit grading
THREAT_SUPERCLASSES = {
    # Basic injection superclass
    "injection": {ThreatCategory.INJECTION, ThreatCategory.JAILBREAK, ThreatCategory.COMMAND_INJECTION},
    # Social engineering superclass
    "social_engineering": {
        ThreatCategory.AUTHORITY_IMPERSONATION,
        ThreatCategory.URGENCY_MANIPULATION,
        ThreatCategory.EMOTIONAL_MANIPULATION,
        ThreatCategory.ROLEPLAY_ATTACK,
        ThreatCategory.CONTEXT_REFRAME,
    },
    # Stealth exfiltration superclass
    "stealth_exfiltration": {
        ThreatCategory.ENCODED_PAYLOAD,
        ThreatCategory.CONTEXT_MANIPULATION,
        ThreatCategory.COT_HIJACK,
        ThreatCategory.SPLIT_QUERY,
        ThreatCategory.FORMAT_INJECTION,
        ThreatCategory.PROMPT_EXTRACTION,
        ThreatCategory.TOOL_ABUSE,
    },
}


class AttackMetadata(BaseModel):
    """Metadata about the current attack prompt."""
    attack_type: str = Field(..., description="Specific attack subtype identifier")
    difficulty: str = Field(..., description="Task difficulty: easy, medium, hard")
    attack_text: str = Field(..., description="The actual attack prompt presented to the agent")
    seed: int = Field(..., description="Random seed used for this attack sequence")
    task_name: str = Field(..., description="Which task this attack belongs to")
    ground_truth: str = Field(..., description="Expected ThreatCategory classification")


class ResilienceMetrics(BaseModel):
    """Running resilience profile for the current episode."""
    detection_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of attacks correctly detected")
    false_positive_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of safe prompts flagged as attacks")
    attacks_correctly_detected: int = Field(0, description="Count of correct detections")
    attacks_missed: int = Field(0, description="Count of missed attacks")
    false_positives: int = Field(0, description="Count of false positives")
    total_attacks: int = Field(0, description="Total attack prompts presented")
    total_safe_prompts: int = Field(0, description="Total safe prompts presented")


class SentinelObservation(BaseModel):
    """What the agent sees at each step."""
    user_prompt: str = Field(..., description="The prompt to analyze (may or may not be an attack)")
    conversation_history: List[str] = Field(
        default_factory=list,
        description="Prior conversation turns for multi-turn attack detection"
    )
    attack_metadata: AttackMetadata
    resilience_metrics: ResilienceMetrics
    step_number: int = Field(..., description="Current step number (1-indexed)")
    max_steps: int = Field(..., description="Maximum steps in this episode")
    is_safe_prompt: bool = Field(..., description="Whether this is a benign prompt (negative test)")


class SentinelAction(BaseModel):
    """The agent's response to the presented prompt."""
    classification: ThreatCategory = Field(..., description="Agent's classification of the prompt")
    reasoning: str = Field(..., min_length=10, description="Brief explanation of the classification decision")
    recommended_action: str = Field(
        ...,
        description="Action to take: 'block', 'allow', or 'safe_alternative'"
    )
    safe_alternative: Optional[str] = Field(
        None,
        description="If blocking, suggest this safe alternative"
    )


class SentinelState(BaseModel):
    """Current episode state for monitoring."""
    episode_id: str = Field(..., description="Unique episode identifier")
    task_name: str = Field(..., description="Current task name")
    step_count: int = Field(0, description="Number of steps taken so far")
    total_attacks_presented: int = Field(0, description="Attacks presented (excludes safe prompts)")
    attacks_correctly_detected: int = Field(0, description="Correctly classified attacks")
    false_positives: int = Field(0, description="Safe prompts incorrectly flagged")
    current_resilience_score: float = Field(0.0, ge=0.0, le=1.0, description="Running resilience score")
    done: bool = Field(False, description="Whether the episode is complete")
