"""Attack catalog modules."""
from .basic_injections import BASIC_INJECTION_ATTACKS
from .social_engineering import SOCIAL_ENGINEERING_ATTACKS
from .stealth_exfiltration import STEALTH_EXFILTRATION_ATTACKS

__all__ = [
    "BASIC_INJECTION_ATTACKS",
    "SOCIAL_ENGINEERING_ATTACKS",
    "STEALTH_EXFILTRATION_ATTACKS",
]
