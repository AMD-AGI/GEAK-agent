"""
Intervention Policy for mini-kernel Agent

Defines when the agent should pause and ask the user for input.

Modes:
- cautious: Pause on any regression, major changes
- balanced: Pause on repeated failures, significant regression
- minimal: Only pause on critical errors, cost limits
"""

from dataclasses import dataclass, field
from typing import List, Set
from enum import Enum


class InterventionMode(Enum):
    """Intervention sensitivity modes."""
    CAUTIOUS = "cautious"    # New to kernel optimization
    BALANCED = "balanced"    # Experienced (default)
    MINIMAL = "minimal"      # Expert, tight deadline


@dataclass
class InterventionPolicy:
    """
    When should the agent pause and ask the user?
    
    This policy controls the agent's autonomy level.
    """
    
    # === ALWAYS PAUSE (cannot disable) ===
    cost_limit_exceeded: bool = True
    about_to_modify_protected_files: bool = True
    agent_explicitly_uncertain: bool = True  # Agent says "I'm not sure"
    
    # === CONFIGURABLE THRESHOLDS ===
    consecutive_failures: int = 3           # Same error category
    performance_regression_pct: float = 5.0  # vs best checkpoint
    time_without_progress_mins: float = 10.0 # Stuck detection
    
    # === SMART HEURISTICS ===
    detect_major_refactor: bool = True      # >50% of file changed
    major_refactor_threshold: float = 0.5   # 50% change threshold
    detect_risky_patterns: bool = True      # e.g., removing safety checks
    
    # === MODE-SPECIFIC DEFAULTS ===
    mode: InterventionMode = InterventionMode.BALANCED
    
    # === PROTECTED FILES ===
    protected_files: Set[str] = field(default_factory=set)
    
    # === COST LIMITS ===
    max_cost_usd: float = 10.0
    
    @classmethod
    def from_mode(cls, mode: str) -> 'InterventionPolicy':
        """Create policy from mode string."""
        mode_enum = InterventionMode(mode)
        
        if mode_enum == InterventionMode.CAUTIOUS:
            return cls(
                consecutive_failures=2,
                performance_regression_pct=2.0,
                time_without_progress_mins=5.0,
                detect_major_refactor=True,
                detect_risky_patterns=True,
                mode=mode_enum,
            )
        elif mode_enum == InterventionMode.MINIMAL:
            return cls(
                consecutive_failures=5,
                performance_regression_pct=15.0,
                time_without_progress_mins=30.0,
                detect_major_refactor=False,
                detect_risky_patterns=False,
                mode=mode_enum,
            )
        else:  # BALANCED (default)
            return cls(
                consecutive_failures=3,
                performance_regression_pct=5.0,
                time_without_progress_mins=10.0,
                detect_major_refactor=True,
                detect_risky_patterns=True,
                mode=mode_enum,
            )
    
    def should_pause_on_failure(self, consecutive_count: int) -> bool:
        """Check if should pause based on failure count."""
        return consecutive_count >= self.consecutive_failures
    
    def should_pause_on_regression(self, regression_pct: float) -> bool:
        """Check if should pause based on performance regression."""
        return regression_pct > self.performance_regression_pct
    
    def should_pause_on_major_change(self, change_pct: float) -> bool:
        """Check if should pause based on file change size."""
        if not self.detect_major_refactor:
            return False
        return change_pct > self.major_refactor_threshold
    
    def is_protected_file(self, filepath: str) -> bool:
        """Check if file is protected."""
        for protected in self.protected_files:
            if protected in filepath:
                return True
        return False
    
    def check_cost_limit(self, current_cost: float) -> bool:
        """Check if cost limit exceeded."""
        return current_cost > self.max_cost_usd


# Interrupt keywords that users can type at any time
INTERRUPT_KEYWORDS = {
    "!stop": "Immediately halt current action, ask for input",
    "!pause": "Finish current step, then pause",
    "!status": "Show current state without interrupting",
    "!rollback": "Revert to last checkpoint",
    "!best": "Rollback to best checkpoint",
    "!help": "Show available commands",
    "!quit": "Quit and save current state",
}


def parse_interrupt(user_input: str) -> tuple:
    """
    Parse user input for interrupt keywords.
    
    Returns (keyword, args) or (None, None) if not an interrupt.
    """
    user_input = user_input.strip()
    
    for keyword in INTERRUPT_KEYWORDS:
        if user_input.startswith(keyword):
            args = user_input[len(keyword):].strip()
            return keyword, args
    
    return None, None


