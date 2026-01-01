"""
Prompt modules for LLM-1 (Router & Clarifier) and LLM-2 (Executor)
"""

from .llm1_prompt import OPTIMIZED_ROUTER_SYSTEM_PROMPT
from .llm2_prompt import llm2_prompt_builder, BASE_LLM2_SYSTEM_PROMPT

__all__ = [
    'OPTIMIZED_ROUTER_SYSTEM_PROMPT',
    'llm2_prompt_builder',
    'BASE_LLM2_SYSTEM_PROMPT',
]
