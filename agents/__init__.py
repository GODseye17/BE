"""
Multi-Agent System for Medical Literature Analysis
"""
from .research_agent import ResearchAgent
from .clinical_agent import ClinicalAgent
from .statistical_agent import StatisticalAgent
from .critic_agent import CriticAgent
from .coordinator import MultiAgentCoordinator

__all__ = [
    'ResearchAgent',
    'ClinicalAgent', 
    'StatisticalAgent',
    'CriticAgent',
    'MultiAgentCoordinator'
]
