"""
Linguistic Analysis Service
Provides dependency parsing, syntactic metrics, and linguistic features for CEFR classification
"""

from .parser import DependencyParser
from .syntactic_metrics import SyntacticMetricsCalculator

__all__ = ["DependencyParser", "SyntacticMetricsCalculator"]

