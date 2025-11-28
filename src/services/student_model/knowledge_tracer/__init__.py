"""
Knowledge Tracing Module
Interface abstrata para diferentes algoritmos de knowledge tracing (BKT, AKT, etc.)
"""

from .base_tracer import KnowledgeTracer
from .bkt_tracer import BayesianKnowledgeTracer
from .akt_tracer import AttentiveKnowledgeTracer

# AKT é o padrão, BKT é fallback
__all__ = ["KnowledgeTracer", "AttentiveKnowledgeTracer", "BayesianKnowledgeTracer"]

