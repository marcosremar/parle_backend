"""
Pydantic models for Linguistic Analysis Service
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ParseRequest(BaseModel):
    """Request model for text parsing"""
    text: str = Field(..., description="Text to parse")


class TokenData(BaseModel):
    """Token information"""
    text: str
    lemma: str
    pos: str
    tag: str
    dep: str
    head: Optional[str] = None
    head_pos: Optional[str] = None
    children: List[str] = []
    is_punct: bool = False
    is_space: bool = False
    is_stop: bool = False
    is_alpha: bool = True


class SentenceData(BaseModel):
    """Sentence information"""
    text: str
    start: int
    end: int
    tokens: List[TokenData]


class DependencyRelation(BaseModel):
    """Dependency relation"""
    token: str
    head: str
    relation: str
    pos: str
    head_pos: Optional[str] = None


class ParseResponse(BaseModel):
    """Response model for parsing"""
    text: str
    sentences: List[SentenceData]
    dependency_relations: List[DependencyRelation]
    pos_tags: List[Dict[str, str]]
    num_sentences: int
    num_tokens: int
    num_words: int


class YngveDepthResponse(BaseModel):
    """Response for Yngve depth calculation"""
    mean_yngve_depth: float
    max_yngve_depth: float
    sentence_yngve_depths: List[float]
    total_sentences: int


class FrazierDepthResponse(BaseModel):
    """Response for Frazier depth calculation"""
    mean_frazier_depth: float
    max_frazier_depth: float
    sentence_frazier_depths: List[float]
    total_sentences: int


class TUnitData(BaseModel):
    """T-unit information"""
    text: str
    main_clause: List[Dict[str, Any]]
    subordinate_clauses: List[List[Dict[str, Any]]]
    num_clauses: int
    num_words: int


class TUnitsResponse(BaseModel):
    """Response for T-units extraction"""
    t_units: List[TUnitData]
    num_t_units: int
    avg_words_per_tunit: float
    avg_clauses_per_tunit: float
    total_words: int
    total_clauses: int


class SubordinationIndexResponse(BaseModel):
    """Response for subordination index"""
    subordination_index: float
    subordinate_clauses_per_tunit: float
    num_t_units: int
    total_subordinate_clauses: int


class SyntacticMetricsResponse(BaseModel):
    """Combined syntactic metrics response"""
    yngve_depth: YngveDepthResponse
    frazier_depth: FrazierDepthResponse
    t_units: TUnitsResponse
    subordination_index: SubordinationIndexResponse

