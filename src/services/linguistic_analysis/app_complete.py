"""
Linguistic Analysis Service - Standalone FastAPI application
Provides dependency parsing, syntactic metrics (Yngve, Frazier, T-units) for CEFR classification
"""

import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import local modules
from .parser import DependencyParser
from .syntactic_metrics import SyntacticMetricsCalculator
from .models import (
    ParseRequest, ParseResponse,
    YngveDepthResponse, FrazierDepthResponse,
    TUnitsResponse, SubordinationIndexResponse,
    SyntacticMetricsResponse
)

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "linguistic_analysis",
        "port": 8901,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    }
}

def get_config():
    """Get service configuration"""
    config = DEFAULT_CONFIG.copy()
    port = int(os.getenv("LINGUISTIC_ANALYSIS_PORT", os.getenv("PORT", "8901")))
    config["service"]["port"] = port
    return config

# ============================================================================
# Service Initialization
# ============================================================================

# Initialize parser and metrics calculator
parser = DependencyParser(model_name="pt_core_news_lg")
metrics_calculator = SyntacticMetricsCalculator(parser)

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Linguistic Analysis Service",
    version="1.0.0",
    description="Service for dependency parsing and syntactic complexity metrics"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "linguistic_analysis",
        "model": parser.model_name,
        "model_loaded": parser._loaded
    }

# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/parse", response_model=ParseResponse)
async def parse_text(request: ParseRequest):
    """
    Parse text and extract linguistic features
    
    Returns dependency relations, POS tags, and sentence structure
    """
    try:
        result = await parser.parse_text(request.text)
        return ParseResponse(**result)
    except Exception as e:
        logger.error(f"Error parsing text: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Parsing failed: {str(e)}"
        )

@app.post("/api/yngve-depth", response_model=YngveDepthResponse)
async def calculate_yngve_depth(request: ParseRequest):
    """
    Calculate Yngve depth (left-branching syntactic complexity)
    """
    try:
        result = await metrics_calculator.calculate_yngve_depth(request.text)
        return YngveDepthResponse(**result)
    except Exception as e:
        logger.error(f"Error calculating Yngve depth: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Yngve depth calculation failed: {str(e)}"
        )

@app.post("/api/frazier-depth", response_model=FrazierDepthResponse)
async def calculate_frazier_depth(request: ParseRequest):
    """
    Calculate Frazier depth (right-branching syntactic complexity)
    """
    try:
        result = await metrics_calculator.calculate_frazier_depth(request.text)
        return FrazierDepthResponse(**result)
    except Exception as e:
        logger.error(f"Error calculating Frazier depth: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Frazier depth calculation failed: {str(e)}"
        )

@app.post("/api/t-units", response_model=TUnitsResponse)
async def extract_t_units(request: ParseRequest):
    """
    Extract T-units (minimal terminable units)
    """
    try:
        result = await metrics_calculator.extract_t_units(request.text)
        # Convert TUnitData dicts to proper format
        t_units = []
        for tu in result["t_units"]:
            t_units.append({
                "text": tu["text"],
                "main_clause": tu["main_clause"],
                "subordinate_clauses": tu["subordinate_clauses"],
                "num_clauses": tu["num_clauses"],
                "num_words": tu["num_words"]
            })
        result["t_units"] = t_units
        return TUnitsResponse(**result)
    except Exception as e:
        logger.error(f"Error extracting T-units: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"T-units extraction failed: {str(e)}"
        )

@app.post("/api/subordination-index", response_model=SubordinationIndexResponse)
async def calculate_subordination_index(request: ParseRequest):
    """
    Calculate subordination index (subordinate clauses per T-unit)
    """
    try:
        result = await metrics_calculator.calculate_subordination_index(request.text)
        return SubordinationIndexResponse(**result)
    except Exception as e:
        logger.error(f"Error calculating subordination index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Subordination index calculation failed: {str(e)}"
        )

@app.post("/api/syntactic-metrics", response_model=SyntacticMetricsResponse)
async def get_all_syntactic_metrics(request: ParseRequest):
    """
    Get all syntactic metrics in one call (Yngve, Frazier, T-units, subordination)
    """
    try:
        yngve = await metrics_calculator.calculate_yngve_depth(request.text)
        frazier = await metrics_calculator.calculate_frazier_depth(request.text)
        t_units = await metrics_calculator.extract_t_units(request.text)
        subordination = await metrics_calculator.calculate_subordination_index(request.text)
        
        # Convert T-units format
        t_units_list = []
        for tu in t_units["t_units"]:
            t_units_list.append({
                "text": tu["text"],
                "main_clause": tu["main_clause"],
                "subordinate_clauses": tu["subordinate_clauses"],
                "num_clauses": tu["num_clauses"],
                "num_words": tu["num_words"]
            })
        
        return SyntacticMetricsResponse(
            yngve_depth=YngveDepthResponse(**yngve),
            frazier_depth=FrazierDepthResponse(**frazier),
            t_units=TUnitsResponse(
                t_units=t_units_list,
                num_t_units=t_units["num_t_units"],
                avg_words_per_tunit=t_units["avg_words_per_tunit"],
                avg_clauses_per_tunit=t_units["avg_clauses_per_tunit"],
                total_words=t_units["total_words"],
                total_clauses=t_units["total_clauses"]
            ),
            subordination_index=SubordinationIndexResponse(**subordination)
        )
    except Exception as e:
        logger.error(f"Error calculating syntactic metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Syntactic metrics calculation failed: {str(e)}"
        )

# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("🚀 Starting Linguistic Analysis Service...")
    config = get_config()
    logger.info(f"📊 Service: {config['service']['name']}")
    logger.info(f"🔌 Port: {config['service']['port']}")
    
    # Pre-load SpaCy model
    try:
        await parser._ensure_loaded()
        logger.info("✅ Linguistic Analysis Service ready")
    except Exception as e:
        logger.warning(f"⚠️  SpaCy model not loaded yet: {e}")
        logger.info("💡 Model will be loaded on first request")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Linguistic Analysis Service...")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        "app_complete:app",
        host=config["service"]["host"],
        port=config["service"]["port"],
        reload=True
    )

