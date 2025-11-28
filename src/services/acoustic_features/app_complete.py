"""
Acoustic Features Service - FastAPI Application
Extracts acoustic features from audio using Wav2Vec 2.0 models.
"""

import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from loguru import logger
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .models import ExtractFeaturesResponse, HealthResponse
from .wav2vec_extractor import Wav2VecExtractor
from .multi_embedding_fusion import MultiEmbeddingFusion

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "acoustic_features",
        "port": 8970,
        "host": "0.0.0.0"
    }
}

def get_config():
    """Get service configuration"""
    config = DEFAULT_CONFIG.copy()
    port = int(os.getenv("ACOUSTIC_FEATURES_PORT", os.getenv("PORT", "8970")))
    config["service"]["port"] = port
    return config

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Acoustic Features Service",
    version="1.0.0",
    description="Service for extracting acoustic features using Wav2Vec 2.0"
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
# Service Instances
# ============================================================================

# Initialize extractor and fusion
device = os.getenv("PYTORCH_DEVICE", None)  # 'mps', 'cuda', or None for auto
extractor: Optional[Wav2VecExtractor] = None
fusion: Optional[MultiEmbeddingFusion] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global extractor, fusion
    
    try:
        logger.info("Initializing Wav2Vec extractor...")
        extractor = Wav2VecExtractor(device=device)
        
        logger.info("Initializing multi-embedding fusion...")
        fusion = MultiEmbeddingFusion(embed_dim=768, num_heads=8)
        fusion.eval()
        
        logger.info("Acoustic Features Service ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Acoustic Features Service")

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    models_loaded = extractor is not None and fusion is not None
    
    return HealthResponse(
        status="healthy" if models_loaded else "initializing",
        service="acoustic_features",
        version="1.0.0",
        device=extractor.device if extractor else "unknown",
        models_loaded=models_loaded
    )

@app.post("/api/acoustic/extract_features", response_model=ExtractFeaturesResponse)
async def extract_features(audio: UploadFile = File(...)):
    """
    Extract acoustic features from audio file.
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
        
    Returns:
        Extracted features and embedding shapes
    """
    if extractor is None or fusion is None:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Models are still loading."
        )
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        # Create temporary file
        suffix = Path(audio.filename).suffix if audio.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_file = tmp.name
            shutil.copyfileobj(audio.file, tmp)
        
        logger.info(f"Processing audio file: {audio.filename}")
        
        # Extract embeddings
        embeddings = await extractor.extract_embeddings(temp_file)
        
        # Fuse embeddings
        native_emb = embeddings["native_embedding"]
        learner_emb = embeddings["learner_embedding"]
        
        # Add batch dimension if needed
        if len(native_emb.shape) == 2:  # (seq_len, embed_dim)
            native_emb = native_emb.unsqueeze(1)  # (seq_len, 1, embed_dim)
        if len(learner_emb.shape) == 2:
            learner_emb = learner_emb.unsqueeze(1)
        
        fused = fusion(native_emb, learner_emb)
        
        # Convert to list for JSON response
        features = fused.squeeze().cpu().numpy().tolist()
        
        # Get model info
        model_info = extractor.get_model_info()
        
        return ExtractFeaturesResponse(
            features=features,
            native_embedding_shape=list(native_emb.shape),
            learner_embedding_shape=list(learner_emb.shape),
            fused_shape=list(fused.shape),
            model_info={
                "model_name": model_info["model_name"],
                "device": model_info["device"]
            }
        )
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract features: {str(e)}"
        )
    
    finally:
        # Cleanup temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")

@app.get("/api/acoustic/model_info")
async def get_model_info():
    """Get information about loaded models"""
    if extractor is None:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized"
        )
    
    return extractor.get_model_info()

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = get_config()
    
    logger.info(f"Starting Acoustic Features Service on port {config['service']['port']}")
    
    uvicorn.run(
        "app_complete:app",
        host=config["service"]["host"],
        port=config["service"]["port"],
        reload=False
    )
