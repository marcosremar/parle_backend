"""
LLM Router
Usa módulo interno para chamadas diretas Python
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from loguru import logger

router = APIRouter()

# Module instance (lazy initialization)
_llm_module = None


def get_llm_module():
    """Get LLM module instance"""
    global _llm_module
    if _llm_module is None:
        from src.modules import create
        _llm_module = create("llm")
    return _llm_module


# Pydantic models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    model: Optional[str] = Field(None, description="Model to use")
    max_tokens: int = Field(500, description="Maximum tokens")
    temperature: float = Field(0.7, description="Sampling temperature")
    system_prompt: Optional[str] = Field(None, description="System prompt")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    model: Optional[str] = Field(None, description="Model to use")
    max_tokens: int = Field(500, description="Maximum tokens")
    temperature: float = Field(0.7, description="Sampling temperature")


@router.get("/health")
async def health():
    """Health check for LLM service"""
    return {"status": "ok", "service": "llm"}


@router.post("/generate")
async def generate(request: GenerateRequest):
    """LLM generation endpoint using direct module"""
    try:
        llm = get_llm_module()
        
        # Build messages if system prompt provided
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        result = await llm.generate(
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return result
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat(request: ChatRequest):
    """Chat completion endpoint using direct module"""
    try:
        llm = get_llm_module()
        
        # Convert Pydantic messages to dict
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        result = await llm.chat(
            messages=messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return result
    except Exception as e:
        logger.error(f"LLM chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_models():
    """Get available LLM models"""
    try:
        llm = get_llm_module()
        return await llm.get_models()
    except Exception as e:
        logger.error(f"Failed to get LLM models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
