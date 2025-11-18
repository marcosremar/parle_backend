"""
External LLM Service Standalone - Consolidated for Nomad deployment
"""
import uvicorn
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, Header, APIRouter
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import logging
import time
from loguru import logger

# Add project root to path for src imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import src modules (fallback to local if not available)
try:
    from src.core.route_helpers import add_standard_endpoints
    from src.core.metrics import increment_metric, set_gauge
except ImportError:
    # Fallback implementations for standalone mode
    def increment_metric(name, value=1, labels=None):
        pass

    def set_gauge(name, value, labels=None):
        pass

    def add_standard_endpoints(router):
        pass

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "external_llm",
        "port": 8110,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    },
    "external_llm": {
        "default_model": "groq/llama-3.1-8b-instant",
        "fallback_model": "groq/llama-3.1-70b-versatile",
        "timeout_seconds": 60,
        "max_retries": 3,
        "cache_enabled": True
    }
}

def get_config():
    """Get external llm service configuration"""
    config = DEFAULT_CONFIG.copy()
    return config

# ============================================================================
# Pydantic Models (Standalone)
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message"""
    role: str = Field(..., description="Message role (system/user/assistant)")
    content: str = Field(..., description="Message content")

class GenerateRequest(BaseModel):
    """Text generation request"""
    prompt: str
    model: Optional[str] = Field(default="groq/llama-3.1-8b-instant", description="Model to use")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1000, ge=1, le=32000)
    stream: Optional[bool] = Field(default=False, description="Stream response")
    api_key: Optional[str] = Field(default=None, description="Optional API key override")

class ChatRequest(BaseModel):
    """Chat completion request"""
    messages: List[ChatMessage]
    model: Optional[str] = Field(default="groq/llama-3.1-8b-instant")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1000, ge=1, le=32000)
    stream: Optional[bool] = Field(default=False)
    api_key: Optional[str] = Field(default=None)

class ModelInfo(BaseModel):
    """Model information"""
    id: str
    provider: str
    supports_chat: bool = True
    supports_streaming: bool = True
    max_tokens: Optional[int] = None
    description: Optional[str] = None

# ============================================================================
# Simple LLM Provider (Standalone)
# ============================================================================

class SimpleLLMProvider:
    """Simple LLM provider using LiteLLM"""

    def __init__(self):
        self.default_model = "groq/llama-3.1-8b-instant"
        self.fallback_model = "groq/llama-3.1-70b-instant"  # Updated fallback model
        self.timeout = 60
        self.max_retries = 3

        # Try to import litellm
        try:
            import litellm
            self.litellm = litellm
            self.available = True
            print("✅ LiteLLM available")
        except ImportError:
            self.available = False
            print("⚠️  LiteLLM not available - LLM functionality disabled")

    def _get_api_key(self, api_key=None):
        """Get API key from parameter or environment"""
        if api_key:
            return api_key
        return os.getenv("GROQ_API_KEY")

    async def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text completion"""
        if not self.available:
            raise HTTPException(status_code=503, detail="LLM provider not available")

        api_key = self._get_api_key(kwargs.get('api_key'))
        if not api_key:
            raise HTTPException(status_code=500, detail="No API key available")

        model = kwargs.get('model', self.default_model)
        system_prompt = kwargs.get('system_prompt')
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 1000)

        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Make API call
            start_time = time.time()
            response = await self.litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                timeout=self.timeout
            )
            end_time = time.time()

            # Extract response
            generated_text = response.choices[0].message.content
            usage = response.usage

            return {
                "text": generated_text,
                "model": model,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                },
                "latency_ms": (end_time - start_time) * 1000,
                "cached": False
            }

        except Exception as e:
            # Try fallback model
            if model == self.default_model:
                try:
                    print(f"⚠️  Primary model failed, trying fallback: {self.fallback_model}")
                    response = await self.litellm.acompletion(
                        model=self.fallback_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                        timeout=self.timeout
                    )

                    generated_text = response.choices[0].message.content
                    usage = response.usage

                    return {
                        "text": generated_text,
                        "model": self.fallback_model,
                        "usage": {
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens
                        },
                        "latency_ms": (time.time() - start_time) * 1000,
                        "cached": False,
                        "fallback_used": True
                    }

                except Exception as fallback_error:
                    print(f"❌ Fallback model also failed: {fallback_error}")

            raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Chat completion"""
        if not self.available:
            raise HTTPException(status_code=503, detail="LLM provider not available")

        api_key = self._get_api_key(kwargs.get('api_key'))
        if not api_key:
            raise HTTPException(status_code=500, detail="No API key available")

        model = kwargs.get('model', self.default_model)
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 1000)

        try:
            # Make API call
            start_time = time.time()
            response = await self.litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                timeout=self.timeout
            )
            end_time = time.time()

            # Extract response
            generated_text = response.choices[0].message.content
            usage = response.usage

            return {
                "text": generated_text,
                "model": model,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                },
                "latency_ms": (end_time - start_time) * 1000,
                "cached": False
            }

        except Exception as e:
            # Try fallback model
            if model == self.default_model:
                try:
                    print(f"⚠️  Primary model failed, trying fallback: {self.fallback_model}")
                    response = await self.litellm.acompletion(
                        model=self.fallback_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                        timeout=self.timeout
                    )

                    generated_text = response.choices[0].message.content
                    usage = response.usage

                    return {
                        "text": generated_text,
                        "model": self.fallback_model,
                        "usage": {
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens
                        },
                        "latency_ms": (time.time() - start_time) * 1000,
                        "cached": False,
                        "fallback_used": True
                    }

                except Exception as fallback_error:
                    print(f"❌ Fallback model also failed: {fallback_error}")

            raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models"""
        if not self.available:
            return []

        try:
            # Return known models
            return [
                {
                    "id": "groq/llama-3.1-8b-instant",
                    "provider": "groq",
                    "supports_chat": True,
                    "supports_streaming": True,
                    "max_tokens": 8000,
                    "description": "Fast inference model"
                },
                {
                    "id": "groq/llama-3.1-70b-instant",
                    "provider": "groq",
                    "supports_chat": True,
                    "supports_streaming": True,
                    "max_tokens": 8000,
                    "description": "High-quality large model"
                }
            ]
        except Exception as e:
            print(f"Error getting models: {e}")
            return []

# ============================================================================
# Global Provider Instance
# ============================================================================

llm_provider = SimpleLLMProvider()

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="External LLM Service", version="1.0.0")

# ============================================================================
# Routes
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    models_available = len(llm_provider.get_available_models()) > 0
    provider_available = llm_provider.available

    return {
        "status": "healthy" if provider_available else "degraded",
        "service": "external_llm",
        "timestamp": datetime.now().isoformat(),
        "llm_provider": {
            "available": provider_available,
            "models_count": len(llm_provider.get_available_models()) if provider_available else 0
        }
    }

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text completion"""
    try:
        result = await llm_provider.generate_text(
            prompt=request.prompt,
            model=request.model,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            api_key=request.api_key
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/chat")
async def chat_completion(request: ChatRequest):
    """Chat completion"""
    try:
        # Convert Pydantic messages to dict
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        result = await llm_provider.chat_completion(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            api_key=request.api_key
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

@app.get("/models")
async def get_models():
    """Get available models"""
    try:
        models = llm_provider.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

# Add standard endpoints
router = APIRouter()
add_standard_endpoints(router)
app.include_router(router)

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service"""
    print("🚀 Initializing External LLM Service...")
    print(f"   LLM Provider Available: {llm_provider.available}")
    if llm_provider.available:
        models = llm_provider.get_available_models()
        print(f"   Available Models: {len(models)}")
        for model in models:
            print(f"     - {model['id']} ({model['provider']})")
    print("✅ External LLM Service initialized successfully!")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8110"))
    print(f"Starting External LLM Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
