"""
LLM Service Standalone - External API Service (Groq via LiteLLM)
This service uses external APIs (Groq) and does NOT load models locally.
No GPU required - it's just an API wrapper.
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
# project_root should be the workspace root (parle_backend/)
# From src/services/llm/app_complete.py, go up 4 levels: llm -> services -> src -> parle_backend
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env file from project root
def _load_env_file(force_reload=False):
    """Load environment variables from .env file"""
    try:
        # Try using python-dotenv first (more reliable)
        from dotenv import load_dotenv
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=force_reload)
            return
    except ImportError:
        pass
    
    # Fallback: manual parsing
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Set if not already in environment, or if force_reload
                    if key and (force_reload or not os.getenv(key)):
                        os.environ[key] = value

# Load .env before anything else
_load_env_file()

# Try to import local utils (fallback implementations if not available)
try:
    from .utils.route_helpers import add_standard_endpoints
    from .utils.metrics import increment_metric, set_gauge
except ImportError:
    # Fallback implementations for standalone mode
    def increment_metric(name, value=1, labels=None):
        pass

    def set_gauge(name, value, labels=None):
        pass

    def add_standard_endpoints(router, service_instance=None, service_name=None):
        pass

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "service": {
        "name": "llm",
        "port": 8006,
        "host": "0.0.0.0"
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    },
    "llm": {
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
        # Ensure .env is loaded
        _load_env_file(force_reload=True)
        
        # Use Gemini Flash 2.5 via OpenRouter
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            # Gemini Flash 2.5 via OpenRouter
            self.default_model = "openrouter/google/gemini-2.5-flash"  # Gemini Flash 2.5
            self.fallback_model = "openrouter/google/gemini-2.5-flash"
            logger.info(f"✅ Using OpenRouter with Gemini Flash 2.5 (key: {openrouter_key[:20]}...)")
        else:
            self.default_model = "openrouter/google/gemini-2.5-flash"
            self.fallback_model = "openrouter/google/gemini-2.5-flash"
            logger.warning("⚠️  OPENROUTER_API_KEY not found, using Gemini Flash 2.5 anyway (may fail)")
        
        self.timeout = 60
        self.max_retries = 3

        # Try to import litellm
        try:
            import litellm
            self.litellm = litellm
            self.available = True
            logger.info("✅ LiteLLM available")
        except ImportError:
            self.available = False
            logger.warning("⚠️  LiteLLM not available - LLM functionality disabled")

    def _get_api_key(self, api_key=None, model=None):
        """Get API key from parameter or environment based on model"""
        if api_key:
            return api_key
        
        # Check if model uses OpenRouter
        if model and "openrouter" in model.lower():
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                return openrouter_key
            raise ValueError("OPENROUTER_API_KEY required for OpenRouter models")
        
        # Check if model requires Anthropic API key
        if model and ("claude" in model.lower() or "anthropic" in model.lower()):
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                return anthropic_key
            # Fallback to GROQ if Anthropic not available
            return os.getenv("GROQ_API_KEY")
        
        # Default to Groq
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

        model = kwargs.get('model', self.default_model)
        api_key = self._get_api_key(kwargs.get('api_key'), model=model)
        if not api_key:
            raise HTTPException(status_code=500, detail="No API key available")

        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 1000)

        try:
            # Make API call
            start_time = time.time()
            # For OpenRouter models, use OPENROUTER_API_KEY
            if "openrouter" in model.lower():
                # Reload .env to ensure we have the latest key
                _load_env_file(force_reload=True)
                openrouter_key = os.getenv("OPENROUTER_API_KEY")
                
                # Debug: log if key is found
                if openrouter_key:
                    logger.info(f"✅ OPENROUTER_API_KEY found (length: {len(openrouter_key)})")
                else:
                    logger.error("❌ OPENROUTER_API_KEY not found in environment")
                    # Try reading directly from .env file
                    env_file = project_root / ".env"
                    if env_file.exists():
                        with open(env_file, 'r') as f:
                            for line in f:
                                if line.startswith('OPENROUTER_API_KEY='):
                                    openrouter_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                                    os.environ["OPENROUTER_API_KEY"] = openrouter_key
                                    logger.info(f"✅ Loaded OPENROUTER_API_KEY from .env file")
                                    break
                
                if not openrouter_key:
                    raise HTTPException(
                        status_code=400,
                        detail=f"OPENROUTER_API_KEY required for {model}. Set OPENROUTER_API_KEY environment variable."
                    )
                # LiteLLM supports OpenRouter models with format: openrouter/model_name
                # For OpenRouter, we need to pass the API key explicitly
                # LiteLLM will use it if provided, otherwise it looks for OPENROUTER_API_KEY env var
                # Set environment variable for LiteLLM
                os.environ["OPENROUTER_API_KEY"] = openrouter_key
                response = await self.litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=openrouter_key,
                    timeout=self.timeout
                )
            # For Anthropic models, use ANTHROPIC_API_KEY environment variable
            elif "claude" in model.lower() or "anthropic" in model.lower():
                anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                if not anthropic_key:
                    # Fallback: try to use Groq with a compatible model
                    print(f"⚠️  ANTHROPIC_API_KEY not set, cannot use {model}. Using fallback.")
                    raise HTTPException(
                        status_code=400,
                        detail=f"ANTHROPIC_API_KEY required for {model}. Set ANTHROPIC_API_KEY environment variable."
                    )
                # LiteLLM will automatically use ANTHROPIC_API_KEY env var for claude models
                response = await self.litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=anthropic_key,
                    timeout=self.timeout
                )
            else:
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

# Initialize provider lazily to ensure .env is loaded
llm_provider = None

def get_llm_provider():
    """Get or create LLM provider instance"""
    global llm_provider
    if llm_provider is None:
        # Ensure .env is loaded before creating provider
        _load_env_file(force_reload=True)
        llm_provider = SimpleLLMProvider()
    return llm_provider

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
    provider = get_llm_provider()
    provider_available = provider.available if provider else False
    models_available = len(provider.get_available_models()) > 0 if provider and provider.available else 0

    return {
        "status": "healthy" if provider_available else "degraded",
        "service": "llm",
        "timestamp": datetime.now().isoformat(),
        "llm_provider": {
            "available": provider_available,
            "models_count": models_available
        }
    }

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text completion"""
    try:
        provider = get_llm_provider()
        result = await provider.generate_text(
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
        provider = get_llm_provider()
        # Convert Pydantic messages to dict
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        result = await provider.chat_completion(
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
        provider = get_llm_provider()
        models = provider.get_available_models() if provider else []
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

# Add standard endpoints
# Create a minimal service instance for health checks
class MinimalService:
    async def health_check(self):
        return {"status": "healthy", "service": "llm"}
    def get_service_info(self):
        return {"service": "llm", "version": "1.0.0", "status": "running"}

minimal_service = MinimalService()
router = APIRouter()
add_standard_endpoints(router, minimal_service, "llm")
app.include_router(router)

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service"""
    print("🚀 Initializing LLM Service (External API - Groq)...")
    provider = get_llm_provider()
    if provider:
        print(f"   LLM Provider Available: {provider.available}")
        if provider.available:
            models = provider.get_available_models()
            print(f"   Available Models: {len(models)}")
            for model in models:
                print(f"     - {model['id']} ({model['provider']})")
    print("✅ LLM Service initialized successfully!")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8110"))
    print(f"Starting LLM Service (External API - Groq) on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
