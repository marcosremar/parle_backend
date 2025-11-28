"""
Conversation Router - Orchestrator, Session, Scenarios
Usa módulos internos para chamadas diretas Python
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger
import base64
import time

router = APIRouter()

# Module instances (lazy initialization)
_orchestrator_module = None
_session_module = None
_scenarios_module = None
_conversation_store_module = None


def get_orchestrator_module():
    """Get Orchestrator module instance"""
    global _orchestrator_module
    if _orchestrator_module is None:
        from src.modules import create
        _orchestrator_module = create("orchestrator")
    return _orchestrator_module


def get_session_module():
    """Get Session module instance"""
    global _session_module
    if _session_module is None:
        from src.modules import create
        _session_module = create("session")
    return _session_module


def get_scenarios_module():
    """Get Scenarios module instance"""
    global _scenarios_module
    if _scenarios_module is None:
        from src.modules import create
        _scenarios_module = create("scenarios")
    return _scenarios_module


def get_conversation_store_module():
    """Get Conversation Store module instance"""
    global _conversation_store_module
    if _conversation_store_module is None:
        from src.modules import create
        _conversation_store_module = create("conversation_store")
    return _conversation_store_module


# Pydantic models
class ProcessRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    text: Optional[str] = Field(None, description="Text input (if no audio)")


@router.get("/health")
async def health():
    """Health check for conversation services"""
    return {"status": "ok", "services": ["orchestrator", "session", "scenarios", "conversation_store"]}


# Session endpoints
@router.post("/session/create")
async def create_session(
    user_id: str = Form(...),
    scenario_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None)
):
    """Create a new session"""
    try:
        session = get_session_module()
        result = await session.create_session(
            user_id=user_id,
            scenario_id=scenario_id,
            conversation_id=conversation_id
        )
        return result
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session by ID"""
    try:
        session = get_session_module()
        result = await session.get_session(session_id)
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session")
async def list_sessions(user_id: Optional[str] = None):
    """List sessions"""
    try:
        session = get_session_module()
        return await session.list_sessions(user_id=user_id)
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Scenarios endpoints
@router.post("/scenarios/create")
async def create_scenario(
    name: str = Form(...),
    description: str = Form(...),
    system_prompt: str = Form(...)
):
    """Create a new scenario"""
    try:
        scenarios = get_scenarios_module()
        result = await scenarios.create_scenario(
            name=name,
            description=description,
            system_prompt=system_prompt
        )
        return result
    except Exception as e:
        logger.error(f"Scenario creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scenarios/{scenario_id}")
async def get_scenario(scenario_id: str):
    """Get scenario by ID"""
    try:
        scenarios = get_scenarios_module()
        result = await scenarios.get_scenario(scenario_id)
        if not result:
            raise HTTPException(status_code=404, detail="Scenario not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scenarios")
async def list_scenarios():
    """List all scenarios"""
    try:
        scenarios = get_scenarios_module()
        return await scenarios.list_scenarios()
    except Exception as e:
        logger.error(f"Failed to list scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Conversation Store endpoints
@router.post("/conversation-store/save")
async def save_message(
    conversation_id: str = Form(...),
    role: str = Form(...),
    content: str = Form(...)
):
    """Save a message to conversation"""
    try:
        store = get_conversation_store_module()
        result = await store.save_message(
            conversation_id=conversation_id,
            role=role,
            content=content
        )
        return result
    except Exception as e:
        logger.error(f"Failed to save message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation-store/{conversation_id}/context")
async def get_context(conversation_id: str, limit: int = 10):
    """Get conversation context"""
    try:
        store = get_conversation_store_module()
        return await store.get_context(conversation_id=conversation_id, limit=limit)
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orchestrator/process")
async def process_speech_to_speech(
    file: Optional[UploadFile] = File(None),
    audio_base64: Optional[str] = Form(None),
    session_id: str = Form(...),
    language: str = Form("pt"),
    voice_id: Optional[str] = Form(None),
    max_tokens: int = Form(100),
    temperature: float = Form(0.7),
    voice_speed: float = Form(1.0),
    stt_model: str = Form("whisper-large-v3")
):
    """
    Process speech-to-speech using orchestrator module
    
    Returns:
        Dict with audio_base64, transcription, response_text, processing_times
    """
    try:
        orchestrator = get_orchestrator_module()
        
        # Get audio data
        audio_data = None
        if file:
            audio_data = await file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        elif audio_base64:
            pass  # Already base64
        else:
            raise HTTPException(status_code=400, detail="Either file or audio_base64 required")
        
        # Process turn
        result = await orchestrator.process_turn(
            session_id=session_id,
            audio_base64=audio_base64,
            language=language,
            voice_id=voice_id,
            max_tokens=max_tokens,
            temperature=temperature,
            voice_speed=voice_speed,
            stt_model=stt_model
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Orchestrator processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orchestrator/stats")
async def get_orchestrator_stats():
    """Get orchestrator statistics"""
    try:
        orchestrator = get_orchestrator_module()
        return await orchestrator.get_stats()
    except Exception as e:
        logger.error(f"Failed to get orchestrator stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orchestrator/process-turn")
async def process_turn(request: ProcessRequest):
    """Process a conversation turn using orchestrator module"""
    try:
        orchestrator = get_orchestrator_module()
        result = await orchestrator.process_turn(
            session_id=request.session_id,
            audio_base64=request.audio_base64,
            text=request.text
        )
        return result
    except Exception as e:
        logger.error(f"Turn processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
