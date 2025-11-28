"""
API Router Consolidado - Todos os endpoints em um único router
Usa módulos internos para chamadas diretas Python
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from loguru import logger
import base64
import jwt
import os

router = APIRouter()
security = HTTPBearer(auto_error=False)

# Cache de módulos (lazy initialization)
_modules_cache = {}


async def get_module(module_name: str):
    """Get module instance (cached and initialized)"""
    if module_name not in _modules_cache:
        from src.modules import module_factory
        module = module_factory.create(module_name)
        # Initialize module if it has initialize method and is not already initialized
        if hasattr(module, 'initialize'):
            is_initialized = getattr(module, 'initialized', False) or getattr(module, '_initialized', False)
            if not is_initialized:
                try:
                    result = await module.initialize()
                    if result is False:
                        logger.warning(f"Module {module_name} initialization returned False")
                except Exception as e:
                    logger.error(f"Failed to initialize {module_name}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
        _modules_cache[module_name] = module
    return _modules_cache[module_name]


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health():
    """Health check geral"""
    return {"status": "ok", "mode": "monolith"}


# ============================================================================
# Speech (STT/TTS)
# ============================================================================

class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    language: str = "pt"
    model: str = "whisper-large-v3"


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    provider: Optional[str] = None
    language: str = "pt"
    speed: float = 1.0


@router.post("/speech/stt/transcribe")
async def transcribe(request: TranscribeRequest):
    """STT transcription"""
    try:
        stt = await get_module("stt")
        result = await stt.transcribe(
            audio_base64=request.audio_base64,
            audio_url=request.audio_url,
            language=request.language,
            model=request.model
        )
        return result
    except Exception as e:
        logger.error(f"STT failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speech/stt/transcribe-file")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("pt"),
    model: str = Form("whisper-large-v3")
):
    """Transcribe uploaded file"""
    try:
        audio_data = await file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        stt = await get_module("stt")
        result = await stt.transcribe(
            audio_base64=audio_base64,
            language=language,
            model=model
        )
        return result
    except Exception as e:
        logger.error(f"STT failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speech/tts/synthesize")
async def synthesize(request: SynthesizeRequest):
    """TTS synthesis"""
    try:
        tts = await get_module("tts")
        result = await tts.synthesize(
            text=request.text,
            voice_id=request.voice_id,
            provider=request.provider,
            language=request.language,
            speed=request.speed
        )
        return result
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LLM
# ============================================================================

class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7
    system_prompt: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7


@router.post("/llm/generate")
async def generate(request: GenerateRequest):
    """LLM generation"""
    try:
        llm = await get_module("llm")
        result = await llm.generate(
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return result
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/chat")
async def chat(request: ChatRequest):
    """Chat completion"""
    try:
        llm = await get_module("llm")
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


# ============================================================================
# Auth
# ============================================================================

class LoginRequest(BaseModel):
    email: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        secret = os.getenv("JWT_SECRET_KEY", "your-secret-key")
        return jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.DecodeError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    """Get current authenticated user"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = verify_token(credentials.credentials)
    user = await get_module("user")
    user_data = await user.get_user(payload.get("user_id"))
    if not user_data:
        raise HTTPException(status_code=401, detail="User not found")
    return user_data


@router.post("/auth/login")
async def login(request: LoginRequest):
    """User login"""
    try:
        user = await get_module("user")
        result = await user.login(email=request.email, password=request.password)
        return result
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/register")
async def register(request: CreateUserRequest):
    """User registration"""
    try:
        user = await get_module("user")
        result = await user.create_user(
            username=request.username,
            email=request.email,
            password=request.password,
            full_name=request.full_name
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user"""
    return current_user


# ============================================================================
# Conversation (Orchestrator, Session, Scenarios, Store)
# ============================================================================

class ProcessRequest(BaseModel):
    session_id: str
    audio_base64: Optional[str] = None
    text: Optional[str] = None


@router.post("/conversation/process")
async def process_speech_to_speech(
    file: Optional[UploadFile] = File(None),
    audio_base64: Optional[str] = Form(None),
    session_id: str = Form(...),
    language: str = Form("pt"),
    voice_id: Optional[str] = Form(None),
    max_tokens: int = Form(100),
    temperature: float = Form(0.7)
):
    """Process speech-to-speech"""
    try:
        orchestrator = await get_module("orchestrator")
        if file:
            audio_data = await file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        if not audio_base64:
            raise HTTPException(status_code=400, detail="Audio required")
        result = await orchestrator.process_turn(
            session_id=session_id,
            audio_base64=audio_base64,
            language=language,
            voice_id=voice_id,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversation/session/create")
async def create_session(
    user_id: str = Form(...),
    scenario_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None)
):
    """Create session"""
    try:
        session = await get_module("session")
        return await session.create_session(user_id=user_id, scenario_id=scenario_id, conversation_id=conversation_id)
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/session/{session_id}")
async def get_session(session_id: str):
    """Get session"""
    try:
        session = await get_module("session")
        result = await session.get_session(session_id)
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversation/store/save")
async def save_message(
    conversation_id: str = Form(...),
    role: str = Form(...),
    content: str = Form(...)
):
    """Save message"""
    try:
        store = await get_module("conversation_store")
        return await store.save_message(conversation_id=conversation_id, role=role, content=content)
    except Exception as e:
        logger.error(f"Failed to save message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/store/{conversation_id}/context")
async def get_context(conversation_id: str, limit: int = 10):
    """Get conversation context"""
    try:
        store = await get_module("conversation_store")
        return await store.get_context(conversation_id=conversation_id, limit=limit)
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Tutoring (Diagnostic, Pedagogical Policy, Learning Path, Student Model)
# ============================================================================

class AnalyzeTurnRequest(BaseModel):
    user_text: str
    ai_text: Optional[str] = None
    valid_skills: Optional[List[str]] = None


@router.post("/tutoring/diagnostic/analyze")
async def analyze_turn(request: AnalyzeTurnRequest):
    """Analyze conversation turn"""
    try:
        diagnostic = await get_module("diagnostic_module")
        return await diagnostic.analyze_turn(
            user_text=request.user_text,
            ai_text=request.ai_text,
            valid_skills=request.valid_skills
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tutoring/policy/compose")
async def compose_prompt(context: dict):
    """Compose pedagogical prompt"""
    try:
        policy = await get_module("pedagogical_policy")
        return await policy.compose_prompt(context=context)
    except Exception as e:
        logger.error(f"Prompt composition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tutoring/path/{user_id}/next")
async def get_next_skill(user_id: str, cefr_level: Optional[str] = None):
    """Get next skill"""
    try:
        path = await get_module("learning_path")
        return await path.get_next_skill(user_id=user_id, cefr_level=cefr_level)
    except Exception as e:
        logger.error(f"Failed to get next skill: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tutoring/student/{user_id}/profile")
async def get_student_profile(user_id: str):
    """Get student profile"""
    try:
        student = await get_module("student_model")
        return await student.get_profile(user_id)
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Storage (File Storage, Database)
# ============================================================================

@router.post("/storage/file/upload")
async def upload_file(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Upload file"""
    try:
        file_storage = await get_module("file_storage")
        file_content = await file.read()
        tags_list = tags.split(",") if tags else None
        metadata_dict = eval(metadata) if metadata else None
        return await file_storage.upload_file(
            file_content=file_content,
            filename=file.filename,
            tags=tags_list,
            metadata=metadata_dict
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/file/{file_id}")
async def download_file(file_id: str):
    """Download file"""
    try:
        file_storage = await get_module("file_storage")
        file_content = await file_storage.download_file(file_id)
        if not file_content:
            raise HTTPException(status_code=404, detail="File not found")
        from fastapi.responses import Response
        return Response(content=file_content, media_type="application/octet-stream")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/storage/database/set")
async def set_data(user_id: str = Form(...), key: str = Form(...), value: dict = Form(...)):
    """Set data"""
    try:
        db = await get_module("database")
        return await db.set_data(user_id=user_id, key=key, value=value)
    except Exception as e:
        logger.error(f"Failed to set data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage/database/get")
async def get_data(user_id: str, key: str):
    """Get data"""
    try:
        db = await get_module("database")
        result = await db.get_data(user_id=user_id, key=key)
        if result is None:
            raise HTTPException(status_code=404, detail="Data not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
