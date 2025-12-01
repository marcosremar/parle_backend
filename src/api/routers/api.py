"""
API Router Consolidado - Todos os endpoints em um único router
Usa módulos internos para chamadas diretas Python
"""

import base64
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from loguru import logger
from pydantic import BaseModel, Field

from src.core.audit_log import log_auth_event
from src.core.config import get_config
from src.core.constants import (
    AUTH_LOGIN_RATE_LIMIT,
    AUTH_REGISTER_RATE_LIMIT,
    CONVERSATION_RATE_LIMIT,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    TUTORING_RATE_LIMIT,
)
from src.core.security import (
    safe_log_error,
    validate_content_type,
    validate_upload_size,
)

config = get_config()

# Limiter will be set from main app
app_limiter = None


def apply_rate_limit(limit: str):
    """Apply rate limit using app limiter"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Find Request in args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                request = kwargs.get("request") or kwargs.get("http_request")

            if request and app_limiter:
                try:
                    limited_func = app_limiter.limit(limit)(func)
                    return await limited_func(*args, **kwargs)
                except Exception as e:
                    from slowapi.errors import RateLimitExceeded

                    if isinstance(e, RateLimitExceeded):
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Rate limit exceeded: {limit}",
                        )
                    raise
            return await func(*args, **kwargs)

        return wrapper

    return decorator


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
        if hasattr(module, "initialize"):
            is_initialized = getattr(module, "initialized", False) or getattr(
                module, "_initialized", False
            )
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


@router.get(
    "/health",
    summary="Health check",
    description="Verifica o status de saúde da API",
    tags=["Health"],
)
async def health():
    """
    Health check endpoint

    Returns the health status of the API and modules.

    Returns:
        Dict with:
        - status: Health status ("ok" or "error")
        - mode: Operation mode ("monolith")

    Example:
        {
            "status": "ok",
            "mode": "monolith"
        }
    """
    return {"status": "ok", "mode": "monolith"}


# ============================================================================
# Speech (STT/TTS)
# ============================================================================


class TranscribeRequest(BaseModel):
    audio_base64: str | None = None
    audio_url: str | None = None
    language: str = "pt"
    model: str = "whisper-large-v3"


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str | None = None
    provider: str | None = None
    language: str = "pt"
    speed: float = 1.0


@router.post(
    "/speech/stt/transcribe",
    summary="Transcrever áudio para texto",
    description="""
    Transcreve áudio para texto usando o provedor STT configurado.
    
    Suporta áudio em base64 ou URL. O sistema detecta automaticamente o idioma
    se não especificado.
    """,
    response_description="Resultado da transcrição com texto e metadados",
    tags=["Speech", "STT"],
)
async def transcribe(request: TranscribeRequest):
    """
    STT transcription endpoint

    Transcribes audio to text using the configured STT provider.

    Args:
        request: TranscribeRequest with audio_base64 or audio_url

    Returns:
        Dict with:
        - text: Transcribed text
        - language: Detected language
        - duration: Audio duration (if available)
        - model: Model used
        - provider: Provider used

    Raises:
        HTTPException: 500 if transcription fails

    Example:
        {
            "audio_base64": "base64_encoded_audio...",
            "language": "pt",
            "model": "whisper-large-v3"
        }
    """
    try:
        stt = await get_module("stt")

        # Use circuit breaker if available
        if hasattr(stt, "transcribe_with_fallback"):
            result = await stt.transcribe_with_fallback(
                audio_base64=request.audio_base64,
                audio_url=request.audio_url,
                language=request.language,
                model=request.model,
            )
        else:
            result = await stt.transcribe(
                audio_base64=request.audio_base64,
                audio_url=request.audio_url,
                language=request.language,
                model=request.model,
            )
        return result
    except Exception as e:
        logger.error(f"STT failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speech/stt/transcribe-file")
async def transcribe_file(
    file: UploadFile = File(...), language: str = Form("pt"), model: str = Form("whisper-large-v3")
):
    """Transcribe uploaded file"""
    try:
        audio_data = await file.read()
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        stt = await get_module("stt")
        result = await stt.transcribe(audio_base64=audio_base64, language=language, model=model)
        return result
    except Exception as e:
        logger.error(f"STT failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/speech/tts/synthesize",
    summary="Sintetizar texto para áudio",
    description="""
    Converte texto em áudio usando o provedor TTS configurado.
    
    Suporta múltiplos provedores (ElevenLabs, HuggingFace, etc.) e permite
    personalização de voz, velocidade e idioma.
    """,
    response_description="Áudio sintetizado em base64 com metadados",
    tags=["Speech", "TTS"],
)
async def synthesize(request: SynthesizeRequest):
    """
    TTS synthesis endpoint

    Converts text to speech audio using the configured TTS provider.

    Args:
        request: SynthesizeRequest with text and optional voice settings

    Returns:
        Dict with:
        - audio_base64: Base64 encoded audio data
        - format: Audio format (wav, mp3, etc.)
        - provider: Provider used
        - voice_id: Voice ID used
        - duration: Audio duration (if available)
        - sample_rate: Sample rate (if available)

    Raises:
        HTTPException: 500 if synthesis fails

    Example:
        {
            "text": "Hello, world!",
            "voice_id": "Rachel",
            "language": "pt",
            "speed": 1.0
        }
    """
    try:
        tts = await get_module("tts")

        # Use circuit breaker if available
        if hasattr(tts, "synthesize_with_fallback"):
            result = await tts.synthesize_with_fallback(
                text=request.text,
                voice_id=request.voice_id,
                provider=request.provider,
                language=request.language,
                speed=request.speed,
            )
        else:
            result = await tts.synthesize(
                text=request.text,
                voice_id=request.voice_id,
                provider=request.provider,
                language=request.language,
                speed=request.speed,
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
    model: str | None = None
    max_tokens: int = 500
    temperature: float = 0.7
    system_prompt: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    max_tokens: int = 500
    temperature: float = 0.7


@router.post(
    "/llm/generate",
    summary="Gerar texto com LLM",
    description="""
    Gera texto usando modelos de linguagem (LLM).
    
    Suporta múltiplos provedores via LiteLLM e permite configuração de
    temperatura, max_tokens e system prompts.
    """,
    response_description="Texto gerado pelo LLM",
    tags=["LLM", "AI"],
)
async def generate(request: GenerateRequest):
    """
    LLM generation endpoint

    Generates text using language models.

    Args:
        request: GenerateRequest with prompt and generation parameters

    Returns:
        Dict with generated text and metadata

    Raises:
        HTTPException: 500 if generation fails
    """
    try:
        llm = await get_module("llm")

        # Use circuit breaker if available
        if hasattr(llm, "generate_with_fallback"):
            result = await llm.generate_with_fallback(
                prompt=request.prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        else:
            result = await llm.generate(
                prompt=request.prompt,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        return result
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/llm/chat",
    summary="Chat completion com LLM",
    description="""
    Completa conversas usando modelos de linguagem.
    
    Aceita uma lista de mensagens com roles (user, assistant, system) e
    retorna a resposta do assistente.
    """,
    response_description="Resposta do chat",
    tags=["LLM", "AI", "Chat"],
)
async def chat(request: ChatRequest):
    """
    Chat completion endpoint

    Completes chat conversations using language models.

    Args:
        request: ChatRequest with messages list and parameters

    Returns:
        Dict with assistant response

    Raises:
        HTTPException: 500 if chat completion fails
    """
    try:
        llm = await get_module("llm")
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        result = await llm.chat(
            messages=messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
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
    full_name: str | None = None


def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        secret = os.getenv("JWT_SECRET_KEY", "your-secret-key")
        # verify_exp=True is default, but being explicit
        return jwt.decode(token, secret, algorithms=["HS256"], options={"verify_exp": True})
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except (jwt.DecodeError, jwt.InvalidTokenError):
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """Get current authenticated user"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = verify_token(credentials.credentials)
    user = await get_module("user")
    user_data = await user.get_user(payload.get("user_id"))
    if not user_data:
        raise HTTPException(status_code=401, detail="User not found")
    return user_data


@router.post(
    "/auth/login",
    summary="Login de usuário",
    description="Autentica um usuário e retorna token JWT",
    response_description="Token JWT e informações do usuário",
    tags=["Auth", "Authentication"],
)
@apply_rate_limit(AUTH_LOGIN_RATE_LIMIT)
async def login(request: LoginRequest, http_request: Request = None):
    """
    User login endpoint

    Authenticates a user with email and password, returns JWT token.

    Args:
        request: LoginRequest with email and password

    Returns:
        Dict with:
        - token: JWT authentication token
        - user: User information
        - expires_in: Token expiration time in seconds

    Raises:
        HTTPException: 401 if credentials are invalid
        HTTPException: 500 if login fails

    Example:
        {
            "email": "user@example.com",
            "password": "securepassword"
        }
    """
    try:
        user = await get_module("user")
        result = await user.login(email=request.email, password=request.password)

        # Audit log
        user_id = result.get("user", {}).get("id") if isinstance(result, dict) else None
        ip_address = http_request.client.host if http_request and http_request.client else None
        log_auth_event(
            "login", user_id=user_id, email=request.email, ip_address=ip_address, success=True
        )

        return result
    except ValueError as e:
        # Audit log failed login
        ip_address = http_request.client.host if http_request and http_request.client else None
        log_auth_event(
            "login",
            email=request.email,
            ip_address=ip_address,
            success=False,
            details={"error": str(e)},
        )
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/auth/register",
    summary="Registro de usuário",
    description="Cria uma nova conta de usuário",
    response_description="Informações do usuário criado e token JWT",
    tags=["Auth", "Authentication"],
)
@apply_rate_limit(AUTH_REGISTER_RATE_LIMIT)
async def register(request: CreateUserRequest, http_request: Request = None):
    """
    User registration endpoint

    Creates a new user account.

    Args:
        request: CreateUserRequest with user information

    Returns:
        Dict with:
        - user: Created user information
        - token: JWT authentication token

    Raises:
        HTTPException: 400 if validation fails (email exists, weak password, etc.)
        HTTPException: 500 if registration fails

    Example:
        {
            "username": "johndoe",
            "email": "john@example.com",
            "password": "securepassword",
            "full_name": "John Doe"
        }
    """
    try:
        user = await get_module("user")
        result = await user.create_user(
            username=request.username,
            email=request.email,
            password=request.password,
            full_name=request.full_name,
        )

        # Audit log
        user_id = result.get("user", {}).get("id") if isinstance(result, dict) else None
        ip_address = http_request.client.host if http_request and http_request.client else None
        log_auth_event(
            "register", user_id=user_id, email=request.email, ip_address=ip_address, success=True
        )

        return result
    except ValueError as e:
        # Audit log failed registration
        ip_address = http_request.client.host if http_request and http_request.client else None
        log_auth_event(
            "register",
            email=request.email,
            ip_address=ip_address,
            success=False,
            details={"error": str(e)},
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/auth/me",
    summary="Obter usuário atual",
    description="Retorna informações do usuário autenticado",
    response_description="Informações do usuário",
    tags=["Auth", "User"],
)
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    Get current authenticated user endpoint

    Returns information about the currently authenticated user.

    Args:
        current_user: Authenticated user (from dependency)

    Returns:
        Dict with user information:
        - id: User ID
        - username: Username
        - email: Email address
        - full_name: Full name (if available)
        - created_at: Account creation timestamp

    Raises:
        HTTPException: 401 if not authenticated
    """
    return current_user


# ============================================================================
# Conversation (Orchestrator, Session, Scenarios, Store)
# ============================================================================


class ConversationRequest(BaseModel):
    """Unified conversation request - accepts text or audio"""

    message: str | None = Field(None, description="Text message (for text conversation)")
    session_id: str = Field(..., description="Session ID")
    voice_id: str | None = Field(None, description="Optional voice ID for audio response")
    language: str = Field("pt", description="Language code")
    max_tokens: int = Field(512, description="Max tokens for LLM")
    temperature: float = Field(0.7, description="Temperature for LLM")


@router.post(
    "/conversation",
    summary="Conversação unificada",
    description="""
    Endpoint unificado para conversação que aceita texto ou áudio.
    
    Suporta dois modos:
    - Texto: Envie campo 'message' para conversação por texto
    - Áudio: Envie 'file' ou 'audio_base64' para speech-to-speech
    
    O sistema processa automaticamente e retorna resposta no mesmo formato.
    """,
    response_description="Resposta da conversação (texto ou áudio)",
    tags=["Conversation", "AI"],
)
@apply_rate_limit(CONVERSATION_RATE_LIMIT)
async def conversation(
    http_request: Request,
    request: ConversationRequest = None,
    file: UploadFile | None = File(None),
    audio_base64: str | None = Form(None),
    message: str | None = Form(None),
    session_id: str = Form(...),
    voice_id: str | None = Form(None),
    language: str = Form("pt"),
    max_tokens: int = Form(100),
    temperature: float = Form(0.7),
):
    """
    Unified conversation endpoint - accepts text or audio input

    Supports both:
    - Text conversation: send 'message' field
    - Speech-to-speech: send 'file' or 'audio_base64' field

    Args:
        http_request: FastAPI Request object
        request: ConversationRequest (optional, for JSON body)
        file: Uploaded audio file (optional)
        audio_base64: Base64 encoded audio (optional)
        message: Text message (optional)
        session_id: Session ID (required)
        voice_id: Voice ID for audio response (optional)
        language: Language code (default: "pt")
        max_tokens: Max tokens for LLM (default: 100)
        temperature: Temperature for LLM (default: 0.7)

    Returns:
        Dict with:
        - For text: {"text": "response text"}
        - For audio: {"audio_base64": "...", "text": "transcribed text"}

    Raises:
        HTTPException: 400 if no input provided
        HTTPException: 413 if file too large
        HTTPException: 500 if processing fails
    """
    try:
        orchestrator = await get_module("orchestrator")

        # Get orchestrator engine
        if hasattr(orchestrator, "orchestrator"):
            orchestrator_engine = orchestrator.orchestrator
        elif hasattr(orchestrator, "process_text_conversation"):
            # Module has direct method
            pass
        else:
            raise HTTPException(
                status_code=500, detail="Orchestrator module not properly initialized"
            )

        # Handle text conversation
        if message or (request and request.message):
            text_message = message or (request.message if request else None)
            if hasattr(orchestrator, "process_text_conversation"):
                result = await orchestrator.process_text_conversation(
                    message=text_message, session_id=session_id, voice_id=voice_id
                )
            else:
                result = await orchestrator_engine.process_text_conversation(
                    message=text_message, session_id=session_id, voice_id=voice_id
                )
            return result

        # Handle audio conversation (speech-to-speech)
        if file:
            # Validate file size
            validate_upload_size(file.size, config.server.max_upload_size_mb)

            # Validate content type
            if file.content_type:
                validate_content_type(file.content_type, ["audio/", "application/octet-stream"])

            audio_data = await file.read()

            # Additional size check after reading
            if len(audio_data) > config.server.max_upload_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File size exceeds maximum allowed size ({config.server.max_upload_size_mb}MB)",
                )

            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        if audio_base64:
            result = await orchestrator.process_turn(
                session_id=session_id,
                audio_base64=audio_base64,
                language=language,
                voice_id=voice_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return result

        raise HTTPException(
            status_code=400,
            detail="Either 'message' (text) or 'file'/'audio_base64' (audio) required",
        )

    except HTTPException:
        raise
    except Exception as e:
        safe_log_error("Conversation processing failed", e, request=http_request)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/conversation/session/create")
async def create_session(
    user_id: str = Form(...),
    scenario_id: str | None = Form(None),
    conversation_id: str | None = Form(None),
):
    """Create session"""
    try:
        session = await get_module("session")
        return await session.create_session(
            user_id=user_id, scenario_id=scenario_id, conversation_id=conversation_id
        )
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
    conversation_id: str = Form(...), role: str = Form(...), content: str = Form(...)
):
    """Save message"""
    try:
        store = await get_module("conversation_store")
        return await store.save_message(conversation_id=conversation_id, role=role, content=content)
    except Exception as e:
        logger.error(f"Failed to save message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/store/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    limit: int = DEFAULT_PAGE_SIZE,
    skip: int = 0,
    max_limit: int = MAX_PAGE_SIZE,
):
    """Get conversation messages with pagination"""
    try:
        limit = min(limit, max_limit)
        store = await get_module("conversation_store")
        if hasattr(store, "get_messages"):
            return await store.get_messages(
                conversation_id=conversation_id, limit=limit, offset=skip
            )
        else:
            raise HTTPException(status_code=501, detail="Messages endpoint not implemented")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/store/user/{user_id}/conversations")
async def list_user_conversations(
    user_id: str, limit: int = DEFAULT_PAGE_SIZE, skip: int = 0, max_limit: int = MAX_PAGE_SIZE
):
    """List user conversations with pagination"""
    try:
        limit = min(limit, max_limit)
        store = await get_module("conversation_store")
        if hasattr(store, "list_user_conversations"):
            return await store.list_user_conversations(user_id=user_id, limit=limit, offset=skip)
        else:
            raise HTTPException(
                status_code=501, detail="List conversations endpoint not implemented"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/store/{conversation_id}/context")
async def get_context(
    conversation_id: str,
    limit: int = DEFAULT_PAGE_SIZE,
    skip: int = 0,
    max_limit: int = MAX_PAGE_SIZE,
):
    """Get conversation context with pagination"""
    try:
        # Enforce max limit
        limit = min(limit, max_limit)

        store = await get_module("conversation_store")
        return await store.get_context(conversation_id=conversation_id, limit=limit, offset=skip)
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Tutoring (Diagnostic, Pedagogical Policy, Learning Path, Student Model)
# ============================================================================


class AnalyzeTurnRequest(BaseModel):
    user_text: str
    ai_text: str | None = None
    valid_skills: list[str] | None = None


@router.post("/tutoring/diagnostic/analyze")
@apply_rate_limit(TUTORING_RATE_LIMIT)
async def analyze_turn(request: AnalyzeTurnRequest):
    """Analyze conversation turn"""
    try:
        diagnostic = await get_module("diagnostic_module")
        # Check if module is disabled
        if hasattr(diagnostic, "disabled") and diagnostic.disabled:
            raise HTTPException(
                status_code=503,
                detail="Tutoring modules are disabled. Set ENABLE_TUTORING_MODULES=true to enable.",
            )
        return await diagnostic.analyze_turn(
            user_text=request.user_text, ai_text=request.ai_text, valid_skills=request.valid_skills
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tutoring/policy/compose")
@apply_rate_limit(TUTORING_RATE_LIMIT)
async def compose_prompt(context: dict):
    """Compose pedagogical prompt"""
    try:
        policy = await get_module("pedagogical_policy")
        # Check if module is disabled
        if hasattr(policy, "disabled") and policy.disabled:
            raise HTTPException(
                status_code=503,
                detail="Tutoring modules are disabled. Set ENABLE_TUTORING_MODULES=true to enable.",
            )
        return await policy.compose_prompt(context=context)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prompt composition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tutoring/path/{user_id}/next")
@apply_rate_limit(TUTORING_RATE_LIMIT)
async def get_next_skill(user_id: str, cefr_level: str | None = None):
    """Get next skill"""
    try:
        path = await get_module("learning_path")
        # Check if module is disabled
        if hasattr(path, "disabled") and path.disabled:
            raise HTTPException(
                status_code=503,
                detail="Tutoring modules are disabled. Set ENABLE_TUTORING_MODULES=true to enable.",
            )
        return await path.get_next_skill(user_id=user_id, cefr_level=cefr_level)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get next skill: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tutoring/student/{user_id}/profile")
@apply_rate_limit(TUTORING_RATE_LIMIT)
async def get_student_profile(user_id: str):
    """Get student profile"""
    try:
        student = await get_module("student_model")
        # Check if module is disabled
        if hasattr(student, "disabled") and student.disabled:
            raise HTTPException(
                status_code=503,
                detail="Tutoring modules are disabled. Set ENABLE_TUTORING_MODULES=true to enable.",
            )
        return await student.get_profile(user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Storage (File Storage, Database)
# ============================================================================


@router.post("/storage/file/upload")
async def upload_file(
    file: UploadFile = File(...),
    tags: str | None = Form(None),
    metadata: str | None = Form(None),
):
    """Upload file"""
    try:
        file_storage = await get_module("file_storage")
        file_content = await file.read()
        tags_list = tags.split(",") if tags else None
        metadata_dict = None
        if metadata:
            import json

            metadata_dict = json.loads(metadata)
        return await file_storage.upload_file(
            file_content=file_content,
            filename=file.filename,
            tags=tags_list,
            metadata=metadata_dict,
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
