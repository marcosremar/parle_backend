"""
Modelos Pydantic para validação de respostas do LLM
Suporta streaming parcial e validação incremental
Suporta análise adaptativa e direcionar conversação
"""

from typing import List, Optional, Union, Any, Literal
from pydantic import ValidationError, BaseModel, Field, validator
from enum import Enum
import json
import time
from src.core.exceptions import UltravoxError, wrap_exception


class ResponseStatus(str, Enum):
    """Status da resposta para streaming"""
    PARTIAL = "partial"
    COMPLETE = "complete"
    ERROR = "error"


class StreamingChunk(BaseModel):
    """Chunk individual para streaming"""
    type: str = Field(..., description="Tipo do chunk: text, audio, suggestions")
    data: Any = Field(..., description="Dados do chunk")
    status: ResponseStatus = Field(default=ResponseStatus.PARTIAL)
    sequence: int = Field(..., description="Número de sequência do chunk")
    timestamp: float = Field(..., description="Timestamp do chunk")


class ConversationSuggestion(BaseModel):
    """Modelo para sugestões de conversação"""
    text: str = Field(..., min_length=1, max_length=100)
    confidence: float = Field(default=1.0, ge=0, le=1)
    category: Optional[str] = Field(None, description="Categoria da sugestão")

    @validator('text')
    def clean_text(cls, v):
        """Remove marcadores e limpa o texto"""
        # Remove números, bullets, etc
        import re
        v = re.sub(r'^[\d\-\.\*\•]+\s*', '', v.strip())
        return v.strip()


class LLMResponse(BaseModel):
    """Resposta principal do LLM com validação"""
    text: str = Field(..., min_length=1, description="Resposta textual")
    confidence: float = Field(default=1.0, ge=0, le=1)
    language: str = Field(default="pt-BR")
    tokens_used: Optional[int] = None

    @validator('text')
    def validate_response(cls, v):
        """Valida que a resposta não é um erro"""
        error_patterns = [
            "erro no llm",
            "falha ao processar",
            "desculpe, estou com dificuldades"
        ]
        if any(pattern in v.lower() for pattern in error_patterns):
            raise ValueError(f"Resposta indica erro: {v}")
        return v


class PipelineResponse(BaseModel):
    """Resposta completa da pipeline com todos os componentes"""

    # Componente principal
    response: LLMResponse = Field(..., description="Resposta principal do LLM")

    # Componentes opcionais (para streaming)
    audio: Optional[bytes] = Field(None, description="Áudio sintetizado")
    suggestions: Optional[List[ConversationSuggestion]] = Field(
        None,
        description="Sugestões de continuação",
        max_items=5
    )

    # Metadados
    session_id: str = Field(..., min_length=1)
    processing_time_ms: float = Field(..., ge=0)
    status: ResponseStatus = Field(default=ResponseStatus.COMPLETE)

    # Timing breakdown para análise
    timing: Optional[dict] = None

    class Config:
        """Configuração do modelo"""
        json_encoders = {
            bytes: lambda v: f"<audio:{len(v)} bytes>"  # Não serializa áudio completo
        }


class StreamingResponse(BaseModel):
    """Container para resposta em streaming com múltiplas partes"""

    session_id: str
    chunks: List[StreamingChunk] = Field(default_factory=list)
    current_text: str = ""
    current_suggestions: List[ConversationSuggestion] = Field(default_factory=list)
    is_complete: bool = False

    def add_text_chunk(self, text: str) -> StreamingChunk:
        """Adiciona chunk de texto"""
        chunk = StreamingChunk(
            type="text",
            data=text,
            sequence=len(self.chunks),
            timestamp=self._get_timestamp(),
            status=ResponseStatus.PARTIAL
        )
        self.chunks.append(chunk)
        self.current_text += text
        return chunk

    def add_suggestion(self, suggestion: ConversationSuggestion) -> StreamingChunk:
        """Adiciona uma sugestão"""
        chunk = StreamingChunk(
            type="suggestion",
            data=suggestion.dict(),
            sequence=len(self.chunks),
            timestamp=self._get_timestamp(),
            status=ResponseStatus.PARTIAL
        )
        self.chunks.append(chunk)
        self.current_suggestions.append(suggestion)
        return chunk

    def finalize(self) -> PipelineResponse:
        """Finaliza e retorna resposta completa validada"""
        self.is_complete = True

        return PipelineResponse(
            response=LLMResponse(text=self.current_text),
            suggestions=self.current_suggestions if self.current_suggestions else None,
            session_id=self.session_id,
            processing_time_ms=self._calculate_total_time(),
            status=ResponseStatus.COMPLETE
        )

    def _get_timestamp(self) -> float:
        """Retorna timestamp atual"""
        import time
        return time.time()

    def _calculate_total_time(self) -> float:
        """Calcula tempo total de processamento"""
        if not self.chunks:
            return 0
        return (self.chunks[-1].timestamp - self.chunks[0].timestamp) * 1000


class SuggestionRequest(BaseModel):
    """Request para geração de sugestões com validação"""
    user_message: str = Field(..., min_length=1, max_length=500)
    assistant_response: str = Field(..., min_length=1)
    context: Optional[str] = Field(None, max_length=2000)
    num_suggestions: int = Field(default=3, ge=1, le=5)

    def to_prompt(self) -> str:
        """Gera prompt estruturado para o LLM"""
        return f"""Baseado na conversa:
Usuário: {self.user_message}
Assistente: {self.assistant_response}

Gere exatamente {self.num_suggestions} perguntas ou frases curtas que o usuário poderia usar para continuar a conversa.

IMPORTANTE:
- Cada sugestão em uma linha separada
- Máximo 100 caracteres por sugestão
- Sem numeração ou marcadores
- Em português brasileiro
- Perguntas naturais e relevantes ao contexto

Sugestões:"""


class ErrorResponse(BaseModel):
    """Resposta de erro estruturada"""
    error: str = Field(..., description="Mensagem de erro")
    error_code: str = Field(..., description="Código do erro")
    component: str = Field(..., description="Componente que gerou o erro")
    fallback_used: bool = Field(default=False)
    details: Optional[dict] = None

    def to_user_message(self) -> str:
        """Retorna mensagem amigável para o usuário"""
        user_messages = {
            "LLM_TIMEOUT": "O sistema está demorando mais que o esperado. Tente novamente.",
            "STT_FAILED": "Não consegui entender o áudio. Pode repetir?",
            "TTS_FAILED": "Houve um problema ao gerar o áudio.",
            "SUGGESTION_FAILED": "Não foi possível gerar sugestões no momento.",
            "VALIDATION_ERROR": "A resposta não pôde ser processada corretamente."
        }
        return user_messages.get(self.error_code, "Desculpe, ocorreu um erro inesperado.")


# ============================================================================
# MVP Modelos para Streaming JSON com Análise Adaptativa
# ============================================================================


class ConversationAnalysis(BaseModel):
    """
    Análise pós-resposta para direcionar conversação futura.
    Emitido quando a resposta do LLM está completa.
    """

    response_type: str = Field(
        ...,
        description="Tipo de resposta: question, explanation, suggestion, apology, confirmation"
    )
    theme: str = Field(..., description="Tema principal detectado")
    tone: str = Field(
        default="helpful",
        description="Tom da resposta: helpful, formal, casual, urgent, empathetic"
    )
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Confiança na análise (0-1)"
    )

    @validator('response_type')
    def validate_response_type(cls, v):
        """Valida tipos de resposta conhecidos"""
        valid_types = ['question', 'explanation', 'suggestion', 'apology', 'confirmation', 'other']
        if v not in valid_types:
            raise ValueError(f"response_type deve ser um de: {valid_types}")
        return v

    @validator('tone')
    def validate_tone(cls, v):
        """Valida tons conhecidos"""
        valid_tones = ['helpful', 'formal', 'casual', 'urgent', 'empathetic', 'professional']
        if v not in valid_tones:
            raise ValueError(f"tone deve ser um de: {valid_tones}")
        return v


class AdaptiveInstructions(BaseModel):
    """
    Instruções adaptativas para direcionar próxima resposta do LLM.
    Geradas com base em análise da resposta anterior.
    """

    next_prompt_prefix: str = Field(
        ...,
        description="Prefixo a ser injetado no próximo prompt do LLM"
    )
    tone_adjustment: str = Field(
        default="maintain_current",
        description="Ajuste de tom: maintain_current, become_more_formal, become_casual"
    )
    verbosity: str = Field(
        default="medium",
        description="Nível de verbosidade: concise, medium, verbose"
    )
    expected_next_topics: List[str] = Field(
        default_factory=list,
        description="Tópicos esperados na próxima mensagem do usuário"
    )
    estimated_turns_remaining: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Estimativa de turnos até resolver o tópico"
    )

    @validator('tone_adjustment')
    def validate_tone_adjustment(cls, v):
        """Valida ajustes de tom conhecidos"""
        valid_adjustments = ['maintain_current', 'become_more_formal', 'become_casual']
        if v not in valid_adjustments:
            raise ValueError(f"tone_adjustment deve ser um de: {valid_adjustments}")
        return v

    @validator('verbosity')
    def validate_verbosity(cls, v):
        """Valida níveis de verbosidade"""
        valid_levels = ['concise', 'medium', 'verbose']
        if v not in valid_levels:
            raise ValueError(f"verbosity deve ser um de: {valid_levels}")
        return v


class ErrorCorrection(BaseModel):
    """
    Detecção e sugestão de correções para erros do usuário.
    Antecipa problemas de compreensão ou expectativa.
    """

    pattern_detected: Optional[str] = Field(
        None,
        description="Padrão de erro detectado (ex: location_format_error)"
    )
    suggested_clarification: Optional[str] = Field(
        None,
        description="Clarificação sugerida para oferecer ao usuário"
    )
    correction_needed: bool = Field(
        default=False,
        description="Se uma correção é necessária"
    )
    correction_urgency: str = Field(
        default="low",
        description="Urgência da correção: low, medium, high"
    )

    @validator('correction_urgency')
    def validate_urgency(cls, v):
        """Valida níveis de urgência"""
        valid_levels = ['low', 'medium', 'high']
        if v not in valid_levels:
            raise ValueError(f"correction_urgency deve ser um de: {valid_levels}")
        return v


class StreamingJSONEvent(BaseModel):
    """
    Evento JSON para streaming com SSE (Server-Sent Events).
    Cada evento representa um tipo diferente de informação durante o processamento.
    """

    event: Literal[
        "text_chunk",
        "analysis",
        "adaptive_instructions",
        "error_correction",
        "complete",
        "error"
    ] = Field(..., description="Tipo de evento sendo transmitido")
    sequence: int = Field(..., ge=0, description="Número de sequência do evento")
    data: Union[str, dict, Any] = Field(..., description="Dados do evento")
    timestamp: float = Field(default_factory=time.time, description="Timestamp do evento")
    is_final: bool = Field(default=False, description="Se este é o último evento")

    class Config:
        """Configuração do modelo"""
        json_encoders = {
            bytes: lambda v: f"<bytes:{len(v)}>"
        }


# Funções auxiliares para parsing de respostas do LLM

def parse_suggestions_from_llm(text: str, max_suggestions: int = 3) -> List[ConversationSuggestion]:
    """
    Parse sugestões do texto retornado pelo LLM
    Robusto contra diferentes formatos
    """
    suggestions = []
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Tenta criar sugestão validada
        try:
            suggestion = ConversationSuggestion(text=line)
            suggestions.append(suggestion)

            if len(suggestions) >= max_suggestions:
                break
        except (ValueError, TypeError, AttributeError):
            # Ignora linhas inválidas (failed validation)
            continue

    # Fallback se não conseguiu parsear nada
    if not suggestions:
        defaults = [
            ConversationSuggestion(text="Me conte mais sobre isso"),
            ConversationSuggestion(text="Pode dar um exemplo?"),
            ConversationSuggestion(text="O que mais você sabe?")
        ]
        return defaults[:max_suggestions]

    return suggestions


def validate_llm_response(response: str) -> Union[LLMResponse, ErrorResponse]:
    """
    Valida resposta do LLM e retorna modelo apropriado
    """
    try:
        # Tenta criar resposta validada
        return LLMResponse(text=response)
    except ValueError as e:
        # Retorna erro estruturado
        return ErrorResponse(
            error=str(e),
            error_code="VALIDATION_ERROR",
            component="llm",
            details={"original_response": response}
        )


# Exemplo de uso com streaming

def create_streaming_response(session_id: str) -> StreamingResponse:
    """Cria nova resposta para streaming"""
    return StreamingResponse(session_id=session_id)


def process_streaming_chunk(response: StreamingResponse, chunk_type: str, data: Any):
    """Processa chunk de streaming com validação"""

    if chunk_type == "text":
        response.add_text_chunk(data)

    elif chunk_type == "suggestion":
        try:
            suggestion = ConversationSuggestion(text=data)
            response.add_suggestion(suggestion)
        except Exception as e:
            # Log erro mas não interrompe streaming
            print(f"Erro ao validar sugestão: {e}")

    elif chunk_type == "finalize":
        return response.finalize()

    return None