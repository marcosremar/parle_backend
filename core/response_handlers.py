#!/usr/bin/env python3
"""
Padrões de Resposta de Erro para Diferentes Canais
Sistema unificado para formatação de respostas de erro por canal
"""

import json
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

# Importar exceções e middleware
from .error_handler import UltravoxError, ErrorSeverity, ErrorCategory, ValidationError, ProcessingError
from .error_handler import error_handler, handle_errors

logger = logging.getLogger(__name__)


class ResponseChannel(Enum):
    """Canais de resposta disponíveis"""
    HTTP_REST = "http_rest"
    WEBSOCKET = "websocket"
    WEBRTC_DATACHANNEL = "webrtc_datachannel"
    WEBRTC_MEDIA = "webrtc_media"
    CONSOLE = "console"


class ResponseFormatter:
    """Formatador de respostas baseado no canal"""

    def __init__(self, include_debug: bool = False):
        """
        Args:
            include_debug: Incluir informações de debug nas respostas
        """
        self.include_debug = include_debug

    def format_error_response(self,
                             error: UltravoxError,
                             channel: ResponseChannel,
                             session_id: Optional[str] = None,
                             request_id: Optional[str] = None,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Formatar resposta de erro baseado no canal

        Args:
            error: Erro da pipeline
            channel: Canal de resposta
            session_id: ID da sessão
            request_id: ID da requisição
            context: Contexto adicional

        Returns:
            Dict com resposta formatada
        """

        # Resposta base comum
        base_response = {
            "success": False,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": {
                "code": error.error_code or "UNKNOWN_ERROR",
                "message": error.message,
                "category": error.category.value,
                "severity": error.severity.value
            }
        }

        # Adicionar IDs se disponíveis
        if session_id:
            base_response["session_id"] = session_id
        if request_id:
            base_response["request_id"] = request_id

        # Formatação específica por canal
        if channel == ResponseChannel.HTTP_REST:
            return self._format_http_response(base_response, error, context)

        elif channel == ResponseChannel.WEBSOCKET:
            return self._format_websocket_response(base_response, error, context)

        elif channel == ResponseChannel.WEBRTC_DATACHANNEL:
            return self._format_webrtc_datachannel_response(base_response, error, context)

        elif channel == ResponseChannel.WEBRTC_MEDIA:
            return self._format_webrtc_media_response(base_response, error, context)

        elif channel == ResponseChannel.CONSOLE:
            return self._format_console_response(base_response, error, context)

        else:
            # Fallback para formato genérico
            return base_response

    def _format_http_response(self,
                             base_response: Dict[str, Any],
                             error: UltravoxError,
                             context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Formatação para respostas HTTP REST"""

        response = base_response.copy()

        # Adicionar status HTTP apropriado
        response["http_status"] = self._get_http_status_code(error)

        # Adicionar detalhes específicos para HTTP
        response["error"].update({
            "details": error.details,
            "suggestions": self._get_user_suggestions(error)
        })

        # Headers HTTP recomendados
        response["headers"] = {
            "Content-Type": "application/json",
            "X-Error-Category": error.category.value,
            "X-Error-Severity": error.severity.value
        }

        # Informações para retry
        if error.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            response["retry"] = {
                "can_retry": True,
                "suggested_delay_ms": self._get_retry_delay(error),
                "max_retries": 3
            }

        # Debug info se habilitado
        if self.include_debug:
            response["debug"] = {
                "original_exception": str(error.original_exception) if error.original_exception else None,
                "traceback": error.traceback_str,
                "context": context
            }

        return response

    def _format_websocket_response(self,
                                  base_response: Dict[str, Any],
                                  error: UltravoxError,
                                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Formatação para respostas WebSocket"""

        response = base_response.copy()

        # Tipo de mensagem WebSocket
        response["type"] = "error"

        # Simplificar para WebSocket (menos overhead)
        response["error"] = {
            "code": error.error_code or "UNKNOWN_ERROR",
            "message": error.message,
            "severity": error.severity.value,
            "can_continue": error.severity != ErrorSeverity.CRITICAL
        }

        # Ações recomendadas para o cliente WebSocket
        if error.category == ErrorCategory.WEBRTC:
            response["action"] = "reconnect_webrtc"
        elif error.category == ErrorCategory.AUDIO_FORMAT:
            response["action"] = "retry_with_different_audio"
        elif error.category == ErrorCategory.NETWORK:
            response["action"] = "retry_connection"
        else:
            response["action"] = "retry_request"

        # Dados de contexto úteis para o cliente
        if context:
            response["context"] = {
                k: v for k, v in context.items()
                if k in ["session_id", "audio_duration", "channel_state"]
            }

        return response

    def _format_webrtc_datachannel_response(self,
                                           base_response: Dict[str, Any],
                                           error: UltravoxError,
                                           context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Formatação para DataChannel WebRTC"""

        # Resposta super compacta para DataChannel (limitações de MTU)
        response = {
            "type": "error",
            "code": error.error_code or "ERROR",
            "msg": error.message[:100],  # Truncar mensagem
            "severity": error.severity.value[0].upper(),  # L/M/H/C
            "ts": int(datetime.utcnow().timestamp())
        }

        # Adicionar session_id se disponível
        if base_response.get("session_id"):
            response["sid"] = base_response["session_id"][:8]  # Truncar

        # Ações específicas para DataChannel
        if error.category == ErrorCategory.AUDIO_FORMAT:
            response["action"] = "audio_retry"
        elif error.category == ErrorCategory.WEBRTC:
            response["action"] = "reconnect"
        else:
            response["action"] = "retry"

        return response

    def _format_webrtc_media_response(self,
                                     base_response: Dict[str, Any],
                                     error: UltravoxError,
                                     context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Formatação para resposta via mídia WebRTC (áudio/vídeo)"""

        # Para erros que afetam mídia, criar resposta de áudio sintético
        response = {
            "type": "audio_error_response",
            "error_code": error.error_code,
            "severity": error.severity.value,
            "audio_message": self._create_audio_error_message(error),
            "fallback_text": f"Erro: {error.message}"
        }

        return response

    def _format_console_response(self,
                                base_response: Dict[str, Any],
                                error: UltravoxError,
                                context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Formatação para console/logs"""

        # Formato verbose para debugging
        response = base_response.copy()

        response.update({
            "full_details": error.to_dict(),
            "context": context,
            "stack_trace": error.traceback_str,
            "suggestions": self._get_developer_suggestions(error)
        })

        return response

    def _get_http_status_code(self, error: UltravoxError) -> int:
        """Mapear erro para código HTTP apropriado"""

        if error.category == ErrorCategory.VALIDATION:
            return 400  # Bad Request

        elif error.category == ErrorCategory.AUDIO_FORMAT:
            return 422  # Unprocessable Entity

        elif error.category == ErrorCategory.CONFIGURATION:
            return 503  # Service Unavailable

        elif error.category == ErrorCategory.MODEL_INFERENCE:
            return 503  # Service Unavailable

        elif error.category == ErrorCategory.NETWORK:
            return 502  # Bad Gateway

        elif error.severity == ErrorSeverity.CRITICAL:
            return 503  # Service Unavailable

        else:
            return 500  # Internal Server Error

    def _get_retry_delay(self, error: UltravoxError) -> int:
        """Calcular delay recomendado para retry"""

        if error.category == ErrorCategory.NETWORK:
            return 3000  # 3 segundos para problemas de rede

        elif error.category == ErrorCategory.MODEL_INFERENCE:
            return 5000  # 5 segundos para problemas do modelo

        elif error.severity == ErrorSeverity.HIGH:
            return 10000  # 10 segundos para erros graves

        else:
            return 1000  # 1 segundo padrão

    def _get_user_suggestions(self, error: UltravoxError) -> list:
        """Obter sugestões para o usuário final"""

        suggestions = []

        if error.category == ErrorCategory.AUDIO_FORMAT:
            suggestions.extend([
                "Verifique se o áudio está em formato válido (WAV, MP3, etc.)",
                "Certifique-se de que o áudio não está corrompido",
                "Tente usar áudio com qualidade de pelo menos 16kHz"
            ])

        elif error.category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Verifique sua conexão com a internet",
                "Tente novamente em alguns segundos",
                "Se o problema persistir, entre em contato com o suporte"
            ])

        elif error.category == ErrorCategory.WEBRTC:
            suggestions.extend([
                "Recarregue a página para reestabelecer a conexão",
                "Verifique se seu navegador suporta WebRTC",
                "Permita acesso ao microfone se solicitado"
            ])

        return suggestions

    def _get_developer_suggestions(self, error: UltravoxError) -> list:
        """Obter sugestões para desenvolvedores"""

        suggestions = []

        if error.category == ErrorCategory.CONFIGURATION:
            suggestions.extend([
                "Verifique as variáveis de ambiente",
                "Confirme se todos os modelos estão carregados",
                "Verifique logs de inicialização"
            ])

        elif error.category == ErrorCategory.MODEL_INFERENCE:
            suggestions.extend([
                "Verifique memória GPU disponível",
                "Confirme compatibilidade das versões",
                "Verifique logs do modelo"
            ])

        return suggestions

    def _create_audio_error_message(self, error: UltravoxError) -> str:
        """Criar mensagem de erro em áudio (texto para TTS)"""

        if error.category == ErrorCategory.AUDIO_FORMAT:
            return "Formato de áudio inválido. Por favor, tente novamente com um áudio diferente."

        elif error.category == ErrorCategory.NETWORK:
            return "Problema de conexão. Tentando reconectar..."

        elif error.category == ErrorCategory.MODEL_INFERENCE:
            return "Serviço temporariamente indisponível. Tente novamente em alguns segundos."

        else:
            return "Ocorreu um erro. Por favor, tente novamente."


# ============================================================================
# RESPONSE HANDLERS ESPECÍFICOS
# ============================================================================

class HTTPErrorHandler:
    """Handler específico para erros HTTP"""

    def __init__(self, formatter: ResponseFormatter):
        self.formatter = formatter

    def handle_error(self,
                    error: UltravoxError,
                    session_id: Optional[str] = None,
                    request_id: Optional[str] = None) -> tuple[Dict[str, Any], int]:
        """
        Tratar erro HTTP

        Returns:
            tuple: (response_body, http_status_code)
        """

        response = self.formatter.format_error_response(
            error=error,
            channel=ResponseChannel.HTTP_REST,
            session_id=session_id,
            request_id=request_id
        )

        status_code = response.get("http_status", 500)
        return response, status_code


class WebSocketErrorHandler:
    """Handler específico para erros WebSocket"""

    def __init__(self, formatter: ResponseFormatter):
        self.formatter = formatter

    async def send_error(self,
                        websocket,
                        error: UltravoxError,
                        session_id: Optional[str] = None):
        """Enviar erro via WebSocket"""

        response = self.formatter.format_error_response(
            error=error,
            channel=ResponseChannel.WEBSOCKET,
            session_id=session_id
        )

        try:
            await websocket.send(json.dumps(response))
        except Exception as ws_error:
            logger.error(f"Falha ao enviar erro via WebSocket: {ws_error}")


class WebRTCErrorHandler:
    """Handler específico para erros WebRTC"""

    def __init__(self, formatter: ResponseFormatter):
        self.formatter = formatter

    def send_datachannel_error(self,
                              datachannel,
                              error: UltravoxError,
                              session_id: Optional[str] = None):
        """Enviar erro via DataChannel"""

        response = self.formatter.format_error_response(
            error=error,
            channel=ResponseChannel.WEBRTC_DATACHANNEL,
            session_id=session_id
        )

        try:
            # Serializar de forma compacta
            message = json.dumps(response, separators=(',', ':'))
            datachannel.send(message)
        except Exception as dc_error:
            logger.error(f"Falha ao enviar erro via DataChannel: {dc_error}")

    def create_audio_error_response(self,
                                   error: UltravoxError,
                                   session_id: Optional[str] = None) -> Dict[str, Any]:
        """Criar resposta de erro em áudio"""

        return self.formatter.format_error_response(
            error=error,
            channel=ResponseChannel.WEBRTC_MEDIA,
            session_id=session_id
        )


# ============================================================================
# FACTORY PARA CRIAÇÃO DE HANDLERS
# ============================================================================

class ErrorHandlerFactory:
    """Factory para criar handlers de erro"""

    @staticmethod
    def create_handler(channel: ResponseChannel,
                      include_debug: bool = False) -> Union[HTTPErrorHandler, WebSocketErrorHandler, WebRTCErrorHandler]:
        """Criar handler apropriado para o canal"""

        formatter = ResponseFormatter(include_debug=include_debug)

        if channel == ResponseChannel.HTTP_REST:
            return HTTPErrorHandler(formatter)

        elif channel == ResponseChannel.WEBSOCKET:
            return WebSocketErrorHandler(formatter)

        elif channel in [ResponseChannel.WEBRTC_DATACHANNEL, ResponseChannel.WEBRTC_MEDIA]:
            return WebRTCErrorHandler(formatter)

        else:
            raise ValueError(f"Canal não suportado: {channel}")


# ============================================================================
# INTEGRAÇÃO COM MIDDLEWARE
# ============================================================================

def integrate_with_middleware(middleware,
                             default_channel: ResponseChannel = ResponseChannel.HTTP_REST,
                             include_debug: bool = False):
    """Integrar handlers de resposta com middleware de erro"""

    formatter = ResponseFormatter(include_debug=include_debug)

    def response_handler(error: UltravoxError, context: Optional[Dict[str, Any]]):
        """Handler que será registrado no middleware"""

        # Determinar canal baseado no contexto
        channel = context.get("response_channel", default_channel) if context else default_channel

        # Formatar resposta
        response = formatter.format_error_response(
            error=error,
            channel=channel,
            session_id=context.get("session_id") if context else None,
            context=context
        )

        # Log da resposta formatada
        logger.info(f"📤 Resposta de erro formatada para {channel.value}: {response.get('error', {}).get('code')}")

        # Armazenar resposta no contexto para uso posterior
        if context:
            context["formatted_response"] = response

    # Registrar handler para todas as categorias
    for category in ErrorCategory:
        middleware.register_error_handler(category, response_handler)

    logger.info("🔗 Handlers de resposta integrados com middleware")


# ============================================================================
# EXEMPLO DE USO COMPLETO
# ============================================================================

def create_complete_error_system(debug_mode: bool = False) -> tuple[Any, Dict[str, Any]]:
    """Criar sistema completo de tratamento de erros"""

    # Criar middleware
    from .error_handler import error_handler, handle_errors
    middleware = error_handler

    # Criar handlers para cada canal
    handlers = {
        "http": ErrorHandlerFactory.create_handler(ResponseChannel.HTTP_REST, debug_mode),
        "websocket": ErrorHandlerFactory.create_handler(ResponseChannel.WEBSOCKET, debug_mode),
        "webrtc": ErrorHandlerFactory.create_handler(ResponseChannel.WEBRTC_DATACHANNEL, debug_mode)
    }

    # Integrar com middleware
    integrate_with_middleware(middleware, include_debug=debug_mode)

    logger.info("🎯 Sistema completo de tratamento de erros criado")

    return middleware, handlers