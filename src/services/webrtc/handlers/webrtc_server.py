"""
WebRTC Server - Python puro com aiortc
Ultra-baixa latência com DataChannel UDP-like
Servidor age como peer direto (não relay)
"""

import asyncio
import json
import logging
import base64
from typing import Dict, Optional, Any
import numpy as np
from datetime import datetime

from aiohttp import web, WSMsgType
from aiohttp.web_fileresponse import FileResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer
from aiortc.contrib.signaling import object_to_string, object_from_string
import aiohttp_cors
import os
from pathlib import Path
import sys

# ✅ Phase 4a: Removed deprecated sys.path manipulation
# DEPRECATED: modules.memory removed (replaced by conversation_store service)

# Importar validador de áudio da pipeline (se disponível)
try:
    # ✅ Phase 4a: Use proper absolute imports (no sys.path manipulation)
    from src.core.audio_pipeline_validator import AudioPipelineValidator
    audio_validator_available = True
    logger.info("✅ Validador de áudio da pipeline disponível")
except ImportError:
    audio_validator_available = False
    logger.warning("⚠️ Validador de áudio da pipeline não disponível")

logger = logging.getLogger(__name__)


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """
    Convert PCM audio data to WAV format with proper headers

    Args:
        pcm_data: Raw PCM audio data (float32 or int16)
        sample_rate: Sample rate in Hz (default 24000)
        channels: Number of audio channels (default 1 for mono)
        bits_per_sample: Bits per sample (default 16)

    Returns:
        WAV format audio data with headers
    """
    import struct
    import io

    # Convert float32 PCM to int16 if needed
    if len(pcm_data) % 4 == 0:  # Likely float32 data
        # Convert from float32 to int16
        audio_array = np.frombuffer(pcm_data, dtype=np.float32)
        # Clamp values to [-1, 1] and convert to int16
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        pcm_data = audio_int16.tobytes()
        bits_per_sample = 16

    # Calculate WAV header values
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_data)
    file_size = 36 + data_size

    # Create WAV header
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF',           # ChunkID
        file_size,         # ChunkSize
        b'WAVE',           # Format
        b'fmt ',           # Subchunk1ID
        16,                # Subchunk1Size (PCM = 16)
        1,                 # AudioFormat (PCM = 1)
        channels,          # NumChannels
        sample_rate,       # SampleRate
        byte_rate,         # ByteRate
        block_align,       # BlockAlign
        bits_per_sample,   # BitsPerSample
        b'data',           # Subchunk2ID
        data_size          # Subchunk2Size
    )

    # Combine header and data
    return wav_header + pcm_data


def create_wav_header(data_size: int, sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """
    Create a WAV file header for given audio parameters

    Args:
        data_size: Size of audio data in bytes
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        bits_per_sample: Bits per sample

    Returns:
        WAV header as bytes
    """
    import struct

    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    file_size = 36 + data_size

    return struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF',           # ChunkID
        file_size,         # ChunkSize
        b'WAVE',           # Format
        b'fmt ',           # Subchunk1ID
        16,                # Subchunk1Size (PCM = 16)
        1,                 # AudioFormat (PCM = 1)
        channels,          # NumChannels
        sample_rate,       # SampleRate
        byte_rate,         # ByteRate
        block_align,       # BlockAlign
        bits_per_sample,   # BitsPerSample
        b'data',           # Subchunk2ID
        data_size          # Subchunk2Size
    )


class WebRTCServer:
    """
    Servidor WebRTC em Python puro
    Ultra-baixa latência com DataChannel não-ordenado
    """
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 8088,
                 ice_servers: list = None):
        """
        Inicializar servidor WebRTC
        
        Args:
            host: Host para bind
            port: Porta WebSocket
            ice_servers: Servidores STUN/TURN
        """
        self.host = host
        self.port = port
        if ice_servers is None:
            self.ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        else:
            self.ice_servers = [RTCIceServer(urls=server["urls"]) for server in ice_servers]
        
        # Conexões ativas
        self.peers: Dict[str, RTCPeerConnection] = {}
        self.data_channels: Dict[str, RTCDataChannel] = {}
        self.connections: Dict[str, dict] = {}  # Armazenar dados das sessões WebSocket
        
        # Módulos
        self.audio_processor = None
        self.tts_module = None
        self.dev_metrics = None

        # Validador de áudio da pipeline (se disponível e em desenvolvimento)
        self.audio_validator = None
        if audio_validator_available and os.getenv('ENVIRONMENT') == 'development':
            try:
                self.audio_validator = AudioPipelineValidator(development_mode=True)
                logger.info("✅ Validador de áudio da pipeline inicializado em modo desenvolvimento")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao inicializar validador de áudio: {e}")
        
        # Sistema de memória para manter contexto
        self.memory_store = SimpleMemoryStore(
            max_sessions=100,
            max_messages_per_session=20
        )
        
        # Caminho para arquivos estáticos do frontend
        self.frontend_build_path = Path(__file__).parent.parent.parent / "frontend" / "build"
        
        # App aiohttp
        self.app = web.Application()
        self.setup_routes()
        self.setup_cors()
        
        # Estatísticas
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_messages": 0,
            "avg_latency": 0,
            "min_latency": 999999,
            "max_latency": 0
        }

        # Conexões WebSocket ativas para envio de métricas
        self.websocket_connections = set()
        
    def set_audio_processor(self, processor) -> Any:
        """Definir processador de áudio (Ultravox)"""
        self.audio_processor = processor
        logger.info("✅ Processador de áudio configurado")
        
    def set_tts_module(self, tts) -> Any:
        """Definir módulo TTS"""
        self.tts_module = tts
        logger.info("✅ Módulo TTS configurado")

    def set_dev_metrics(self, dev_metrics) -> Any:
        """Definir módulo de métricas de desenvolvimento"""
        self.dev_metrics = dev_metrics
        logger.info("✅ Métricas de desenvolvimento configuradas")

    def get_voice_info(self, voice_id) -> Any:
        """Obter informações sobre personagem/voz"""
        # Mapeamento básico de vozes para personagens
        voice_map = {
            # American English voices
            'af_bella': {'name': 'Bella', 'language': 'English', 'personality': 'warm and friendly'},
            'af_alloy': {'name': 'Alloy', 'language': 'English', 'personality': 'professional and clear'},
            'af_nova': {'name': 'Nova', 'language': 'English', 'personality': 'energetic and enthusiastic'},
            'am_adam': {'name': 'Adam', 'language': 'English', 'personality': 'calm and professional'},
            'am_liam': {'name': 'Liam', 'language': 'English', 'personality': 'friendly and approachable'},

            # Portuguese voices
            'pm_alex': {'name': 'Alex', 'language': 'Portuguese', 'personality': 'profissional e claro'},

            # Spanish voices
            'ef_dora': {'name': 'Dora', 'language': 'Spanish', 'personality': 'amigable y expresiva'},
            'em_alex': {'name': 'Alex', 'language': 'Spanish', 'personality': 'profesional y claro'},

            # Italian voices
            'if_sara': {'name': 'Sara', 'language': 'Italian', 'personality': 'espressiva e appassionata'},
            'im_nicola': {'name': 'Nicola', 'language': 'Italian', 'personality': 'professionale e chiaro'},

            # Chinese voices
            'zf_xiaobei': {'name': 'Xiaobei', 'language': 'Chinese', 'personality': '友好亲切'},
            'zm_yunjian': {'name': 'Yunjian', 'language': 'Chinese', 'personality': '专业清晰'},
        }

        # Retornar informações da voz ou um padrão
        return voice_map.get(voice_id, {
            'name': 'Assistant',
            'language': 'English',
            'personality': 'helpful and professional'
        })

    async def send_to_all_clients(self, message: str) -> Any:
        """Enviar mensagem para todos os clientes conectados via WebSocket"""
        if not self.websocket_connections:
            return

        dead_connections = set()
        for ws in self.websocket_connections.copy():
            try:
                await ws.send_str(message)
            except Exception as e:
                logger.debug(f"Conexão WebSocket morta removida: {e}")
                dead_connections.add(ws)

        # Remover conexões mortas
        self.websocket_connections -= dead_connections
        
    def setup_routes(self) -> Any:
        """Configurar rotas HTTP/WebSocket"""
        self.app.router.add_post("/offer", self.handle_offer)
        self.app.router.add_get("/stats", self.get_stats)
        self.app.router.add_get("/health", self.health_check)
        self.app.router.add_get("/ws", self.handle_websocket)

        # Add audio converter endpoint
        from src.services.webrtc.handlers.audio_converter_endpoint import setup_audio_converter_routes
        setup_audio_converter_routes(self.app)

        # Servir arquivos estáticos do frontend React
        self.app.router.add_static("/static", self.frontend_build_path / "static")
        self.app.router.add_get("/", self.serve_frontend)
        self.app.router.add_get("/{path:.*}", self.serve_frontend)  # Catch-all for React Router
    
    def setup_cors(self) -> Any:
        """Configurar CORS para permitir conexões cross-origin"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Adicionar CORS a todas as rotas
        for route in list(self.app.router.routes()):
            if not isinstance(route.resource, web.StaticResource):
                cors.add(route)
        
    async def handle_offer(self, request) -> Any:
        """
        Lidar com oferta WebRTC do cliente
        Servidor age como peer respondente
        """
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        
        # Gerar ID de sessão
        session_id = f"peer_{datetime.now().timestamp()}_{id(request)}"
        
        # Criar peer connection
        config = RTCConfiguration(iceServers=self.ice_servers)
        pc = RTCPeerConnection(configuration=config)
        
        self.peers[session_id] = pc
        self.stats["total_connections"] += 1
        self.stats["active_connections"] = len(self.peers)
        
        logger.info(f"🔌 Nova conexão: {session_id}")
        
        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> Any:
            """DataChannel criado pelo cliente"""
            logger.info(f"✅ DataChannel aberto: {channel.label}")
            self.data_channels[session_id] = channel
            
            @channel.on("message")
            async def on_message(message) -> Any:
                """Processar mensagem do DataChannel"""
                await self.handle_data_channel_message(
                    session_id, message, channel
                )
                
        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> Any:
            """Monitorar estado da conexão"""
            logger.info(f"📡 Estado {session_id}: {pc.connectionState}")
            
            if pc.connectionState in ["failed", "closed"]:
                await self.cleanup_peer(session_id)
                
        # Definir descrição remota (oferta)
        await pc.setRemoteDescription(offer)
        
        # Criar resposta
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "session_id": session_id
            })
        )
        
    async def handle_data_channel_message(self,
                                         session_id: str,
                                         message: Any,
                                         channel: RTCDataChannel):
        """
        Processar mensagem recebida via DataChannel
        Ultra-baixa latência com processamento direto
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Parse mensagem
            if isinstance(message, str):
                data = json.loads(message)

                # Verificar se é mensagem de configuração
                if data.get('type') == 'config':
                    await self.handle_config_message(data, session_id, channel)
                    return
                elif data.get('type') == 'voice_change':
                    await self.handle_voice_change(data, session_id, channel)
                    return
                elif data.get('type') == 'audio_binary_header':
                    # 🚀 OTIMIZAÇÃO WebRTC: Header de áudio binário - esperar dados binários
                    logger.info(f"🚀 === OTIMIZAÇÃO WEBRTC BINÁRIA ATIVADA ===")
                    logger.info(f"   📦 Header recebido: {data.get('samples')} samples, {data.get('bytes')} bytes")
                    logger.info(f"   🔊 Taxa: {data.get('sampleRate')}Hz, Formato: {data.get('format')}")
                    logger.info(f"   ⏰ Timestamp: {data.get('timestamp')}")
                    logger.info(f"   📱 Recorder: {data.get('recorder')}")

                    # Armazenar metadados no buffer da sessão
                    if session_id not in self.connections:
                        self.connections[session_id] = {'audio_buffer': {}}

                    self.connections[session_id]['audio_buffer'] = {
                        'waiting_for_binary': True,
                        'sample_rate': data.get('sampleRate', 16000),
                        'format': data.get('format', 'pcm16'),
                        'expected_samples': data.get('samples', 0),
                        'expected_bytes': data.get('bytes', 0),
                        'timestamp': data.get('timestamp'),
                        'recorder': data.get('recorder', 'unknown')
                    }

                    logger.info(f"   ✅ Aguardando {data.get('bytes')} bytes de áudio binário via WebRTC...")
                    return

                # Caso contrário, processar como áudio JSON (backward compatibility)
                audio_data = np.array(data.get("audio", []), dtype=np.float32)
                logger.info(f"📦 Áudio JSON recebido: {len(audio_data)} samples")

            elif isinstance(message, bytes):
                # Verificar se estamos aguardando dados binários após header
                audio_buffer = self.connections.get(session_id, {}).get('audio_buffer', {})
                if audio_buffer.get('waiting_for_binary'):
                    # 🚀 OTIMIZAÇÃO WebRTC: Processar áudio binário com metadados do header
                    logger.info(f"🚀 === ÁUDIO WEBRTC BINÁRIO RECEBIDO ===")
                    logger.info(f"   📊 Dados brutos: {len(message)} bytes")
                    logger.info(f"   📦 Esperados: {audio_buffer.get('expected_bytes', 0)} bytes")
                    logger.info(f"   🎤 Samples esperados: {audio_buffer.get('expected_samples', 0)}")
                    logger.info(f"   🔊 Taxa: {audio_buffer.get('sample_rate', 16000)}Hz")
                    logger.info(f"   📱 Recorder: {audio_buffer.get('recorder', 'unknown')}")

                    # Validar se recebemos o tamanho correto
                    expected_bytes = audio_buffer.get('expected_bytes', 0)
                    if len(message) != expected_bytes:
                        logger.warning(f"⚠️ TAMANHO INCONSISTENTE: Esperado {expected_bytes}, recebido {len(message)}")

                    # Converter dados binários diretamente para array de áudio
                    # Os dados já estão em int16 como enviado pelo frontend
                    audio_int16 = np.frombuffer(message, dtype=np.int16)
                    audio_data = audio_int16.astype(np.float32) / 32768.0

                    logger.info(f"   🎧 Áudio processado: {len(audio_data)} samples")
                    logger.info(f"   ⏱️  Duração: {len(audio_data) / audio_buffer.get('sample_rate', 16000):.2f}s")
                    logger.info(f"   ✅ WEBRTC BINÁRIO - Ultra performance!")

                    # Limpar buffer após processar
                    self.connections[session_id]['audio_buffer'] = {}
                else:
                    # Áudio binário direto (sem header) - legacy
                    audio_data = np.frombuffer(message, dtype=np.float32)
                    logger.info(f"📦 Áudio binário direto: {len(audio_data)} samples")
            else:
                audio_data = np.array(message, dtype=np.float32)
                logger.info(f"📦 Áudio array direto: {len(audio_data)} samples")
                
            logger.info(f"🎤 Áudio recebido: {len(audio_data)} samples")
            
            # Processar com Ultravox
            response_text = ""
            response_audio = None
            
            if self.audio_processor:
                # Obter contexto da sessão
                context_messages = await self.memory_store.get_context(session_id, max_messages=10)
                context = ""
                
                if context_messages:
                    # Formatar contexto para o Ultravox
                    for msg in context_messages[-6:]:  # Últimas 6 mensagens
                        if msg['role'] == 'user':
                            context += f"User: {msg['content']}\n"
                        elif msg['role'] == 'assistant':
                            context += f"Assistant: {msg['content']}\n\n"
                    logger.info(f"🧠 Usando contexto de {len(context_messages)} mensagens")
                
                # Obter voice_id da sessão
                voice_id = self.connections.get(session_id, {}).get('voice_id', 'af_bella')

                # Processar áudio com contexto e voice_id
                response_text = await self.audio_processor.process_audio(
                    audio_data,
                    session_id,
                    context=context if context else None,
                    voice_id=voice_id
                )
                
                # Salvar interação na memória
                await self.memory_store.add_message(
                    session_id=session_id,
                    role="user",
                    content="[áudio processado]"
                )
                await self.memory_store.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=response_text
                )
                
                # Gerar áudio de resposta com TTS
                if self.tts_module and response_text:
                    response_audio = await self.tts_module.synthesize(response_text)
                    
                    # TTS retorna bytes de float32, precisamos converter para int16
                    if isinstance(response_audio, bytes):
                        # Converter bytes float32 para numpy array
                        audio_float32 = np.frombuffer(response_audio, dtype=np.float32)
                        # Converter para int16
                        response_audio = (audio_float32 * 32767).astype(np.int16).tobytes()
                        logger.info(f"🎵 Áudio TTS gerado: {len(response_audio)} bytes de int16")
                    elif isinstance(response_audio, np.ndarray):
                        # Se já for numpy array, converter diretamente
                        response_audio = (response_audio * 32767).astype(np.int16).tobytes()
                        logger.info(f"🎵 Áudio TTS gerado: {len(response_audio)} bytes")
                    
                    # Converter bytes para base64 para enviar via JSON
                    import base64
                    response_audio_b64 = base64.b64encode(response_audio).decode('utf-8') if response_audio else None
                else:
                    response_audio_b64 = None
            else:
                response_text = "Processador de áudio não configurado"
                response_audio_b64 = None
                
            # Calcular latência
            latency = (asyncio.get_event_loop().time() - start_time) * 1000
            self.update_stats(latency)
            
            # Enviar resposta via DataChannel (ultra-rápido!)
            response = {
                "type": "response",
                "text": response_text,
                "audio": response_audio_b64,  # Enviando como base64
                "latency": latency
            }
            
            # Enviar como JSON ou binário
            if channel.readyState == "open":
                channel.send(json.dumps(response))
                logger.info(f"⚡ Resposta enviada em {latency:.1f}ms")
                
            self.stats["total_messages"] += 1
            
        except Exception as e:
            logger.error(f"❌ Erro processando áudio: {e}")
            if channel.readyState == "open":
                channel.send(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
                
    async def cleanup_peer(self, session_id: str) -> Any:
        """Limpar peer desconectado"""
        if session_id in self.peers:
            pc = self.peers[session_id]
            await pc.close()
            del self.peers[session_id]
            
        if session_id in self.data_channels:
            del self.data_channels[session_id]
            
        self.stats["active_connections"] = len(self.peers)
        logger.info(f"🧹 Peer removido: {session_id}")
        
    async def handle_config_message(self, data: dict, session_id: str, channel: RTCDataChannel) -> Any:
        """Processar mensagem de configuração"""
        logger.info(f"⚙️ Configuração recebida para {session_id}: {data}")

        # Processar voice_id se estiver presente (aceitar tanto 'voice' quanto 'voice_id')
        voice_id = data.get('voice_id') or data.get('voice')
        if voice_id and self.tts_module:
            if hasattr(self.tts_module, 'set_voice'):
                self.tts_module.set_voice(voice_id)
                logger.info(f"🔊 Voz alterada para: {voice_id}")

        # Enviar confirmação
        response = {
            "type": "config_ack",
            "message": "✅ Configuração aplicada",
            "voice_id": voice_id if voice_id else None
        }

        if channel.readyState == "open":
            channel.send(json.dumps(response))

    async def handle_voice_change(self, data: dict, session_id: str, channel: RTCDataChannel) -> Any:
        """Processar mudança de voz"""
        voice_id = data.get('voice_id')
        logger.info(f"🔊 Solicitação de mudança de voz para {session_id}: {voice_id}")

        if voice_id and self.tts_module:
            if hasattr(self.tts_module, 'set_voice'):
                self.tts_module.set_voice(voice_id)
                logger.info(f"✅ Voz alterada para: {voice_id}")

                # Enviar confirmação de sucesso
                response = {
                    "type": "voice_changed",
                    "voice_id": voice_id,
                    "status": "success"
                }
            else:
                logger.warning(f"⚠️ TTS module não suporta mudança de voz dinâmica")
                response = {
                    "type": "voice_changed",
                    "voice_id": voice_id,
                    "status": "unsupported"
                }
        else:
            logger.error(f"❌ Voz inválida ou TTS não configurado: {voice_id}")
            response = {
                "type": "voice_changed",
                "voice_id": voice_id,
                "status": "error"
            }

        if channel.readyState == "open":
            channel.send(json.dumps(response))

    def update_stats(self, latency: float) -> Any:
        """Atualizar estatísticas de latência"""
        n = self.stats["total_messages"]
        if n > 0:
            self.stats["avg_latency"] = (
                (self.stats["avg_latency"] * (n - 1) + latency) / n
            )
        else:
            self.stats["avg_latency"] = latency
            
        self.stats["min_latency"] = min(self.stats["min_latency"], latency)
        self.stats["max_latency"] = max(self.stats["max_latency"], latency)
        
    async def get_stats(self, request) -> Any:
        """Endpoint de estatísticas"""
        return web.json_response(self.stats)
        
    async def health_check(self, request) -> Any:
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "active_connections": len(self.peers),
            "uptime": datetime.now().isoformat()
        })
        
    async def handle_websocket(self, request) -> Any:
        """
        Endpoint WebSocket para compatibilidade com UltravoxChat frontend
        Processa áudio PCM diretamente via WebSocket
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Usar IP/porta como identificador de sessão para manter contexto
        remote = request.remote
        session_id = f"ws_{remote}_{request.headers.get('User-Agent', 'unknown')[:20]}".replace(" ", "_")
        logger.info(f"🔌 Nova conexão WebSocket: {session_id}")
        
        # Inicializar memory store se necessário
        if not self.memory_store.is_initialized:
            await self.memory_store.initialize()

        self.stats["total_connections"] += 1
        self.stats["active_connections"] += 1

        # Adicionar conexão ao set para envio de métricas
        self.websocket_connections.add(ws)

        # Armazenar informações da sessão (incluindo voz selecionada)
        session_data = {
            'voice_id': 'af_bella',  # Voz padrão
            'system_prompt': None,
            'audio_buffer': {},  # Buffer para chunks de áudio
        }
        self.connections[session_id] = session_data
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    # Verificar se estamos aguardando dados binários após header
                    audio_buffer = self.connections.get(session_id, {}).get('audio_buffer', {})
                    if audio_buffer.get('waiting_for_binary'):
                        # 🚀 OTIMIZAÇÃO: Processar áudio binário com metadados do header
                        start_time = asyncio.get_event_loop().time()

                        logger.info(f"🚀 === ÁUDIO BINÁRIO RECEBIDO (OTIMIZAÇÃO) ===")
                        logger.info(f"   📊 Dados brutos: {len(msg.data)} bytes")
                        logger.info(f"   📦 Esperados: {audio_buffer.get('expected_bytes', 0)} bytes")
                        logger.info(f"   🎤 Samples esperados: {audio_buffer.get('expected_samples', 0)}")
                        logger.info(f"   🔊 Taxa: {audio_buffer.get('sample_rate', 16000)}Hz")
                        logger.info(f"   📱 Recorder: {audio_buffer.get('recorder', 'unknown')}")

                        # Validar se recebemos o tamanho correto
                        expected_bytes = audio_buffer.get('expected_bytes', 0)
                        if len(msg.data) != expected_bytes:
                            logger.warning(f"⚠️ TAMANHO INCONSISTENTE: Esperado {expected_bytes}, recebido {len(msg.data)}")

                        # Converter dados binários diretamente para array de áudio
                        # Os dados já estão em int16 como enviado pelo frontend
                        audio_int16 = np.frombuffer(msg.data, dtype=np.int16)
                        audio_data = audio_int16.astype(np.float32) / 32768.0

                        audio_duration_sec = len(audio_data) / audio_buffer.get('sample_rate', 16000)
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                        logger.info(f"   🎧 Áudio processado: {len(audio_data)} samples")
                        logger.info(f"   ⏱️  Duração: {audio_duration_sec:.2f}s")
                        logger.info(f"   ✅ SEM CONVERSÃO BASE64 - Dados diretos!")

                        # Limpar buffer após processar
                        self.connections[session_id]['audio_buffer'] = {}

                    else:
                        # Se não estamos esperando dados binários, ignorar mensagem binária não reconhecida
                        logger.warning(f"⚠️ Mensagem binária recebida sem header prévio - ignorando {len(msg.data)} bytes")
                        continue

                    response_text = ""
                    response_audio = None

                    if self.audio_processor:
                        logger.info(f"🤖 === PROCESSAMENTO ULTRAVOX ===")
                        logger.info(f"   🔄 Processando {len(audio_data)} samples com Ultravox v0.6 8B")

                        # Obter dados da sessão
                        voice_id = self.connections.get(session_id, {}).get('voice_id', 'af_bella')

                        # Obter contexto da sessão
                        context_messages = await self.memory_store.get_context(session_id, max_messages=10)
                        context = ""

                        if context_messages:
                            # Formatar contexto para o Ultravox
                            for msg_ctx in context_messages[-6:]:  # Últimas 6 mensagens
                                if msg_ctx['role'] == 'user':
                                    context += f"User: {msg_ctx['content']}\n"
                                elif msg_ctx['role'] == 'assistant':
                                    context += f"Assistant: {msg_ctx['content']}\n\n"
                            logger.info(f"   🧠 Contexto: {len(context_messages)} mensagens ({len(context)} chars)")
                        else:
                            logger.info(f"   🧠 Contexto: Nenhuma mensagem anterior")

                        # Validar áudio de entrada se validador disponível
                        if self.audio_validator:
                            try:
                                validated_audio, validation_metadata = self.audio_validator.validate_user_input(
                                    audio_data,
                                    session_id=session_id,
                                    expected_language='pt'  # Configurar idioma conforme necessário
                                )
                                audio_data = validated_audio  # Usar áudio validado

                                # Log da validação com Groq se disponível
                                if 'groq_validation' in validation_metadata:
                                    groq = validation_metadata['groq_validation']
                                    if 'transcription' in groq:
                                        logger.info(f"   🤖 [Groq] Transcrição entrada: '{groq['transcription'][:50]}...'")
                                    if not groq.get('has_voice', True):
                                        logger.warning(f"   ⚠️ [Groq] Nenhuma voz detectada no áudio de entrada")
                            except Exception as e:
                                logger.warning(f"   ⚠️ Erro na validação de entrada: {e}")
                                # Continuar mesmo com erro de validação

                        # Processar com Ultravox incluindo contexto e voice_id
                        ultravox_start = asyncio.get_event_loop().time()
                        response_text = await self.audio_processor.process_audio(
                            audio_data,
                            session_id,
                            context=context if context else None,
                            voice_id=voice_id
                        )
                        ultravox_time = (asyncio.get_event_loop().time() - ultravox_start) * 1000

                        logger.info(f"   ✅ Ultravox processado em {ultravox_time:.1f}ms")
                        logger.info(f"   📝 Resposta: '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")

                        # Salvar interação na memória
                        # TODO: Idealmente teríamos a transcrição do áudio
                        await self.memory_store.add_message(
                            session_id=session_id,
                            role="user",
                            content="[áudio processado]"
                        )
                        await self.memory_store.add_message(
                            session_id=session_id,
                            role="assistant",
                            content=response_text
                        )


                        # Gerar áudio de resposta com TTS
                        if self.tts_module and response_text:
                            logger.info(f"🔊 === GERAÇÃO TTS ===")
                            logger.info(f"   📝 Texto: '{response_text[:80]}{'...' if len(response_text) > 80 else ''}'")
                            logger.info(f"   📏 Tamanho texto: {len(response_text)} caracteres")

                            tts_start = asyncio.get_event_loop().time()
                            response_audio = await self.tts_module.synthesize(response_text)
                            tts_time = (asyncio.get_event_loop().time() - tts_start) * 1000

                            # Salvar áudio original para validação antes da conversão
                            original_audio_for_validation = response_audio

                            # TTS retorna bytes de float32, precisamos converter para int16
                            if isinstance(response_audio, bytes):
                                # Converter bytes float32 para numpy array
                                audio_float32 = np.frombuffer(response_audio, dtype=np.float32)
                                audio_duration_tts = len(audio_float32) / 37800.0  # Taxa do TTS
                                # Converter para int16
                                response_audio = (audio_float32 * 32767).astype(np.int16).tobytes()
                                logger.info(f"   ✅ TTS gerado em {tts_time:.1f}ms")
                                logger.info(f"   🎵 Formato: Float32 → Int16")
                                logger.info(f"   📊 Dados: {len(response_audio)} bytes ({len(audio_float32)} samples)")
                                logger.info(f"   ⏱️  Duração áudio: {audio_duration_tts:.2f}s")
                                logger.info(f"   🔊 Taxa: 37.8kHz (TTS)")
                            elif isinstance(response_audio, np.ndarray):
                                # Se já for numpy array, converter diretamente
                                audio_duration_tts = len(response_audio) / 37800.0
                                response_audio = (response_audio * 32767).astype(np.int16).tobytes()
                                logger.info(f"   ✅ TTS gerado em {tts_time:.1f}ms")
                                logger.info(f"   🎵 Formato: NumPy → Int16")
                                logger.info(f"   📊 Dados: {len(response_audio)} bytes")
                                logger.info(f"   ⏱️  Duração áudio: {audio_duration_tts:.2f}s")
                            else:
                                logger.warning(f"   ⚠️  Formato TTS não reconhecido: {type(response_audio)}")

                            # Validar qualidade do TTS com o validador da pipeline
                            if self.audio_validator and original_audio_for_validation and response_text:
                                try:
                                    # Validar saída do TTS
                                    validated_tts, tts_metadata = self.audio_validator.validate_tts_output(
                                        original_audio_for_validation,
                                        text_input=response_text,
                                        tts_engine="http_service",
                                        voice_id=voice_id or None,
                                        session_id=session_id
                                    )

                                    # Log da validação com Groq se disponível
                                    if 'groq_validation' in tts_metadata:
                                        groq = tts_metadata['groq_validation']
                                        if 'transcription' in groq:
                                            logger.info(f"   🤖 [Groq] Transcrição TTS: '{groq['transcription'][:50]}...'")
                                        if 'quality_score' in groq:
                                            logger.info(f"   🤖 [Groq] Qualidade TTS: {groq['quality_score']}/5")
                                        if groq.get('quality_score', 5) < 3:
                                            logger.warning(f"   ⚠️ [Groq] Baixa qualidade detectada no TTS")
                                except Exception as e:
                                    logger.warning(f"   ⚠️ Erro na validação TTS: {e}")
                                    # Continuar mesmo com erro de validação

                            # Validar qualidade do TTS em modo desenvolvimento (mantém compatibilidade)
                            elif self.dev_metrics and original_audio_for_validation and response_text:
                                try:
                                    # Usar o áudio original em bytes para validação
                                    if isinstance(original_audio_for_validation, (bytes, np.ndarray)):
                                        # Assumir que são dados float32 PCM
                                        sample_rate = 37800  # Taxa do TTS
                                        voice_id_dev = None

                                        # Executar validação assíncrona (não bloqueia o TTS)
                                        asyncio.create_task(
                                            self.dev_metrics.validate_tts_quality(
                                                original_audio_for_validation,
                                                response_text,
                                                sample_rate,
                                                voice_id_dev
                                            )
                                        )
                                        logger.debug(f"🔍 [DEV] Validação TTS iniciada para: '{response_text[:30]}...'")
                                except Exception as e:
                                    logger.debug(f"⚠️ [DEV] Erro ao iniciar validação TTS: {e}")
                        else:
                            if not self.tts_module:
                                logger.warning("⚠️  === TTS NÃO DISPONÍVEL ===")
                                logger.warning("   ❌ Módulo TTS não configurado")
                            elif not response_text:
                                logger.warning("⚠️  === TTS CANCELADO ===")
                                logger.warning("   📝 Resposta vazia do Ultravox")
                    else:
                        logger.error("❌ Processador de áudio não configurado!")
                        response_text = "Processador de áudio não configurado"

                    # Calcular latência
                    latency = (asyncio.get_event_loop().time() - start_time) * 1000
                    self.update_stats(latency)

                    # Enviar resposta via WebSocket no formato esperado pelo frontend
                    response = {
                        "type": "metrics",
                        "response": response_text,
                        "latency": latency
                    }

                    # Enviar texto primeiro
                    await ws.send_str(json.dumps(response))

                    # Depois enviar áudio se houver
                    audio_sent = False
                    if response_audio:
                        await ws.send_bytes(response_audio)
                        audio_sent = True

                    # Log de resumo da resposta
                    logger.info(f"📤 === RESPOSTA ENVIADA ===")
                    logger.info(f"   ⚡ Latência total: {latency:.1f}ms")
                    logger.info(f"   📝 Texto: {'✅ Enviado' if response_text else '❌ Vazio'}")
                    logger.info(f"   🎵 Áudio: {'✅ Enviado' if audio_sent else '❌ Não enviado'}")
                    logger.info(f"   👤 Cliente: {session_id}")
                    logger.info(f"   🔢 Mensagem #{self.stats['total_messages'] + 1}")

                    self.stats["total_messages"] += 1
                    
                elif msg.type == WSMsgType.TEXT:
                    # Mensagem JSON recebida
                    try:
                        data = json.loads(msg.data)
                        logger.info(f"📝 Mensagem JSON recebida: {data.get('type', 'unknown')}")

                        # Handle voice change message
                        if data.get('type') == 'voice_change':
                            voice_id = data.get('voice_id')
                            if voice_id and self.tts_module:
                                # Update TTS module voice
                                if hasattr(self.tts_module, 'set_voice'):
                                    self.tts_module.set_voice(voice_id)
                                    logger.info(f"🔊 Voz alterada para: {voice_id}")
                                    # Send confirmation to client
                                    await ws.send_str(json.dumps({
                                        'type': 'voice_changed',
                                        'voice_id': voice_id,
                                        'status': 'success'
                                    }))
                                else:
                                    logger.warning(f"⚠️ TTS module não suporta mudança de voz dinâmica")
                                    await ws.send_str(json.dumps({
                                        'type': 'voice_changed',
                                        'voice_id': voice_id,
                                        'status': 'unsupported'
                                    }))
                        elif data.get('type') == 'audio_binary_header':
                            # 🚀 OTIMIZAÇÃO: Header de áudio binário - esperar dados binários
                            logger.info(f"🚀 === OTIMIZAÇÃO BINÁRIA WEBSOCKET ATIVADA ===")
                            logger.info(f"   📦 Header recebido: {data.get('samples')} samples, {data.get('bytes')} bytes")
                            logger.info(f"   🔊 Taxa: {data.get('sampleRate')}Hz, Formato: {data.get('format')}")
                            logger.info(f"   ⏰ Timestamp: {data.get('timestamp')}")
                            logger.info(f"   📱 Recorder: {data.get('recorder')}")

                            # Armazenar metadados no buffer da sessão
                            self.connections[session_id]['audio_buffer'] = {
                                'waiting_for_binary': True,
                                'sample_rate': data.get('sampleRate', 16000),
                                'format': data.get('format', 'pcm16'),
                                'expected_samples': data.get('samples', 0),
                                'expected_bytes': data.get('bytes', 0),
                                'timestamp': data.get('timestamp'),
                                'recorder': data.get('recorder', 'unknown')
                            }

                            logger.info(f"   ✅ Aguardando {data.get('bytes')} bytes de áudio binário...")

                        elif data.get('type') == 'audio_chunk':
                            # Decodificar dados de áudio base64
                            audio_b64 = data.get('data', '')
                            try:
                                logger.info(f"🔍 [DEBUG] Dados base64 recebidos: {len(audio_b64)} caracteres")

                                # Decodificar base64
                                audio_bytes = base64.b64decode(audio_b64)
                                logger.info(f"🎧 Audio chunk decodificado: {len(audio_bytes)} bytes")
                                
                                # Processar como mensagem binária
                                start_time = asyncio.get_event_loop().time()
                                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                
                                # Converter bytes para numpy array int16 -> float32
                                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                                audio_data = audio_int16.astype(np.float32) / 32768.0
                                audio_duration_sec = len(audio_data) / 16000

                                # Calcular estatísticas do áudio recebido
                                audio_rms = np.sqrt(np.mean(audio_data**2))
                                audio_peak = np.abs(audio_data).max()

                                logger.info(f"🎧 === REQUISIÇÃO ÁUDIO RECEBIDA ===")
                                logger.info(f"   ⏰ Timestamp: {timestamp}")
                                logger.info(f"   📊 Dados brutos: {len(audio_bytes)} bytes")
                                logger.info(f"   📦 Header PCM: ❌ Não (JSON)")
                                logger.info(f"   🎤 Samples áudio: {len(audio_data)}")
                                logger.info(f"   ⏱️  Duração: {audio_duration_sec:.2f}s")
                                logger.info(f"   🔊 Taxa: 16kHz, Int16→Float32")
                                logger.info(f"   📈 RMS: {audio_rms:.4f}, Peak: {audio_peak:.4f}")
                                logger.info(f"   👤 Sessão: {session_id}")

                                # Validar qualidade do áudio
                                if audio_duration_sec < 0.5:
                                    logger.warning(f"⚠️ ÁUDIO MUITO CURTO: {audio_duration_sec:.2f}s (mínimo: 0.5s)")
                                    logger.warning(f"   Ultravox pode não conseguir processar áudio tão curto!")

                                if audio_rms < 0.001:
                                    logger.warning(f"⚠️ ÁUDIO MUITO BAIXO: RMS={audio_rms:.6f}")
                                    logger.warning(f"   Possível silêncio ou problema de captura no frontend!")

                                if audio_peak < 0.01:
                                    logger.warning(f"⚠️ SINAL MUITO FRACO: Peak={audio_peak:.6f}")
                                    logger.warning(f"   Verificar configuração do microfone no frontend!")

                                # VALIDAÇÃO COM GROQ (desenvolvimento)
                                groq_transcription = None
                                # Importar config aqui para verificar se está em desenvolvimento
                                from config import get_config
                                config = get_config()
                                if self.dev_metrics and config.IS_DEVELOPMENT:
                                    try:
                                        logger.info(f"🔍 === VALIDAÇÃO GROQ (DEBUG) ===")
                                        logger.info(f"   📊 Validando áudio: {len(audio_data)} samples @ 16kHz")
                                        logger.info(f"   ⏱️  Duração: {audio_duration_sec:.2f}s")

                                        # Converter áudio para WAV para transcrição
                                        import wave
                                        import tempfile
                                        import os

                                        # Criar arquivo WAV temporário
                                        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                                        temp_filename = temp_file.name

                                        try:
                                            with wave.open(temp_file, 'wb') as wav_file:
                                                wav_file.setnchannels(1)  # Mono
                                                wav_file.setsampwidth(2)  # 16-bit
                                                wav_file.setframerate(16000)
                                                # Converter float32 para int16
                                                audio_int16_temp = (audio_data * 32767).astype(np.int16)
                                                wav_file.writeframes(audio_int16_temp.tobytes())
                                            temp_file.close()

                                            # Ler o arquivo WAV criado
                                            with open(temp_filename, 'rb') as f:
                                                wav_data = f.read()

                                            # Transcrever com Groq
                                            from src.services.stt.transcription.groq_transcription import GroqTranscription
                                            groq = GroqTranscription()
                                            groq_transcription = await groq.transcribe_audio(
                                                wav_data,
                                                sample_rate=16000,
                                                language='pt'  # Usar português como padrão (pode ser en, es, etc)
                                            )

                                            if groq_transcription:
                                                logger.info(f"   ✅ GROQ Transcrição: '{groq_transcription}'")
                                                logger.info(f"   📊 Tamanho: {len(groq_transcription)} caracteres")

                                                # Enviar transcrição para frontend
                                                await ws.send_str(json.dumps({
                                                    "type": "groq_transcription",
                                                    "text": groq_transcription,
                                                    "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                                    "audio_duration": audio_duration_sec,
                                                    "audio_samples": len(audio_data),
                                                    "audio_rms": float(audio_rms),
                                                    "audio_peak": float(audio_peak)
                                                }))
                                            else:
                                                logger.warning(f"   ⚠️ GROQ não conseguiu transcrever o áudio!")
                                                logger.warning(f"   ⚠️ Possível problema: áudio muito baixo, silêncio ou ruído")
                                                logger.warning(f"   ⚠️ RMS={audio_rms:.6f}, Peak={audio_peak:.6f}")

                                                # Enviar alerta para frontend
                                                await ws.send_str(json.dumps({
                                                    "type": "groq_error",
                                                    "error": "Áudio não pôde ser transcrito - possível silêncio ou ruído",
                                                    "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                                    "audio_rms": float(audio_rms),
                                                    "audio_peak": float(audio_peak)
                                                }))
                                        finally:
                                            # Limpar arquivo temporário
                                            if os.path.exists(temp_filename):
                                                os.unlink(temp_filename)

                                    except Exception as e:
                                        logger.error(f"   ❌ Erro na validação Groq: {e}")
                                        import traceback
                                        logger.error(traceback.format_exc())
                                        # Enviar erro para frontend
                                        await ws.send_str(json.dumps({
                                            "type": "groq_error",
                                            "error": f"Erro na transcrição: {str(e)}",
                                            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                        }))

                                response_text = ""
                                response_audio = None
                                
                                if self.audio_processor:
                                    logger.info(f"🤖 === PROCESSAMENTO ULTRAVOX ===")
                                    logger.info(f"   🔄 Processando {len(audio_data)} samples com Ultravox v0.6 8B")

                                    # Obter dados da sessão para personalizar resposta
                                    voice_id = self.connections.get(session_id, {}).get('voice_id', 'af_bella')
                                    voice_info = self.get_voice_info(voice_id)

                                    # Obter contexto da sessão
                                    context_messages = await self.memory_store.get_context(session_id, max_messages=10)
                                    context = ""

                                    # Adicionar instrução sobre voz/personagem ao contexto
                                    if voice_info:
                                        voice_instruction = f"You are {voice_info['name']}, speaking in {voice_info['language']}. "
                                        voice_instruction += f"Your personality is {voice_info['personality']}. "
                                        context = voice_instruction + "\n\n"
                                        logger.info(f"   🎭 Personagem: {voice_info['name']} ({voice_info['language']})")

                                    if context_messages:
                                        # Formatar contexto para o Ultravox
                                        for msg in context_messages[-6:]:  # Últimas 6 mensagens
                                            if msg['role'] == 'user':
                                                context += f"User: {msg['content']}\n"
                                            elif msg['role'] == 'assistant':
                                                context += f"Assistant: {msg['content']}\n\n"
                                        logger.info(f"   🧠 Contexto: {len(context_messages)} mensagens ({len(context)} chars)")
                                    else:
                                        logger.info(f"   🧠 Contexto: Nenhuma mensagem anterior")

                                    # Importar pipeline de conversação
                                    from pipeline.conversation import ConversationPipeline

                                    # Obter system_prompt da sessão
                                    system_prompt = self.connections.get(session_id, {}).get('system_prompt', '')

                                    # Detectar idioma baseado na voz
                                    language = ConversationPipeline.get_language_from_voice_id(voice_id)

                                    # Formatar contexto com instruções da pipeline
                                    formatted_context = ConversationPipeline.format_context_with_instructions(
                                        context=context,
                                        language=language,
                                        custom_prompt=system_prompt
                                    )

                                    # Processar com Ultravox incluindo contexto formatado e voice_id
                                    ultravox_start = asyncio.get_event_loop().time()
                                    response_text = await self.audio_processor.process_audio(
                                        audio_data,
                                        context=formatted_context,
                                        voice_id=voice_id
                                    )
                                    ultravox_time = asyncio.get_event_loop().time() - ultravox_start
                                    
                                    logger.info(f"   ✅ Resposta: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
                                    logger.info(f"   ⏱️  Tempo Ultravox: {ultravox_time*1000:.0f}ms")

                                    # Salvar mensagem do usuário com transcrição se disponível
                                    user_message = groq_transcription if groq_transcription else "[Mensagem de áudio]"

                                    # Validar coerência da resposta do Ultravox com Groq LLM
                                    if self.audio_validator and user_message and response_text:
                                        try:
                                            llm_validation = self.audio_validator.validate_ultravox_response(
                                                user_question=user_message,
                                                ultravox_response=response_text,
                                                session_id=session_id
                                            )

                                            if 'llm_validation' in llm_validation:
                                                llm_data = llm_validation['llm_validation']
                                                if 'coherence_score' in llm_data:
                                                    logger.info(f"   🤖 === VALIDAÇÃO LLM DA RESPOSTA ===")
                                                    logger.info(f"      Coerência: {llm_data['coherence_score']}/10")
                                                    logger.info(f"      Completude: {llm_data['completeness_score']}/10")
                                                    logger.info(f"      Qualidade: {llm_data['quality_score']}/10")
                                                    logger.info(f"      Válida: {'✅' if llm_data.get('is_valid', False) else '❌'}")

                                                    if llm_data.get('issues'):
                                                        logger.warning(f"      ⚠️ Problemas: {', '.join(llm_data['issues'])}")

                                                    # Score geral
                                                    overall_score = llm_validation.get('overall_score', 0)
                                                    if overall_score < 5:
                                                        logger.warning(f"      ⚠️ ATENÇÃO: Score baixo ({overall_score:.1f}/10)")
                                                        logger.warning(f"      📝 Análise: {llm_data.get('analysis', '')[:150]}")
                                                    else:
                                                        logger.info(f"      ✅ Score geral: {overall_score:.1f}/10")
                                        except Exception as e:
                                            logger.debug(f"⚠️ Erro ao validar resposta com LLM: {e}")

                                    await self.memory_store.add_message(session_id, "user", user_message)
                                    # Salvar resposta do assistente
                                    await self.memory_store.add_message(session_id, "assistant", response_text)
                                
                                # Gerar TTS se há resposta
                                if response_text and response_text.strip() and self.tts_module:
                                    logger.info(f"🔊 === GERAÇÃO TTS ===")
                                    logger.info(f"   📝 Texto: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
                                    
                                    tts_start = asyncio.get_event_loop().time()
                                    response_audio = await self.tts_module.synthesize(response_text)
                                    tts_time = asyncio.get_event_loop().time() - tts_start
                                    
                                    if response_audio is not None:
                                        logger.info(f"   ✅ TTS gerado: {len(response_audio)} bytes")
                                        logger.info(f"   ⏱️  Tempo TTS: {tts_time*1000:.0f}ms")
                                    else:
                                        logger.warning(f"   ❌ Falha na geração TTS")
                                
                                # Enviar resposta via WebSocket
                                # Comprimir áudio para MP3 para reduzir tamanho
                                compressed_audio = None
                                if response_audio:
                                    try:
                                        # Verificar se é bytes ou numpy array
                                        if isinstance(response_audio, bytes):
                                            # Já está em bytes, apenas codificar
                                            compressed_audio = base64.b64encode(response_audio).decode('utf-8')
                                        else:
                                            # É numpy array, converter
                                            if response_audio.dtype == np.float32:
                                                audio_int16 = (response_audio * 32767).astype(np.int16)
                                            else:
                                                audio_int16 = response_audio
                                            compressed_audio = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
                                        
                                        # Se ainda for muito grande, dividir em chunks
                                        max_size = 500000  # 500KB limite seguro
                                        if len(compressed_audio) > max_size:
                                            logger.warning(f"⚠️ Áudio muito grande ({len(compressed_audio)} bytes), enviando apenas primeiros {max_size} bytes")
                                            # Enviar apenas parte do áudio
                                            compressed_audio = compressed_audio[:max_size]
                                    except Exception as e:
                                        logger.error(f"❌ Erro ao comprimir áudio: {e}")
                                        compressed_audio = None
                                
                                response = {
                                    'type': 'response',
                                    'text': response_text,
                                    'audio': compressed_audio,
                                    'timestamp': timestamp
                                }
                                
                                response_json = json.dumps(response)
                                logger.info(f"📤 Enviando resposta: {len(response_json)} bytes")
                                
                                if len(response_json) > 1000000:  # 1MB
                                    logger.warning(f"⚠️ Resposta muito grande: {len(response_json)} bytes")
                                    # Enviar sem áudio se for muito grande
                                    response = {
                                        'type': 'response',
                                        'text': response_text,
                                        'audio': None,
                                        'error': 'Audio too large',
                                        'timestamp': timestamp
                                    }
                                    await ws.send_str(json.dumps(response))
                                else:
                                    await ws.send_str(response_json)
                                
                                end_time = asyncio.get_event_loop().time()
                                total_time = end_time - start_time
                                
                                logger.info(f"📊 === RESPOSTA ENVIADA ===")
                                logger.info(f"   ⏱️  Tempo total: {total_time*1000:.0f}ms")
                                logger.info(f"   📝 Texto: {'✅ Enviado' if response_text else '❌ Não enviado'}")
                                logger.info(f"   🎵 Áudio: {'✅ Enviado' if response_audio else '❌ Não enviado'}")
                                logger.info(f"   👤 Cliente: {session_id}")
                                logger.info(f"   🔢 Mensagem #{self.stats['total_messages'] + 1}")
                                
                                self.stats["total_messages"] += 1
                                
                            except Exception as e:
                                logger.error(f"❌ Erro ao processar audio_chunk: {e}")
                        elif data.get('type') == 'config':
                            # Configuração inicial do cliente
                            logger.info(f"⚙️ Configuração inicial do cliente recebida")

                            # Processar voice_id se estiver presente (aceitar tanto 'voice' quanto 'voice_id')
                            voice_id = data.get('voice_id') or data.get('voice')
                            system_prompt = data.get('system_prompt')

                            # Atualizar dados da sessão
                            if session_id in self.connections:
                                if voice_id:
                                    self.connections[session_id]['voice_id'] = voice_id
                                    logger.info(f"🎯 Voz da sessão atualizada para: {voice_id}")
                                if system_prompt:
                                    self.connections[session_id]['system_prompt'] = system_prompt
                                    logger.info(f"📝 System prompt atualizado para sessão")

                            # Atualizar TTS se disponível
                            if voice_id and self.tts_module:
                                if hasattr(self.tts_module, 'set_voice'):
                                    self.tts_module.set_voice(voice_id)
                                    logger.info(f"🔊 Voz do TTS alterada para: {voice_id} via WebSocket config")

                            await ws.send_str(json.dumps({
                                'type': 'config_ack',
                                'message': '✅ Conectado ao servidor Ultravox+TTS',
                                'tts_enabled': True,
                                'speech_enabled': True,
                                'server_info': 'ultravox_server v1.0',
                                'voice_id': voice_id if voice_id else None
                            }))

                        elif data.get('type') == 'text_message':
                            # Mensagem de texto (como cliques de opção)
                            text = data.get('text', '')
                            timestamp = data.get('timestamp', 0)

                            logger.info(f"💬 === MENSAGEM DE TEXTO RECEBIDA ===")
                            logger.info(f"   📝 Texto: '{text}'")
                            logger.info(f"   ⏰ Timestamp: {timestamp}")
                            logger.info(f"   👤 Sessão: {session_id}")

                            if text.strip():
                                try:
                                    # Enviar status de processamento
                                    await ws.send_str(json.dumps({
                                        'type': 'processing',
                                        'message': '🤖 Processando sua mensagem...'
                                    }))

                                    # Obter dados da sessão
                                    voice_id = self.connections.get(session_id, {}).get('voice_id', 'af_bella')
                                    system_prompt = self.connections.get(session_id, {}).get('system_prompt', '')

                                    # Processar mensagem de texto com IA (se disponível)
                                    response_text = ""
                                    if self.audio_processor:
                                        # Obter contexto da conversa
                                        context_messages = await self.memory_store.get_context(session_id, max_messages=10)
                                        context = ""

                                        if context_messages:
                                            for msg in context_messages[-6:]:
                                                if msg['role'] == 'user':
                                                    context += f"User: {msg['content']}\n"
                                                elif msg['role'] == 'assistant':
                                                    context += f"Assistant: {msg['content']}\n\n"

                                        # Usar pipeline de conversação se disponível
                                        try:
                                            from pipeline.conversation import ConversationPipeline

                                            # Detectar idioma baseado na voz
                                            language = ConversationPipeline.get_language_from_voice_id(voice_id)

                                            # Formatar contexto com instruções da pipeline
                                            formatted_context = ConversationPipeline.format_context_with_instructions(
                                                context=context,
                                                language=language,
                                                custom_prompt=system_prompt
                                            )

                                            # Processar texto diretamente com LLM
                                            response_text = await ConversationPipeline.process_text_message(
                                                text_input=text,
                                                context=formatted_context,
                                                language=language,
                                                session_id=session_id
                                            )

                                            logger.info(f"   ✅ Resposta LLM: '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")

                                        except ImportError:
                                            # Fallback simples se pipeline não disponível
                                            response_text = f"Entendi sua mensagem: '{text}'. Como posso ajudar mais?"
                                            logger.warning("   ⚠️ Pipeline não disponível, usando resposta fallback")
                                    else:
                                        # Fallback sem processador de áudio
                                        response_text = f"Recebi sua mensagem: '{text}'. Obrigado!"

                                    # Salvar interação na memória
                                    await self.memory_store.add_message(session_id, "user", text)
                                    await self.memory_store.add_message(session_id, "assistant", response_text)

                                    # Gerar áudio TTS se módulo disponível
                                    response_audio = None
                                    if response_text and self.tts_module:
                                        logger.info(f"   🔊 Gerando TTS para resposta...")

                                        # Set voice before synthesis
                                        if hasattr(self.tts_module, 'set_voice'):
                                            self.tts_module.set_voice(voice_id)

                                        tts_start = asyncio.get_event_loop().time()
                                        response_audio = await self.tts_module.synthesize(response_text)
                                        tts_time = (asyncio.get_event_loop().time() - tts_start) * 1000

                                        if response_audio:
                                            logger.info(f"   ✅ TTS gerado: {len(response_audio)} bytes em {tts_time:.0f}ms")

                                    # Preparar resposta
                                    response_data = {
                                        'type': 'response',
                                        'text': response_text,
                                        'timestamp': datetime.now().isoformat(),
                                        'processing_info': {
                                            'input_type': 'text_message',
                                            'voice_id': voice_id,
                                            'original_text': text
                                        }
                                    }

                                    # Incluir áudio se gerado
                                    if response_audio:
                                        # Converter PCM para WAV com headers apropriados
                                        wav_audio = pcm_to_wav(response_audio, sample_rate=24000, channels=1, bits_per_sample=16)

                                        # Converter para base64
                                        audio_b64 = base64.b64encode(wav_audio).decode('utf-8')
                                        response_data['audio'] = audio_b64
                                        response_data['audio_format'] = 'wav'
                                        response_data['sample_rate'] = 24000

                                    # Enviar resposta
                                    await ws.send_str(json.dumps(response_data))

                                    logger.info(f"   ✅ Resposta enviada para clique de opção")
                                    logger.info(f"   📝 Texto: {'✅' if response_text else '❌'}")
                                    logger.info(f"   🎵 Áudio: {'✅' if response_audio else '❌'}")

                                except Exception as e:
                                    logger.error(f"❌ Erro ao processar text_message: {e}")
                                    await ws.send_str(json.dumps({
                                        'type': 'error',
                                        'message': f'Erro ao processar mensagem: {str(e)}'
                                    }))
                            else:
                                logger.warning(f"⚠️ Texto vazio em text_message")
                                await ws.send_str(json.dumps({
                                    'type': 'error',
                                    'message': 'Mensagem de texto vazia'
                                }))

                        elif data.get('type') == 'text_to_speech':
                            # Mensagem de texto para TTS
                            text = data.get('text', '')
                            voice_id = data.get('voice_id', 'af_bella')
                            speed = data.get('speed', 1.0)
                            volume = data.get('volume', 1.0)

                            logger.info(f"💬 TTS solicitado: '{text[:50]}...' ({len(text)} chars)")
                            logger.info(f"🎵 Voice: {voice_id}, Speed: {speed}, Volume: {volume}")

                            if text.strip() and self.tts_module:
                                try:
                                    # Enviar status inicial
                                    await ws.send_str(json.dumps({
                                        'type': 'processing',
                                        'message': f'🎵 Processando com TTS ({voice_id})...',
                                        'text_length': len(text),
                                        'voice_id': voice_id
                                    }))

                                    # Set the voice before synthesis
                                    if hasattr(self.tts_module, 'set_voice'):
                                        self.tts_module.set_voice(voice_id)
                                        logger.info(f"🎯 Voice set to: {voice_id} before TTS synthesis")

                                    # Sintetizar com TTS
                                    tts_start = asyncio.get_event_loop().time()
                                    response_audio = await self.tts_module.synthesize(text, voice_id=voice_id, speed=speed)
                                    tts_time = (asyncio.get_event_loop().time() - tts_start) * 1000

                                    if response_audio is not None:
                                        logger.info(f"✅ TTS gerado: {len(response_audio)} bytes em {tts_time:.0f}ms")

                                        # Converter PCM para WAV com headers apropriados
                                        logger.info(f"🔊 Convertendo PCM para WAV...")
                                        wav_audio = pcm_to_wav(response_audio, sample_rate=24000, channels=1, bits_per_sample=16)
                                        logger.info(f"✅ WAV criado: {len(wav_audio)} bytes (de {len(response_audio)} PCM)")

                                        # Converter áudio WAV para base64
                                        audio_b64 = base64.b64encode(wav_audio).decode('utf-8')

                                        # Enviar resposta TTS
                                        await ws.send_str(json.dumps({
                                            'type': 'tts_response',
                                            'text': text,
                                            'audio_data': audio_b64,
                                            'audio_format': 'wav',
                                            'sample_rate': 24000,  # TTS default
                                            'is_final': True,
                                            'voice_id': voice_id,
                                            'processing_info': {
                                                'tts_latency_ms': round(tts_time),
                                                'audio_bytes': len(wav_audio),
                                                'pcm_bytes': len(response_audio),
                                                'voice': voice_id,
                                                'speed': speed,
                                                'volume': volume
                                            }
                                        }))
                                    else:
                                        logger.error(f"❌ Falha na geração TTS")
                                        await ws.send_str(json.dumps({
                                            'type': 'error',
                                            'message': 'Erro na síntese de voz'
                                        }))

                                except Exception as e:
                                    logger.error(f"❌ Erro no TTS: {e}")
                                    await ws.send_str(json.dumps({
                                        'type': 'error',
                                        'message': f'Erro na síntese de voz: {str(e)}'
                                    }))
                            else:
                                if not text.strip():
                                    logger.warning(f"⚠️ Texto vazio para TTS")
                                    await ws.send_str(json.dumps({
                                        'type': 'error',
                                        'message': 'Texto vazio - digite algo para sintetizar'
                                    }))
                                elif not self.tts_module:
                                    logger.warning(f"⚠️ TTS não disponível")
                                    await ws.send_str(json.dumps({
                                        'type': 'error',
                                        'message': 'Módulo TTS não configurado'
                                    }))
                        else:
                            logger.warning(f"⚠️  Tipo de mensagem não suportado: {data.get('type')}")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Erro ao decodificar JSON: {e}")
                    except Exception as e:
                        logger.error(f"❌ Erro ao processar mensagem TEXT: {e}")
                
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'❌ Erro WebSocket: {ws.exception()}')
                    
        except Exception as e:
            logger.error(f"❌ Erro no WebSocket: {e}")
            
        finally:
            self.stats["active_connections"] -= 1
            # Remover conexão do set de métricas
            self.websocket_connections.discard(ws)
            logger.info(f"🧹 WebSocket desconectado: {session_id}")
            
        return ws
        
    async def serve_frontend(self, request) -> FileResponse:
        """Servir frontend React (index.html para qualquer rota)"""
        try:
            index_path = self.frontend_build_path / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            else:
                return web.Response(
                    text="Frontend não encontrado. Execute: cd frontend && npm run build",
                    status=404
                )
        except Exception as e:
            logger.error(f"Erro servindo frontend: {e}")
            return web.Response(text="Erro interno", status=500)
        
    async def start(self) -> Any:
        """Iniciar servidor WebRTC"""
        logger.info(f"🚀 Iniciando servidor WebRTC Python...")
        logger.info(f"📡 Escutando em {self.host}:{self.port}")
        logger.info(f"⚡ Modo: Ultra-baixa latência com DataChannel")
        logger.info(f"🎯 Latência esperada: 25-40ms")
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        # Monitor de stats
        asyncio.create_task(self.stats_monitor())
        
    async def stats_monitor(self) -> Any:
        """Monitor periódico de estatísticas"""
        while True:
            await asyncio.sleep(30)
            if self.stats["total_messages"] > 0:
                logger.info("📊 === ESTATÍSTICAS ===")
                logger.info(f"Conexões ativas: {self.stats['active_connections']}")
                logger.info(f"Total mensagens: {self.stats['total_messages']}")
                logger.info(f"Latência média: {self.stats['avg_latency']:.1f}ms")
                logger.info(f"Latência mín: {self.stats['min_latency']:.1f}ms")
                logger.info(f"Latência máx: {self.stats['max_latency']:.1f}ms")


class WebRTCModule:
    """
    Módulo WebRTC para integração com arquitetura Python
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # ✅ Phase 3c: Dynamic port support via environment variable
        import os
        from src.config.service_config import ServiceType, get_service_port
        try:
            dynamic_port = int(os.getenv("WEBRTC_PORT") or get_service_port(ServiceType.WEBRTC_GATEWAY))
        except (OSError, ConnectionError, RuntimeError):
            dynamic_port = config.get("port", 8020)  # Fallback to config or PORT_MATRIX default

        self.server = WebRTCServer(
            host=config.get("host", "0.0.0.0"),
            port=dynamic_port,
            ice_servers=config.get("ice_servers", None)
        )
        
    async def initialize(self) -> Any:
        """Inicializar módulo"""
        await self.server.start()
        
    def set_audio_processor(self, processor) -> Any:
        """Definir processador de áudio"""
        self.server.set_audio_processor(processor)
        
    def set_tts_module(self, tts) -> Any:
        """Definir módulo TTS"""
        self.server.set_tts_module(tts)
        
    async def cleanup(self) -> Any:
        """Limpar recursos"""
        for session_id in list(self.server.peers.keys()):
            await self.server.cleanup_peer(session_id)


# Exemplo de uso standalone
async def main() -> Any:
    logging.basicConfig(level=logging.INFO)
    
    server = WebRTCServer()
    
    # Configurar módulos (exemplo)
    # server.set_audio_processor(ultravox_module)
    # server.set_tts_module(tts_module)
    
    await server.start()
    
    # Manter rodando
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("🛑 Encerrando servidor...")


if __name__ == "__main__":
    asyncio.run(main())