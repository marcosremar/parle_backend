"""
WebRTC Bridge - Wrapper Python para o módulo Node.js
Permite integração com a arquitetura Python existente
"""

import asyncio
import subprocess
import json
import logging
from typing import Optional, Callable, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class WebRTCBridge:
    """
    Bridge entre Python e o módulo WebRTC Node.js
    Permite usar WebRTC nativo com a arquitetura Python
    """
    
    def __init__(self, 
                 ws_port: int = 8088,
                 ice_servers: list = None):
        """
        Inicializar bridge WebRTC
        
        Args:
            ws_port: Porta WebSocket para sinalização
            ice_servers: Servidores STUN/TURN
        """
        self.ws_port = ws_port
        self.ice_servers = ice_servers or [
            {"urls": "stun:stun.l.google.com:19302"}
        ]
        
        self.process = None
        self.audio_processor = None
        self.tts_module = None
        self.running = False
        
    def set_audio_processor(self, processor):
        """Definir processador de áudio (Ultravox)"""
        self.audio_processor = processor
        logger.info("Processador de áudio configurado")
        
    def set_tts_module(self, tts):
        """Definir módulo TTS"""
        self.tts_module = tts
        logger.info("Módulo TTS configurado")
        
    async def initialize(self):
        """Inicializar módulo WebRTC Node.js"""
        logger.info("🚀 Iniciando módulo WebRTC Node.js...")
        
        # Criar script de inicialização
        init_script = self._create_init_script()
        
        # Salvar script temporário
        script_path = str(Path.home() / ".cache" / "ultravox-pipeline" / "webrtc_server.js")
        with open(script_path, "w") as f:
            f.write(init_script)
        
        # Iniciar processo Node.js
        self.process = await asyncio.create_subprocess_exec(
            "node", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        self.running = True
        
        # Monitor de saída
        asyncio.create_task(self._monitor_output())
        
        logger.info(f"✅ WebRTC Bridge iniciado na porta {self.ws_port}")
        
    def _create_init_script(self) -> str:
        """Criar script de inicialização do servidor"""
        return f"""
const {{ WebRTCModule }} = require('{__file__.replace("webrtc_bridge.py", "")}');
const net = require('net');

// Criar módulo WebRTC
const webrtc = new WebRTCModule({{
    wsPort: {self.ws_port},
    iceServers: {json.dumps(self.ice_servers)}
}});

// Criar servidor IPC para comunicação com Python
const ipcServer = net.createServer((socket) => {{
    socket.on('data', async (data) => {{
        try {{
            const message = JSON.parse(data.toString());
            
            if (message.type === 'process_audio') {{
                // Processar áudio (seria chamado do Python)
                const result = await processAudio(message.audio, message.sessionId);
                socket.write(JSON.stringify(result));
            }}
        }} catch (error) {{
            socket.write(JSON.stringify({{error: error.message}}));
        }}
    }});
}});

ipcServer.listen(os.path.expanduser("~/.cache/ultravox-pipeline/tmp/webrtc_ipc.sock"));

// Configurar processador de áudio customizado
webrtc.setAudioProcessor({{
    processAudio: async (audioData, sessionId) => {{
        // Enviar para Python via IPC
        return new Promise((resolve) => {{
            const client = net.createConnection(os.path.expanduser("~/.cache/ultravox-pipeline/tmp/webrtc_python.sock"), () => {{
                client.write(JSON.stringify({{
                    type: 'process',
                    audio: Array.from(audioData),
                    sessionId: sessionId
                }}));
            }});
            
            client.on('data', (data) => {{
                const result = JSON.parse(data.toString());
                resolve(result.text);
                client.end();
            }});
        }});
    }}
}});

// Inicializar
webrtc.initialize().then(() => {{
    console.log('WebRTC Module ready');
}});

// Manter processo rodando
process.stdin.resume();
"""
        
    async def _monitor_output(self):
        """Monitorar saída do processo Node.js"""
        while self.running:
            if self.process and self.process.stdout:
                line = await self.process.stdout.readline()
                if line:
                    logger.debug(f"[WebRTC] {line.decode().strip()}")
            await asyncio.sleep(0.1)
            
    async def process_audio(self, audio_data: np.ndarray, session_id: str) -> Dict[str, Any]:
        """
        Processar áudio recebido via WebRTC
        
        Args:
            audio_data: Dados de áudio como numpy array
            session_id: ID da sessão
            
        Returns:
            Dict com resposta de texto e áudio
        """
        if not self.audio_processor:
            return {"text": "Processador não configurado", "audio": None}
            
        try:
            # Processar com Ultravox
            response_text = await self.audio_processor.process_audio(audio_data)
            
            # Gerar áudio com TTS
            response_audio = None
            if self.tts_module and response_text:
                response_audio = await self.tts_module.synthesize(response_text)
                
            return {
                "text": response_text,
                "audio": response_audio.tolist() if response_audio is not None else None
            }
            
        except Exception as e:
            logger.error(f"Erro processando áudio: {e}")
            return {"text": f"Erro: {str(e)}", "audio": None}
            
    async def stop(self):
        """Parar módulo WebRTC"""
        logger.info("Parando WebRTC Bridge...")
        
        self.running = False
        
        if self.process:
            self.process.terminate()
            await self.process.wait()
            
        logger.info("WebRTC Bridge parado")
        

class WebRTCModulePython:
    """
    Módulo WebRTC Python-native para integração com a arquitetura
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # ✅ Phase 3c: Dynamic port support via environment variable
        import os
        from src.config.service_config import ServiceType, get_service_port
        try:
            dynamic_port = int(os.getenv("WEBRTC_PORT") or get_service_port(ServiceType.WEBRTC_GATEWAY))
        except (OSError, ConnectionError, RuntimeError):
            dynamic_port = config.get('ws_port', 8020)  # Fallback to config or PORT_MATRIX default

        self.bridge = WebRTCBridge(
            ws_port=dynamic_port,
            ice_servers=config.get('ice_servers', None)
        )
        
    async def initialize(self):
        """Inicializar módulo"""
        await self.bridge.initialize()
        
    def set_audio_processor(self, processor):
        """Definir processador de áudio"""
        self.bridge.set_audio_processor(processor)
        
    def set_tts_module(self, tts):
        """Definir módulo TTS"""
        self.bridge.set_tts_module(tts)
        
    async def cleanup(self):
        """Limpar recursos"""
        await self.bridge.stop()