/**
 * WebRTC Module - Classe modular para integração
 * Ultra-baixa latência com WebRTC puro
 */

const wrtc = require('wrtc');
const WebSocket = require('ws');
const EventEmitter = require('events');

class WebRTCModule extends EventEmitter {
    constructor(config = {}) {
        super();
        
        // Configurações
        this.config = {
            wsPort: config.wsPort || 8088,
            iceServers: config.iceServers || [
                { urls: 'stun:stun.l.google.com:19302' }
            ],
            maxConnections: config.maxConnections || 100,
            ...config
        };

        // Estado
        this.wss = null;
        this.peers = new Map();
        this.audioProcessor = null;
        this.ttsModule = null;
        this.initialized = false;

        // Estatísticas
        this.stats = {
            totalConnections: 0,
            activeConnections: 0,
            totalMessages: 0,
            avgLatency: 0,
            minLatency: 999999,
            maxLatency: 0
        };
    }

    /**
     * Definir processador de áudio (Ultravox)
     */
    setAudioProcessor(processor) {
        this.audioProcessor = processor;
        console.log('✅ Processador de áudio configurado');
    }

    /**
     * Definir módulo TTS
     */
    setTTSModule(tts) {
        this.ttsModule = tts;
        console.log('✅ Módulo TTS configurado');
    }

    /**
     * Inicializar módulo
     */
    async initialize() {
        if (this.initialized) {
            console.warn('⚠️ Módulo já inicializado');
            return;
        }

        console.log('🚀 Inicializando módulo WebRTC...');

        // Criar servidor WebSocket para sinalização
        this.wss = new WebSocket.Server({ 
            port: this.config.wsPort 
        });

        this.wss.on('connection', (ws) => {
            this.handleNewConnection(ws);
        });

        this.initialized = true;

        console.log(`✅ Módulo WebRTC inicializado na porta ${this.config.wsPort}`);
        console.log(`📡 WebSocket: ws://localhost:${this.config.wsPort}`);
        console.log(`⚡ Modo: Ultra-baixa latência com DataChannel`);

        // Emitir evento
        this.emit('ready', {
            port: this.config.wsPort,
            maxConnections: this.config.maxConnections
        });
    }

    /**
     * Lidar com nova conexão
     */
    handleNewConnection(ws) {
        const sessionId = this.generateSessionId();
        console.log(`🔌 Nova conexão: ${sessionId}`);

        let pc = null;
        let dataChannel = null;

        ws.on('message', async (message) => {
            try {
                const data = JSON.parse(message);

                switch (data.type) {
                    case 'offer':
                        // Criar peer e responder
                        const result = await this.createPeerConnection(sessionId, ws, data.offer);
                        pc = result.pc;
                        dataChannel = result.dataChannel;

                        // Enviar resposta
                        ws.send(JSON.stringify({
                            type: 'answer',
                            answer: pc.localDescription
                        }));
                        break;

                    case 'ice':
                        // Adicionar candidato ICE
                        if (pc && data.candidate) {
                            await pc.addIceCandidate(
                                new wrtc.RTCIceCandidate(data.candidate)
                            );
                        }
                        break;

                    case 'stats':
                        // Enviar estatísticas
                        ws.send(JSON.stringify({
                            type: 'stats',
                            stats: this.getStats()
                        }));
                        break;
                }
            } catch (error) {
                console.error('❌ Erro:', error);
                ws.send(JSON.stringify({
                    type: 'error',
                    error: error.message
                }));
            }
        });

        ws.on('close', () => {
            this.cleanupPeer(sessionId);
        });
    }

    /**
     * Criar conexão peer
     */
    async createPeerConnection(sessionId, ws, offer) {
        const pc = new wrtc.RTCPeerConnection({
            iceServers: this.config.iceServers,
            bundlePolicy: 'max-bundle',
            rtcpMuxPolicy: 'require'
        });

        // Criar data channel
        const dataChannel = pc.createDataChannel('audio', {
            ordered: false,
            maxRetransmits: 0,
            maxPacketLifeTime: 100
        });

        // Configurar handlers
        dataChannel.onopen = () => {
            console.log(`✅ DataChannel aberto: ${sessionId}`);
            this.emit('peer-connected', sessionId);
        };

        dataChannel.onmessage = async (event) => {
            await this.handleAudioData(sessionId, event.data, dataChannel);
        };

        // ICE candidates
        pc.onicecandidate = (event) => {
            if (event.candidate) {
                ws.send(JSON.stringify({
                    type: 'ice',
                    candidate: event.candidate
                }));
            }
        };

        // Definir oferta e criar resposta
        await pc.setRemoteDescription(
            new wrtc.RTCSessionDescription(offer)
        );
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);

        // Armazenar peer
        this.peers.set(sessionId, {
            pc,
            ws,
            dataChannel,
            stats: {
                packetsReceived: 0,
                lastActivity: Date.now()
            }
        });

        this.stats.activeConnections++;
        this.stats.totalConnections++;

        return { pc, dataChannel };
    }

    /**
     * Processar dados de áudio
     */
    async handleAudioData(sessionId, data, dataChannel) {
        const startTime = Date.now();

        try {
            // Converter dados para Float32Array
            let audioData;
            if (data instanceof ArrayBuffer) {
                audioData = new Float32Array(data);
            } else if (typeof data === 'string') {
                const parsed = JSON.parse(data);
                audioData = new Float32Array(parsed.audio);
            } else {
                audioData = new Float32Array(data);
            }

            console.log(`🎤 Áudio recebido: ${audioData.length} samples`);

            // Processar com módulo de áudio
            let responseText = '';
            let responseAudio = null;

            if (this.audioProcessor) {
                responseText = await this.audioProcessor.processAudio(audioData, sessionId);
                
                // Gerar áudio de resposta com TTS
                if (this.ttsModule && responseText) {
                    responseAudio = await this.ttsModule.synthesize(responseText);
                }
            } else {
                responseText = 'Processador de áudio não configurado';
            }

            const latency = Date.now() - startTime;
            this.updateStats(latency);

            // Enviar resposta via DataChannel
            if (dataChannel.readyState === 'open') {
                dataChannel.send(JSON.stringify({
                    type: 'response',
                    text: responseText,
                    audio: responseAudio ? Array.from(responseAudio) : null,
                    latency: latency
                }));

                console.log(`⚡ Resposta em ${latency}ms`);
            }

            // Emitir evento
            this.emit('audio-processed', {
                sessionId,
                latency,
                responseLength: responseText.length
            });

        } catch (error) {
            console.error('❌ Erro processando áudio:', error);
            if (dataChannel.readyState === 'open') {
                dataChannel.send(JSON.stringify({
                    type: 'error',
                    error: error.message
                }));
            }
        }
    }

    /**
     * Limpar peer
     */
    cleanupPeer(sessionId) {
        const peer = this.peers.get(sessionId);
        if (peer) {
            if (peer.pc) {
                peer.pc.close();
            }
            this.peers.delete(sessionId);
            this.stats.activeConnections--;
            console.log(`🧹 Peer removido: ${sessionId}`);
            this.emit('peer-disconnected', sessionId);
        }
    }

    /**
     * Atualizar estatísticas
     */
    updateStats(latency) {
        this.stats.totalMessages++;
        this.stats.minLatency = Math.min(this.stats.minLatency, latency);
        this.stats.maxLatency = Math.max(this.stats.maxLatency, latency);
        this.stats.avgLatency = 
            ((this.stats.avgLatency * (this.stats.totalMessages - 1)) + latency) 
            / this.stats.totalMessages;
    }

    /**
     * Obter estatísticas
     */
    getStats() {
        return {
            ...this.stats,
            activeConnections: this.peers.size,
            uptime: this.initialized ? Date.now() - this.startTime : 0
        };
    }

    /**
     * Gerar ID de sessão
     */
    generateSessionId() {
        return `peer_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Parar módulo
     */
    async stop() {
        console.log('🛑 Parando módulo WebRTC...');

        // Fechar todas as conexões
        this.peers.forEach((peer, id) => {
            this.cleanupPeer(id);
        });

        // Fechar WebSocket server
        if (this.wss) {
            this.wss.close();
        }

        this.initialized = false;
        this.emit('stopped');
        console.log('✅ Módulo WebRTC parado');
    }
}

module.exports = WebRTCModule;