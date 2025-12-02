/**
 * WebRTC Server - Ultra-low latency audio streaming
 * Servidor age como peer WebRTC direto (não relay)
 * Latência alvo: 15-30ms
 */

const wrtc = require('wrtc');
const WebSocket = require('ws');
const path = require('path');

// Importar módulo Ultravox diretamente (sem gRPC)
const { UltravoxTransformers } = require('../ultravox/ultravox_transformers');

// Configurações
const WS_PORT = 8088;  // WebSocket para sinalização

// Instância do Ultravox
let ultravoxModule = null;

// Conexões ativas
const peers = new Map();

// Estatísticas
const stats = {
    totalConnections: 0,
    activeConnections: 0,
    totalMessages: 0,
    avgLatency: 0,
    minLatency: 999999,
    maxLatency: 0
};

/**
 * Inicializar módulo Ultravox
 */
async function initUltravox() {
    ultravoxModule = new UltravoxTransformers();
    await ultravoxModule.initialize();
    await ultravoxModule.warmup(2); // Warm-up rápido
    console.log(`✅ Módulo Ultravox inicializado e aquecido`);
}

/**
 * Processar áudio com Ultravox
 */
async function processAudioWithUltravox(audioData, sessionId) {
    if (!ultravoxModule) {
        throw new Error('Ultravox não inicializado');
    }

    const startTime = Date.now();
    
    try {
        // Processar diretamente com módulo Ultravox
        const responseText = await ultravoxModule.process_audio(audioData);
        
        const latency = Date.now() - startTime;
        updateLatencyStats(latency);
        
        console.log(`📝 Resposta Ultravox em ${latency}ms`);
        
        // TODO: Adicionar TTS aqui se necessário
        return {
            text: responseText,
            audio: null, // TODO: Adicionar síntese TTS
            latency: latency
        };
    } catch (error) {
        console.error('❌ Erro Ultravox:', error.message);
        throw error;
    }
}

/**
 * Atualizar estatísticas de latência
 */
function updateLatencyStats(latency) {
    stats.totalMessages++;
    stats.minLatency = Math.min(stats.minLatency, latency);
    stats.maxLatency = Math.max(stats.maxLatency, latency);
    stats.avgLatency = ((stats.avgLatency * (stats.totalMessages - 1)) + latency) / stats.totalMessages;
}

/**
 * Criar peer WebRTC para cada cliente
 */
async function createPeer(ws, sessionId) {
    const pc = new wrtc.RTCPeerConnection({
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' }
        ],
        // Otimizações para baixa latência
        bundlePolicy: 'max-bundle',
        rtcpMuxPolicy: 'require'
    });

    // Armazenar peer
    const peerInfo = {
        pc: pc,
        ws: ws,
        sessionId: sessionId,
        dataChannel: null,
        audioBuffer: [],
        stats: {
            packetsReceived: 0,
            bytesReceived: 0,
            lastActivity: Date.now()
        }
    };

    peers.set(sessionId, peerInfo);
    stats.activeConnections++;
    stats.totalConnections++;

    // Configurar data channel para áudio
    const dataChannel = pc.createDataChannel('audio', {
        ordered: false,           // Não ordenado (como UDP)
        maxRetransmits: 0,        // Sem retransmissão (ultra-baixa latência)
        maxPacketLifeTime: 100    // Descartar pacotes antigos (100ms)
    });

    dataChannel.onopen = () => {
        console.log(`✅ DataChannel aberto para ${sessionId}`);
        peerInfo.dataChannel = dataChannel;
    };

    dataChannel.onmessage = async (event) => {
        const startTime = Date.now();
        peerInfo.stats.packetsReceived++;
        peerInfo.stats.bytesReceived += event.data.byteLength || event.data.length;
        peerInfo.stats.lastActivity = Date.now();

        try {
            // Converter dados recebidos para Float32Array
            let audioData;
            if (event.data instanceof ArrayBuffer) {
                audioData = new Float32Array(event.data);
            } else if (typeof event.data === 'string') {
                // Se for JSON com dados de áudio
                const parsed = JSON.parse(event.data);
                audioData = new Float32Array(parsed.audio);
            } else {
                audioData = new Float32Array(event.data);
            }

            console.log(`🎤 Áudio recebido: ${audioData.length} samples via DataChannel`);

            // Processar com Ultravox
            const result = await processAudioWithUltravox(audioData, sessionId);

            // Enviar resposta via DataChannel (ultra-rápido!)
            if (dataChannel.readyState === 'open') {
                dataChannel.send(JSON.stringify({
                    type: 'response',
                    text: result.text,
                    audio: result.audio ? Array.from(result.audio) : null,
                    latency: result.latency,
                    totalLatency: Date.now() - startTime
                }));

                console.log(`⚡ Resposta enviada em ${Date.now() - startTime}ms total`);
            }

        } catch (error) {
            console.error('❌ Erro processando áudio:', error);
            if (dataChannel.readyState === 'open') {
                dataChannel.send(JSON.stringify({
                    type: 'error',
                    error: error.message
                }));
            }
        }
    };

    dataChannel.onerror = (error) => {
        console.error(`❌ Erro no DataChannel ${sessionId}:`, error);
    };

    // Configurar ICE
    pc.onicecandidate = (event) => {
        if (event.candidate) {
            ws.send(JSON.stringify({
                type: 'ice',
                candidate: event.candidate
            }));
        }
    };

    pc.onconnectionstatechange = () => {
        console.log(`📡 Estado da conexão ${sessionId}: ${pc.connectionState}`);
        if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
            cleanupPeer(sessionId);
        }
    };

    return pc;
}

/**
 * Limpar peer desconectado
 */
function cleanupPeer(sessionId) {
    const peerInfo = peers.get(sessionId);
    if (peerInfo) {
        if (peerInfo.pc) {
            peerInfo.pc.close();
        }
        peers.delete(sessionId);
        stats.activeConnections--;
        console.log(`🧹 Peer ${sessionId} removido`);
    }
}

/**
 * Servidor WebSocket para sinalização
 */
function startSignalingServer() {
    const wss = new WebSocket.Server({ port: WS_PORT });

    wss.on('connection', (ws) => {
        const sessionId = `peer_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        console.log(`🔌 Nova conexão WebSocket: ${sessionId}`);

        let pc = null;

        ws.on('message', async (message) => {
            try {
                const data = JSON.parse(message);

                switch (data.type) {
                    case 'offer':
                        // Cliente enviou oferta, criar peer e responder
                        pc = await createPeer(ws, sessionId);
                        
                        // Definir descrição remota (oferta do cliente)
                        await pc.setRemoteDescription(
                            new wrtc.RTCSessionDescription(data.offer)
                        );

                        // Criar e enviar resposta
                        const answer = await pc.createAnswer();
                        await pc.setLocalDescription(answer);

                        ws.send(JSON.stringify({
                            type: 'answer',
                            answer: pc.localDescription
                        }));

                        console.log(`✅ Resposta SDP enviada para ${sessionId}`);
                        break;

                    case 'ice':
                        // Adicionar candidato ICE
                        if (pc && data.candidate) {
                            await pc.addIceCandidate(
                                new wrtc.RTCIceCandidate(data.candidate)
                            );
                            console.log(`🧊 ICE candidate adicionado para ${sessionId}`);
                        }
                        break;

                    case 'stats':
                        // Enviar estatísticas
                        ws.send(JSON.stringify({
                            type: 'stats',
                            stats: {
                                ...stats,
                                activeConnections: peers.size
                            }
                        }));
                        break;

                    default:
                        console.log(`Mensagem desconhecida: ${data.type}`);
                }
            } catch (error) {
                console.error('❌ Erro processando mensagem:', error);
                ws.send(JSON.stringify({
                    type: 'error',
                    error: error.message
                }));
            }
        });

        ws.on('close', () => {
            console.log(`👋 Desconectado: ${sessionId}`);
            cleanupPeer(sessionId);
        });

        ws.on('error', (error) => {
            console.error(`❌ Erro WebSocket ${sessionId}:`, error);
        });
    });

    console.log(`🚀 Servidor WebRTC rodando na porta ${WS_PORT}`);
    console.log(`📡 WebSocket para sinalização: ws://localhost:${WS_PORT}`);
    console.log(`⚡ Modo: Servidor como Peer (Ultra-baixa latência)`);
    console.log(`🎯 Latência esperada: 15-30ms`);
}

/**
 * Monitor de estatísticas
 */
function startStatsMonitor() {
    setInterval(() => {
        if (stats.totalMessages > 0) {
            console.log('\n📊 === ESTATÍSTICAS ===');
            console.log(`Conexões ativas: ${peers.size}`);
            console.log(`Total de mensagens: ${stats.totalMessages}`);
            console.log(`Latência média: ${stats.avgLatency.toFixed(1)}ms`);
            console.log(`Latência mínima: ${stats.minLatency}ms`);
            console.log(`Latência máxima: ${stats.maxLatency}ms`);
            console.log('===================\n');
        }
    }, 30000); // A cada 30 segundos
}

/**
 * Inicializar servidor
 */
async function init() {
    console.log('🚀 Iniciando servidor WebRTC puro...');
    
    try {
        // Inicializar Ultravox
        await initUltravox();

        // Iniciar servidor de sinalização
        startSignalingServer();

        // Iniciar monitor de stats
        startStatsMonitor();

        console.log('✅ Servidor WebRTC pronto!');
        console.log('🎯 Características:');
        console.log('   • Node WebRTC puro (sem frameworks)');
        console.log('   • Servidor age como peer direto');
        console.log('   • DataChannel não-ordenado (UDP-like)');
        console.log('   • Sem retransmissão (máxima velocidade)');
        console.log('   • Integração direta com Ultravox via gRPC');

    } catch (error) {
        console.error('❌ Erro iniciando servidor:', error);
        process.exit(1);
    }
}

// Tratamento de saída limpa
process.on('SIGINT', () => {
    console.log('\n🛑 Encerrando servidor...');
    peers.forEach((peer, id) => cleanupPeer(id));
    process.exit(0);
});

// Iniciar!
init();