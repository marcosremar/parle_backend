#!/bin/bash

cd /Users/marcos/Documents/projects/backend/parle_backend

echo "=== ATIVANDO AMBIENTE CONDA ==="
./activate_conda.sh

echo ""
echo "=== INICIANDO SERVIÇOS DIRETAMENTE ==="
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "1. Iniciando TTS..."
nohup python src/services/tts/app_complete.py > /tmp/tts.log 2>&1 &
TTS_PID=$!
echo "TTS PID: $TTS_PID"
sleep 3

echo "2. Iniciando STT..."
nohup python src/services/stt/app_complete.py > /tmp/stt.log 2>&1 &
STT_PID=$!
echo "STT PID: $STT_PID"
sleep 3

echo "3. Iniciando Orchestrator..."
nohup python src/services/orchestrator/app_complete.py > /tmp/orchestrator.log 2>&1 &
ORCH_PID=$!
echo "Orchestrator PID: $ORCH_PID"
sleep 3

echo "4. Iniciando LLM..."
nohup python src/services/llm/app_complete.py > /tmp/llm.log 2>&1 &
LLM_PID=$!
echo "LLM PID: $LLM_PID"
sleep 5

echo ""
echo "=== VERIFICANDO PROCESSOS ==="
ps aux | grep -E "(uvicorn|python.*app_complete)" | grep -v grep

echo ""
echo "=== TESTANDO CONEXÕES ==="
echo "TTS (5000):"
curl -s http://localhost:5000/api/health | head -1 || echo "❌ Não responde"
echo "STT (9000):"
curl -s http://localhost:9000/api/health | head -1 || echo "❌ Não responde"
echo "Orchestrator (8080):"
curl -s http://localhost:8080/api/health | head -1 || echo "❌ Não responde"
echo "LLM (8100):"
curl -s http://localhost:8100/api/health | head -1 || echo "❌ Não responde"

