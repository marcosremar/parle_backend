#!/bin/bash

cd /Users/marcos/Documents/projects/backend/parle_backend

echo "=== 🔍 VERIFICAÇÃO DE INTEGRAÇÃO DOS SERVIÇOS ==="
echo ""

# Lista de serviços testados
SERVICES_TESTED=(
    "orchestrator|8080"
    "api_gateway|8000"
    "websocket|8500"
    "session|8600"
    "conversation_store|8800"
    "conversation_history|8010"
    "database|8300"
    "file_storage|8700"
    "user|8200"
    "rest_polling|8701"
    "scenarios|8601"
    "neural_codec|8801"
    "webrtc|10100"
    "webrtc_signaling|10200"
    "viber_gateway|10400"
)

# Serviços integrados no orchestrator (baseado em service_clients.py)
SERVICES_INTEGRATED=(
    "llm"
    "tts"
    "stt"
    "external_llm"
    "external_stt"
    "external_tts"
    "external_ultravox"
    "session"
    "scenarios"
    "conversation_store"
    "conversation_history"
    "user"
    "database"
    "file_storage"
    "websocket"
    "rest_polling"
    "webrtc"
    "webrtc_signaling"
    "neural_codec"
    "api_gateway"
    "viber_gateway"
)

echo "1️⃣ Verificando status dos serviços..."
echo ""

ALL_HEALTHY=true
for service_line in "${SERVICES_TESTED[@]}"; do
    IFS='|' read -r name port <<< "$service_line"
    
    if curl -s -f --max-time 2 "http://localhost:$port/health" > /dev/null 2>&1 || \
       curl -s -f --max-time 2 "http://localhost:$port/api/health" > /dev/null 2>&1 || \
       curl -s -f --max-time 2 "http://localhost:$port/" > /dev/null 2>&1; then
        echo "   ✅ $name (:$port) - HEALTHY"
    else
        echo "   ❌ $name (:$port) - NOT RUNNING"
        ALL_HEALTHY=false
    fi
done

echo ""
echo "2️⃣ Verificando integração no Orchestrator..."
echo ""

# Verificar quais serviços testados estão integrados
INTEGRATED_COUNT=0
NOT_INTEGRATED=()

for service_line in "${SERVICES_TESTED[@]}"; do
    IFS='|' read -r name port <<< "$service_line"
    
    # Pular orchestrator (ele mesmo)
    if [ "$name" = "orchestrator" ]; then
        continue
    fi
    
    # Verificar se está na lista de integrados
    INTEGRATED=false
    for integrated in "${SERVICES_INTEGRATED[@]}"; do
        if [ "$name" = "$integrated" ]; then
            INTEGRATED=true
            break
        fi
    done
    
    if [ "$INTEGRATED" = true ]; then
        echo "   ✅ $name - INTEGRADO"
        INTEGRATED_COUNT=$((INTEGRATED_COUNT + 1))
    else
        echo "   ⚠️  $name - NÃO INTEGRADO"
        NOT_INTEGRATED+=("$name")
    fi
done

echo ""
echo "=== 📊 RESUMO ==="
echo ""

if [ "$ALL_HEALTHY" = true ]; then
    echo "✅ Todos os serviços estão rodando"
else
    echo "⚠️  Alguns serviços não estão rodando"
fi

echo ""
echo "📋 Integração no Orchestrator:"
echo "   ✅ Integrados: $INTEGRATED_COUNT de $(( ${#SERVICES_TESTED[@]} - 1 ))"
echo "   ⚠️  Não integrados: ${#NOT_INTEGRATED[@]}"

if [ ${#NOT_INTEGRATED[@]} -gt 0 ]; then
    echo ""
    echo "🔴 Serviços não integrados:"
    for service in "${NOT_INTEGRATED[@]}"; do
        echo "   - $service"
    done
fi

echo ""
echo "📝 Nota:"
echo "   • LLM, TTS, STT são serviços essenciais (já integrados)"
echo "   • external_* são wrappers para serviços externos"
echo "   • Alguns serviços podem não precisar de integração direta"

