#!/bin/bash

echo "=== 🛑 Parando todos os serviços ==="
echo ""

# Parar por PID se existir
if [ -f /tmp/service_pids.txt ]; then
    while IFS=':' read -r service_name pid; do
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Parando $service_name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
        fi
    done < /tmp/service_pids.txt
    rm -f /tmp/service_pids.txt
fi

# Parar por nome de processo (excluindo _fase-two)
for service in orchestrator api_gateway websocket session conversation_store conversation_history database file_storage user rest_polling scenarios neural_codec webrtc webrtc_signaling viber_gateway; do
    pkill -f "python.*${service}/app_complete" 2>/dev/null || true
done

echo ""
echo "✅ Todos os serviços foram parados"
echo ""

