#!/usr/bin/env python3
"""
Script para verificar serviços migrados - testa imports e estrutura básica
"""
import sys
from pathlib import Path

PARLE_DIR = Path("/Users/marcos/Documents/projects/backend/parle_backend")
sys.path.insert(0, str(PARLE_DIR))

# Serviços migrados
MIGRATED_SERVICES = [
    "orchestrator", "session", "rest_polling", "conversation_store",
    "tts", "stt", "diarization", "vad_service", "sentiment_analysis",
    "broadcaster", "communication_strategy", "group_orchestrator",
    "group_session", "metrics_testing", "runpod_llm",
    "streaming_orchestrator", "webrtc", "webrtc_signaling",
    "discord_voice", "viber_gateway", "whatsapp_gateway"
]

def check_service(service_name: str):
    """Verifica estrutura básica de um serviço"""
    service_dir = PARLE_DIR / "src" / "services" / service_name
    
    results = {
        "name": service_name,
        "exists": service_dir.exists(),
        "has_app_complete": (service_dir / "app_complete.py").exists(),
        "has_service": (service_dir / "service.py").exists(),
        "has_routes": (service_dir / "routes.py").exists(),
        "has_config": (service_dir / "config.py").exists(),
        "import_error": None
    }
    
    # Tentar importar app_complete
    if results["has_app_complete"]:
        try:
            # Apenas verificar sintaxe
            compile((service_dir / "app_complete.py").read_text(), 
                   str(service_dir / "app_complete.py"), 'exec')
            results["syntax_ok"] = True
        except SyntaxError as e:
            results["syntax_ok"] = False
            results["import_error"] = f"SyntaxError: {e}"
        except Exception as e:
            results["syntax_ok"] = False
            results["import_error"] = f"Error: {e}"
    
    return results

def main():
    """Verifica todos os serviços migrados"""
    print("🔍 Verificando serviços migrados...\n")
    
    all_ok = []
    has_issues = []
    
    for service in MIGRATED_SERVICES:
        result = check_service(service)
        
        if result["exists"] and result["has_app_complete"] and result.get("syntax_ok", False):
            all_ok.append(service)
            print(f"✅ {service:25} - OK")
        else:
            has_issues.append(result)
            status = []
            if not result["exists"]:
                status.append("❌ não existe")
            if not result["has_app_complete"]:
                status.append("❌ sem app_complete.py")
            if not result.get("syntax_ok", True):
                status.append(f"❌ erro: {result['import_error']}")
            
            print(f"⚠️  {service:25} - {' | '.join(status)}")
    
    print(f"\n📊 Resumo:")
    print(f"   ✅ OK: {len(all_ok)}/{len(MIGRATED_SERVICES)}")
    print(f"   ⚠️  Com problemas: {len(has_issues)}/{len(MIGRATED_SERVICES)}")
    
    if has_issues:
        print(f"\n⚠️  Serviços com problemas:")
        for issue in has_issues:
            print(f"   • {issue['name']}")

if __name__ == "__main__":
    main()

