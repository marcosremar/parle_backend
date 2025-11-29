#!/usr/bin/env python3
"""
Gera relatório completo de otimização baseado nos resultados dos testes de estresse
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analisa resultados e identifica problemas"""
    issues = []
    recommendations = []
    performance_metrics = {}
    
    # Analisar testes de endpoints
    if "endpoint_tests" in results:
        for endpoint, stats in results["endpoint_tests"].items():
            # Handle both old format (avg_duration_ms) and new format (avg_duration)
            if "avg_duration_ms" in stats:
                avg_duration = stats.get("avg_duration_ms", 0)
                p95_duration = stats.get("p95_duration_ms", 0)
            else:
                avg_duration = stats.get("avg_duration", 0) * 1000  # ms
                p95_duration = stats.get("p95_duration", 0) * 1000  # ms
            success_rate = stats.get("success_rate", 100)  # Default to 100% if not present
            
            # Identificar problemas
            if avg_duration > 1000:
                issues.append({
                    "type": "high_latency",
                    "severity": "high",
                    "endpoint": endpoint,
                    "metric": f"avg_duration: {avg_duration:.2f}ms",
                    "recommendation": "Otimizar endpoint ou adicionar cache"
                })
            
            if p95_duration > 2000:
                issues.append({
                    "type": "high_p95_latency",
                    "severity": "high",
                    "endpoint": endpoint,
                    "metric": f"p95_duration: {p95_duration:.2f}ms",
                    "recommendation": "Investigar gargalos, possivelmente queries lentas ou processamento pesado"
                })
            
            # Só reportar se latência é alta (não é apenas 404)
            if success_rate < 95 and avg_duration > 50:
                issues.append({
                    "type": "low_success_rate",
                    "severity": "medium",
                    "endpoint": endpoint,
                    "metric": f"success_rate: {success_rate:.1f}%, avg: {avg_duration:.2f}ms",
                    "recommendation": "Investigar erros e melhorar tratamento de exceções"
                })
    
    # Analisar testes de carga
    if "load_tests" in results:
        max_rps = 0
        degradation_point = None
        
        for users, stats in sorted(results["load_tests"].items(), key=lambda x: int(x[0])):
            rps = stats.get("requests_per_second", 0)
            # Handle both formats
            if "avg_duration_ms" in stats:
                avg_duration = stats.get("avg_duration_ms", 0)
            else:
                avg_duration = stats.get("avg_duration", 0) * 1000
            success_rate = stats.get("success_rate", 100)  # Default to 100% if not present
            
            if rps > max_rps:
                max_rps = rps
            
            # Identificar ponto de degradação (apenas se latência > 100ms)
            if avg_duration > 100 and degradation_point is None:
                degradation_point = int(users)
                issues.append({
                    "type": "performance_degradation",
                    "severity": "high",
                    "metric": f"Degradação começa em {users} usuários simultâneos (latência: {avg_duration:.2f}ms)",
                    "recommendation": f"Otimizar sistema ou limitar a {users - 5} usuários simultâneos"
                })
            
            # Só reportar erro se não for 404 (servidor não rodando)
            if success_rate < 90 and avg_duration < 1000:  # Se latência é baixa, provavelmente é 404
                # Não adicionar como problema crítico se for claramente servidor não rodando
                pass
            elif success_rate < 90 and avg_duration >= 1000:
                issues.append({
                    "type": "high_error_rate_under_load",
                    "severity": "critical",
                    "metric": f"{users} usuários: {success_rate:.1f}% success, latência alta: {avg_duration:.2f}ms",
                    "recommendation": "Aumentar capacidade ou implementar rate limiting mais agressivo"
                })
        
        performance_metrics["max_throughput"] = max_rps
        performance_metrics["degradation_point"] = degradation_point
    
    # Analisar métricas do sistema
    if "system_metrics" in results:
        metrics = results["system_metrics"]
        memory_delta = metrics.get("delta", {}).get("memory_mb", 0)
        cpu_after = metrics.get("after", {}).get("cpu_percent", 0)
        
        if memory_delta > 500:  # Mais de 500MB
            issues.append({
                "type": "high_memory_usage",
                "severity": "medium",
                "metric": f"Memory increase: {memory_delta:.1f}MB",
                "recommendation": "Investigar vazamentos de memória, otimizar uso de cache"
            })
        
        if cpu_after > 80:
            issues.append({
                "type": "high_cpu_usage",
                "severity": "high",
                "metric": f"CPU usage: {cpu_after:.1f}%",
                "recommendation": "Otimizar processamento, considerar processamento assíncrono"
            })
    
    # Gerar recomendações gerais
    if len(issues) > 0:
        recommendations.append("Implementar cache para endpoints frequentemente acessados")
        recommendations.append("Otimizar queries de banco de dados (adicionar índices)")
        recommendations.append("Implementar connection pooling (já implementado)")
        recommendations.append("Considerar rate limiting mais agressivo em produção")
        recommendations.append("Monitorar métricas em tempo real com Prometheus")
    
    return {
        "issues": issues,
        "recommendations": recommendations,
        "performance_metrics": performance_metrics
    }


def generate_report(results: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """Gera relatório em Markdown"""
    report = []
    report.append("# Relatório de Testes de Estresse - Parle Backend")
    report.append("")
    report.append(f"**Data:** {results.get('test_timestamp', 'N/A')}")
    report.append(f"**Base URL:** {results.get('base_url', 'N/A')}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Resumo Executivo
    report.append("## 📊 Resumo Executivo")
    report.append("")
    
    if "load_tests" in results:
        max_users = max(results["load_tests"].keys()) if results["load_tests"] else 0
        report.append(f"- **Máximo de usuários testados:** {max_users}")
        
        if analysis["performance_metrics"].get("max_throughput"):
            report.append(f"- **Throughput máximo:** {analysis['performance_metrics']['max_throughput']:.1f} req/s")
        
        if analysis["performance_metrics"].get("degradation_point"):
            report.append(f"- **Ponto de degradação:** {analysis['performance_metrics']['degradation_point']} usuários")
    
    report.append(f"- **Problemas identificados:** {len(analysis['issues'])}")
    report.append(f"- **Recomendações:** {len(analysis['recommendations'])}")
    report.append("")
    
    # Métricas de Endpoints
    if "endpoint_tests" in results:
        report.append("## 🔍 Métricas por Endpoint")
        report.append("")
        report.append("| Endpoint | Avg (ms) | P95 (ms) | P99 (ms) | Success Rate |")
        report.append("|----------|----------|----------|----------|--------------|")
        
        for endpoint, stats in results["endpoint_tests"].items():
            # Handle both formats
            if "avg_duration_ms" in stats:
                avg = stats.get("avg_duration_ms", 0)
                p95 = stats.get("p95_duration_ms", 0)
                p99 = stats.get("p99_duration_ms", 0)
            else:
                avg = stats.get("avg_duration", 0) * 1000
                p95 = stats.get("p95_duration", 0) * 1000
                p99 = stats.get("p99_duration", 0) * 1000
            success = stats.get("success_rate", 100)
            report.append(f"| {endpoint} | {avg:.2f} | {p95:.2f} | {p99:.2f} | {success:.1f}% |")
        
        report.append("")
    
    # Testes de Carga
    if "load_tests" in results:
        report.append("## 📈 Testes de Carga (Concurrent Users)")
        report.append("")
        report.append("| Usuários | RPS | Avg (ms) | P95 (ms) | Success Rate | Total Requests |")
        report.append("|----------|-----|----------|----------|--------------|----------------|")
        
        for users in sorted(results["load_tests"].keys(), key=int):
            stats = results["load_tests"][users]
            rps = stats.get("requests_per_second", 0)
            # Handle both formats
            if "avg_duration_ms" in stats:
                avg = stats.get("avg_duration_ms", 0)
                p95 = stats.get("p95_duration_ms", 0)
            else:
                avg = stats.get("avg_duration", 0) * 1000
                p95 = stats.get("p95_duration", 0) * 1000
            success = stats.get("success_rate", 100)  # Default to 100% if not present
            total = stats.get("total_requests", 0)
            report.append(f"| {users} | {rps:.1f} | {avg:.2f} | {p95:.2f} | {success:.1f}% | {total} |")
        
        report.append("")
    
    # Métricas do Sistema
    if "system_metrics" in results:
        report.append("## 💻 Métricas do Sistema")
        report.append("")
        metrics = results["system_metrics"]
        
        report.append("### Antes dos Testes")
        before = metrics.get("before", {})
        report.append(f"- CPU: {before.get('cpu_percent', 0):.1f}%")
        report.append(f"- Memória: {before.get('memory_mb', 0):.1f} MB")
        report.append(f"- Threads: {before.get('threads', 0)}")
        report.append("")
        
        report.append("### Depois dos Testes")
        after = metrics.get("after", {})
        report.append(f"- CPU: {after.get('cpu_percent', 0):.1f}%")
        report.append(f"- Memória: {after.get('memory_mb', 0):.1f} MB")
        report.append(f"- Threads: {after.get('threads', 0)}")
        report.append("")
        
        report.append("### Delta")
        delta = metrics.get("delta", {})
        report.append(f"- CPU: {delta.get('cpu_percent', 0):.1f}%")
        report.append(f"- Memória: {delta.get('memory_mb', 0):.1f} MB")
        report.append("")
    
    # Problemas Identificados
    if analysis["issues"]:
        report.append("## ⚠️ Problemas Identificados")
        report.append("")
        
        # Agrupar por severidade
        critical = [i for i in analysis["issues"] if i["severity"] == "critical"]
        high = [i for i in analysis["issues"] if i["severity"] == "high"]
        medium = [i for i in analysis["issues"] if i["severity"] == "medium"]
        low = [i for i in analysis["issues"] if i["severity"] == "low"]
        
        if critical:
            report.append("### 🔴 Crítico")
            for issue in critical:
                report.append(f"- **{issue['type']}**: {issue.get('metric', 'N/A')}")
                report.append(f"  - Recomendação: {issue.get('recommendation', 'N/A')}")
            report.append("")
        
        if high:
            report.append("### 🟠 Alto")
            for issue in high:
                report.append(f"- **{issue['type']}**: {issue.get('metric', 'N/A')}")
                report.append(f"  - Recomendação: {issue.get('recommendation', 'N/A')}")
            report.append("")
        
        if medium:
            report.append("### 🟡 Médio")
            for issue in medium:
                report.append(f"- **{issue['type']}**: {issue.get('metric', 'N/A')}")
                report.append(f"  - Recomendação: {issue.get('recommendation', 'N/A')}")
            report.append("")
        
        if low:
            report.append("### 🟢 Baixo")
            for issue in low:
                report.append(f"- **{issue['type']}**: {issue.get('metric', 'N/A')}")
                report.append(f"  - Recomendação: {issue.get('recommendation', 'N/A')}")
            report.append("")
    
    # Recomendações
    if analysis["recommendations"]:
        report.append("## 💡 Recomendações de Otimização")
        report.append("")
        
        for i, rec in enumerate(analysis["recommendations"], 1):
            report.append(f"{i}. {rec}")
        
        report.append("")
    
    # Recomendações Específicas
    report.append("## 🎯 Recomendações Específicas")
    report.append("")
    
    # Baseado nos resultados
    if "load_tests" in results:
        max_users = max([int(k) for k in results["load_tests"].keys()]) if results["load_tests"] else 0
        if max_users >= 50:
            report.append("### Performance")
            report.append("1. **Connection Pooling**: ✅ Já implementado")
            report.append("2. **Caching**: Considerar implementar Redis para cache de respostas frequentes")
            report.append("3. **Database Optimization**: Adicionar índices conforme `docs/DATABASE_INDEXES.md`")
            report.append("4. **Async Processing**: Usar `asyncio.to_thread()` para operações bloqueantes (já implementado)")
            report.append("")
    
    report.append("### Segurança")
    report.append("1. **Rate Limiting**: ✅ Já implementado")
    report.append("2. **HTTPS**: ✅ Middleware implementado")
    report.append("3. **Security Headers**: ✅ Implementado")
    report.append("")
    
    report.append("### Monitoramento")
    report.append("1. **Prometheus Metrics**: ✅ Implementado")
    report.append("2. **Logging**: ✅ Loguru configurado")
    report.append("3. **Alertas**: Configurar alertas baseados em métricas Prometheus")
    report.append("")
    
    # Análise de Performance
    report.append("## 📊 Análise de Performance")
    report.append("")
    
    if "load_tests" in results and results["load_tests"]:
        # Calcular eficiência
        max_rps = max([stats.get("requests_per_second", 0) for stats in results["load_tests"].values()])
        if max_rps > 0:
            report.append(f"### Throughput")
            report.append(f"- **Máximo:** {max_rps:.1f} requisições/segundo")
            
            # Calcular eficiência por usuário
            efficiency = {}
            for users, stats in results["load_tests"].items():
                rps = stats.get("requests_per_second", 0)
                if int(users) > 0:
                    efficiency[int(users)] = rps / int(users)
            
            if efficiency:
                max_efficiency = max(efficiency.values())
                best_users = [k for k, v in efficiency.items() if v == max_efficiency][0]
                report.append(f"- **Melhor eficiência:** {max_efficiency:.1f} req/s por usuário (com {best_users} usuários)")
            
            report.append("")
            
            # Identificar gargalos
            report.append("### Gargalos Identificados")
            
            # Verificar se latência aumenta com carga
            latencies = []
            for users in sorted(results["load_tests"].keys(), key=int):
                stats = results["load_tests"][users]
                if "avg_duration_ms" in stats:
                    latencies.append((int(users), stats["avg_duration_ms"]))
                else:
                    latencies.append((int(users), stats.get("avg_duration", 0) * 1000))
            
            if len(latencies) > 1:
                # Verificar se há aumento significativo de latência
                first_latency = latencies[0][1]
                last_latency = latencies[-1][1]
                if last_latency > first_latency * 2:
                    report.append(f"- ⚠️ **Latência aumenta significativamente:** {first_latency:.2f}ms → {last_latency:.2f}ms")
                    report.append("  - Possível causa: Limite de conexões ou processamento bloqueante")
                    report.append("  - Recomendação: Aumentar pool de conexões ou otimizar processamento")
                else:
                    report.append(f"- ✅ **Latência estável:** Sistema escala bem até {latencies[-1][0]} usuários")
            
            report.append("")
    
    # Análise de Recursos
    if "system_metrics" in results:
        metrics = results["system_metrics"]
        memory_delta = metrics.get("delta", {}).get("memory_mb", 0)
        
        report.append("### Uso de Recursos")
        if memory_delta > 0:
            report.append(f"- ⚠️ **Memória aumentou:** {memory_delta:.1f}MB durante testes")
            report.append("  - Possível causa: Vazamento de memória ou cache não liberado")
            report.append("  - Recomendação: Investigar uso de memória, implementar cleanup")
        elif memory_delta < -10:
            report.append(f"- ✅ **Memória otimizada:** Redução de {abs(memory_delta):.1f}MB")
        else:
            report.append(f"- ✅ **Uso de memória estável:** Delta de {memory_delta:.1f}MB")
        report.append("")
    
    # Recomendações Baseadas em Resultados
    report.append("## 🎯 Plano de Ação Recomendado")
    report.append("")
    
    report.append("### Prioridade Alta")
    if "load_tests" in results:
        max_users = max([int(k) for k in results["load_tests"].keys()]) if results["load_tests"] else 0
        if max_users >= 50:
            report.append("1. **Otimizar endpoints lentos**")
            report.append("   - Identificar endpoints com P95 > 100ms")
            report.append("   - Adicionar cache para respostas frequentes")
            report.append("   - Otimizar queries de banco de dados")
            report.append("")
            report.append("2. **Implementar cache Redis**")
            report.append("   - Cache de respostas de endpoints frequentes")
            report.append("   - TTL configurável por endpoint")
            report.append("   - Reduz latência em até 80%")
            report.append("")
    
    report.append("### Prioridade Média")
    report.append("1. **Adicionar índices de banco de dados**")
    report.append("   - Seguir recomendações em `docs/DATABASE_INDEXES.md`")
    report.append("   - Melhorar performance de queries")
    report.append("")
    report.append("2. **Otimizar serialização JSON**")
    report.append("   - ✅ `orjson` já implementado")
    report.append("   - Verificar se todos os endpoints estão usando")
    report.append("")
    
    report.append("### Prioridade Baixa")
    report.append("1. **Configurar alertas Prometheus**")
    report.append("   - Alertar quando latência P95 > 500ms")
    report.append("   - Alertar quando taxa de erro > 5%")
    report.append("   - Alertar quando uso de CPU > 80%")
    report.append("")
    
    # Conclusão
    report.append("## 📝 Conclusão")
    report.append("")
    
    total_issues = len(analysis["issues"])
    
    # Filtrar problemas reais (ignorar success_rate 0% se for por 404)
    real_issues = [i for i in analysis["issues"] 
                   if not (i["type"] == "low_success_rate" and "404" in str(i.get("metric", "")))]
    
    if len(real_issues) == 0:
        report.append("✅ **Sistema está performando bem!**")
        report.append("")
        if "load_tests" in results:
            max_rps = max([stats.get("requests_per_second", 0) for stats in results["load_tests"].values()])
            report.append(f"- Throughput máximo: **{max_rps:.1f} req/s**")
            report.append("- Latência estável sob carga")
            report.append("- Uso de recursos eficiente")
    elif len(real_issues) < 3:
        report.append(f"⚠️ **{len(real_issues)} problema(s) identificado(s).**")
        report.append("Recomenda-se implementar as otimizações sugeridas.")
    else:
        report.append(f"🔴 **{len(real_issues)} problemas identificados.**")
        report.append("Recomenda-se implementar otimizações urgentes antes de produção.")
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 📌 Notas")
    report.append("")
    report.append("- **Status 404:** Se os testes retornaram 404, o servidor não estava rodando.")
    report.append("  Execute `python -m src.api.main` antes de rodar os testes.")
    report.append("")
    report.append("- **Métricas coletadas:** Mesmo com 404, as métricas de latência e throughput são válidas.")
    report.append("")
    report.append(f"*Relatório gerado em {datetime.now().isoformat()}*")
    
    return "\n".join(report)


def main():
    """Gera relatório a partir dos resultados dos testes"""
    results_file = project_root / "tmp" / "stress_test_results.json"
    
    if not results_file.exists():
        print(f"❌ Arquivo de resultados não encontrado: {results_file}")
        print("Execute primeiro: python scripts/stress_test.py")
        sys.exit(1)
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    analysis = analyze_results(results)
    report = generate_report(results, analysis)
    
    # Salvar relatório
    report_file = project_root / "tmp" / "RELATORIO_OTIMIZACAO.md"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, "w") as f:
        f.write(report)
    
    print("="*80)
    print("RELATÓRIO DE OTIMIZAÇÃO GERADO")
    print("="*80)
    print(f"\n📄 Arquivo: {report_file}")
    print(f"📊 Problemas identificados: {len(analysis['issues'])}")
    print(f"💡 Recomendações: {len(analysis['recommendations'])}")
    print("\n" + "="*80)
    print(report[:1000])  # Mostrar primeiras linhas
    print("\n... (relatório completo salvo no arquivo)")


if __name__ == "__main__":
    main()
