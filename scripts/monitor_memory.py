#!/usr/bin/env python3
"""
Monitor de memória para MacBook M1 8GB
Uso: python scripts/monitor_memory.py
"""

import psutil
import os
import sys
from datetime import datetime
from typing import List, Dict


def format_bytes(bytes_value: int) -> str:
    """Formata bytes em formato legível"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_memory_info() -> Dict:
    """Obtém informações de memória do sistema"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'ram_total': mem.total,
        'ram_used': mem.used,
        'ram_available': mem.available,
        'ram_percent': mem.percent,
        'swap_total': swap.total,
        'swap_used': swap.used,
        'swap_percent': swap.percent,
    }


def get_top_processes(limit: int = 10) -> List[Dict]:
    """Obtém os processos que mais usam memória"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'memory_percent']):
        try:
            pinfo = proc.info
            processes.append({
                'pid': pinfo['pid'],
                'name': pinfo['name'],
                'memory_mb': pinfo['memory_info'].rss / (1024**2),
                'memory_percent': pinfo['memory_percent']
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    processes.sort(key=lambda x: x['memory_mb'], reverse=True)
    return processes[:limit]


def get_python_processes() -> List[Dict]:
    """Obtém processos Python em execução"""
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'memory_percent']):
        try:
            pinfo = proc.info
            if 'python' in pinfo['name'].lower() or \
               (pinfo['cmdline'] and any('python' in str(cmd).lower() for cmd in pinfo['cmdline'])):
                python_procs.append({
                    'pid': pinfo['pid'],
                    'name': pinfo['name'],
                    'cmdline': ' '.join(pinfo['cmdline'][:3]) if pinfo['cmdline'] else '',
                    'memory_mb': pinfo['memory_info'].rss / (1024**2),
                    'memory_percent': pinfo['memory_percent']
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return python_procs


def print_memory_info():
    """Imprime informações de memória formatadas"""
    print("=" * 70)
    print(f"📊 Monitor de Memória - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    # Informações gerais de memória
    mem_info = get_memory_info()
    
    print("💾 Memória RAM:")
    print(f"  Total:     {format_bytes(mem_info['ram_total'])}")
    print(f"  Usada:     {format_bytes(mem_info['ram_used'])} ({mem_info['ram_percent']:.1f}%)")
    print(f"  Disponível: {format_bytes(mem_info['ram_available'])} ({100 - mem_info['ram_percent']:.1f}%)")
    
    # Status de memória
    if mem_info['ram_percent'] > 90:
        print("  ⚠️  ALERTA: Memória RAM acima de 90%!")
    elif mem_info['ram_percent'] > 75:
        print("  ⚠️  AVISO: Memória RAM acima de 75%")
    else:
        print("  ✅ Memória RAM em nível saudável")
    
    print()
    print("💿 Swap:")
    print(f"  Total:     {format_bytes(mem_info['swap_total'])}")
    print(f"  Usada:     {format_bytes(mem_info['swap_used'])} ({mem_info['swap_percent']:.1f}%)")
    
    if mem_info['swap_used'] > 0:
        print("  ⚠️  AVISO: Swap está sendo usado (pode indicar falta de RAM)")
    else:
        print("  ✅ Swap não está sendo usado")
    
    print()
    print("=" * 70)
    print("🔝 Top 10 Processos por Memória")
    print("=" * 70)
    print(f"{'PID':<8} {'Nome':<25} {'Memória (MB)':<15} {'% RAM':<10}")
    print("-" * 70)
    
    top_procs = get_top_processes(10)
    for proc in top_procs:
        print(f"{proc['pid']:<8} {proc['name'][:24]:<25} {proc['memory_mb']:>12.2f} MB  {proc['memory_percent']:>6.2f}%")
    
    print()
    print("=" * 70)
    print("🐍 Processos Python")
    print("=" * 70)
    
    python_procs = get_python_processes()
    if python_procs:
        print(f"{'PID':<8} {'Nome':<20} {'Memória (MB)':<15} {'% RAM':<10} {'Comando':<30}")
        print("-" * 70)
        total_python_memory = 0
        for proc in python_procs:
            total_python_memory += proc['memory_mb']
            cmd = proc['cmdline'][:28] + '...' if len(proc['cmdline']) > 28 else proc['cmdline']
            print(f"{proc['pid']:<8} {proc['name'][:19]:<20} {proc['memory_mb']:>12.2f} MB  {proc['memory_percent']:>6.2f}%  {cmd:<30}")
        print("-" * 70)
        print(f"{'Total Python:':<30} {total_python_memory:>12.2f} MB")
    else:
        print("  Nenhum processo Python em execução")
    
    print()
    print("=" * 70)
    print("💡 Recomendações:")
    print("=" * 70)
    
    recommendations = []
    
    if mem_info['ram_percent'] > 90:
        recommendations.append("❌ FECHE aplicações desnecessárias imediatamente!")
        recommendations.append("❌ Reduza SERVER_WORKERS para 1 no .env")
    
    if mem_info['ram_percent'] > 75:
        recommendations.append("⚠️  Considere fechar algumas abas do navegador")
        recommendations.append("⚠️  Verifique se há processos Python órfãos")
    
    if mem_info['swap_used'] > 0:
        recommendations.append("⚠️  Swap está sendo usado - sistema pode estar lento")
        recommendations.append("⚠️  Feche aplicações pesadas")
    
    if python_procs:
        total_python = sum(p['memory_mb'] for p in python_procs)
        if total_python > 2000:  # Mais de 2GB em processos Python
            recommendations.append("⚠️  Processos Python usando muita memória (>2GB)")
            recommendations.append("⚠️  Considere reiniciar o servidor")
    
    if not recommendations:
        recommendations.append("✅ Sistema está funcionando bem!")
        recommendations.append("✅ Continue monitorando para manter performance")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print()


def main():
    """Função principal"""
    try:
        print_memory_info()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoramento interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro ao monitorar memória: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
