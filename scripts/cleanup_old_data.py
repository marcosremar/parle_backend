#!/usr/bin/env python3
"""
Script de limpeza de dados antigos conforme política de retenção
Executa limpeza automática de dados expirados
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

# Configure logging
logger.add(
    project_root / "logs" / "cleanup.log",
    rotation="100 MB",
    retention="30 days",
    level="INFO"
)


# Retention periods (in days)
RETENTION_PERIODS = {
    "conversations": int(os.getenv("CONVERSATION_RETENTION_DAYS", "365")),
    "messages": int(os.getenv("MESSAGE_RETENTION_DAYS", "365")),
    "sessions": int(os.getenv("SESSION_RETENTION_DAYS", "7")),
    "audit_logs": int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "730")),
    "application_logs": int(os.getenv("LOG_RETENTION_DAYS", "30")),
    "audio_files": int(os.getenv("AUDIO_RETENTION_DAYS", "180")),
}


async def cleanup_expired_sessions():
    """Clean up expired sessions"""
    try:
        from src.modules import module_factory
        session_module = module_factory.create("session")
        
        if hasattr(session_module, "cleanup_expired_sessions"):
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=RETENTION_PERIODS["sessions"])
            deleted = await session_module.cleanup_expired_sessions(cutoff_date)
            logger.info(f"🧹 Limpeza de sessões: {deleted} sessões expiradas removidas")
            return deleted
    except Exception as e:
        logger.error(f"❌ Erro ao limpar sessões: {e}")
    return 0


async def cleanup_old_conversations():
    """Clean up old inactive conversations"""
    try:
        from src.modules import module_factory
        store_module = module_factory.create("conversation_store")
        
        if hasattr(store_module, "cleanup_old_conversations"):
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=RETENTION_PERIODS["conversations"])
            deleted = await store_module.cleanup_old_conversations(cutoff_date)
            logger.info(f"🧹 Limpeza de conversações: {deleted} conversações antigas removidas")
            return deleted
    except Exception as e:
        logger.error(f"❌ Erro ao limpar conversações: {e}")
    return 0


async def cleanup_old_messages():
    """Clean up old messages"""
    try:
        from src.modules import module_factory
        store_module = module_factory.create("conversation_store")
        
        if hasattr(store_module, "cleanup_old_messages"):
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=RETENTION_PERIODS["messages"])
            deleted = await store_module.cleanup_old_messages(cutoff_date)
            logger.info(f"🧹 Limpeza de mensagens: {deleted} mensagens antigas removidas")
            return deleted
    except Exception as e:
        logger.error(f"❌ Erro ao limpar mensagens: {e}")
    return 0


def cleanup_old_log_files():
    """Clean up old log files"""
    logs_dir = project_root / "logs"
    if not logs_dir.exists():
        return 0
    
    cutoff_date = datetime.now() - timedelta(days=RETENTION_PERIODS["application_logs"])
    deleted = 0
    
    for log_file in logs_dir.glob("*.log*"):
        try:
            # Check file modification time
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if mtime < cutoff_date:
                log_file.unlink()
                deleted += 1
                logger.debug(f"🗑️  Deletado: {log_file.name}")
        except Exception as e:
            logger.warning(f"⚠️  Erro ao deletar {log_file}: {e}")
    
    logger.info(f"🧹 Limpeza de logs: {deleted} arquivos de log antigos removidos")
    return deleted


async def cleanup_old_audio_files():
    """Clean up old audio files"""
    try:
        from src.modules import module_factory
        storage_module = module_factory.create("file_storage")
        
        if hasattr(storage_module, "cleanup_old_files"):
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=RETENTION_PERIODS["audio_files"])
            deleted = await storage_module.cleanup_old_files(
                cutoff_date=cutoff_date,
                file_type="audio"
            )
            logger.info(f"🧹 Limpeza de arquivos de áudio: {deleted} arquivos antigos removidos")
            return deleted
    except Exception as e:
        logger.error(f"❌ Erro ao limpar arquivos de áudio: {e}")
    return 0


async def main():
    """Main cleanup function"""
    logger.info("🧹 Iniciando limpeza de dados antigos...")
    logger.info(f"📅 Data de corte: {datetime.now(timezone.utc).isoformat()}")
    
    stats = {
        "sessions": 0,
        "conversations": 0,
        "messages": 0,
        "logs": 0,
        "audio_files": 0
    }
    
    try:
        # Cleanup sessions
        stats["sessions"] = await cleanup_expired_sessions()
        
        # Cleanup conversations
        stats["conversations"] = await cleanup_old_conversations()
        
        # Cleanup messages
        stats["messages"] = await cleanup_old_messages()
        
        # Cleanup log files
        stats["logs"] = cleanup_old_log_files()
        
        # Cleanup audio files
        stats["audio_files"] = await cleanup_old_audio_files()
        
        total = sum(stats.values())
        
        logger.info("✅ Limpeza concluída!")
        logger.info(f"📊 Estatísticas:")
        logger.info(f"   - Sessões: {stats['sessions']}")
        logger.info(f"   - Conversações: {stats['conversations']}")
        logger.info(f"   - Mensagens: {stats['messages']}")
        logger.info(f"   - Logs: {stats['logs']}")
        logger.info(f"   - Arquivos de áudio: {stats['audio_files']}")
        logger.info(f"   - Total: {total} registros removidos")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Erro durante limpeza: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
