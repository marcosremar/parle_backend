#!/usr/bin/env python3
"""
Context Manager - Sliding Window + RAG
Combina contexto recente (sliding window) com busca de memórias antigas (RAG)
"""

import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import logging

# DEPRECATED: modules.memory removed (replaced by conversation_store service)
# from modules.memory import VectorMemoryStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configurar handler se não existir
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - [32m%(levelname)s[0m - %(message)s',
                                          datefmt='%H:%M:%S'))
    logger.addHandler(handler)


@dataclass
class Message:
    """Mensagem na conversa"""
    role: str  # 'user' ou 'assistant'
    content: str
    timestamp: float
    tokens: int = 0  # Estimativa de tokens
    importance: float = 1.0  # Score de importância (para futuro)


class ContextManager:
    """
    Gerenciador de contexto híbrido:
    - Sliding Window para mensagens recentes
    - RAG (Vector Search) para memórias relevantes antigas
    """

    def __init__(self,
                 window_size: int = 30,
                 max_tokens: int = 3000,
                 rag_top_k: int = 3,
                 rag_threshold: float = 0.7):
        """
        Args:
            window_size: Número de mensagens recentes a manter
            max_tokens: Máximo de tokens no contexto
            rag_top_k: Número de memórias antigas a buscar
            rag_threshold: Threshold de similaridade para RAG
        """
        # Sliding Window
        self.sliding_window = deque(maxlen=window_size)
        self.window_size = window_size
        self.max_tokens = max_tokens

        # RAG Configuration
        self.rag_top_k = rag_top_k
        self.rag_threshold = rag_threshold
        self.vector_memory = None  # Inicializado depois

        # Session info
        self.session_id = None
        self.user_id = None

        # Statistics
        self.stats = {
            'total_messages': 0,
            'window_hits': 0,
            'rag_searches': 0,
            'avg_context_tokens': 0
        }

        logger.info(f"📝 ContextManager criado (window={window_size}, max_tokens={max_tokens})")

    async def initialize(self, session_id: str, user_id: str):
        """
        Inicializar com sessão e memória vetorial

        Args:
            session_id: ID da sessão atual
            user_id: ID do usuário
        """
        self.session_id = session_id
        self.user_id = user_id

        # Inicializar memória vetorial para RAG
        self.vector_memory = VectorMemoryStore()
        await self.vector_memory.initialize()

        logger.info(f"✅ ContextManager inicializado para sessão {session_id[:12]}...")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimar número de tokens em um texto
        Aproximação: 1 token ~= 4 caracteres
        """
        return len(text) // 4

    def add_message(self, role: str, content: str, importance: float = 1.0):
        """
        Adicionar mensagem ao sliding window

        Args:
            role: 'user' ou 'assistant'
            content: Conteúdo da mensagem
            importance: Score de importância (futuro uso)
        """
        message = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            tokens=self.estimate_tokens(content),
            importance=importance
        )

        self.sliding_window.append(message)
        self.stats['total_messages'] += 1

        logger.info(f"💬 Mensagem adicionada ao window ({role}): {content[:50]}...")

    async def add_to_memory(self, user_msg: str, assistant_msg: str):
        """
        Adicionar exchange completo à memória vetorial (para RAG futuro)

        Args:
            user_msg: Mensagem do usuário
            assistant_msg: Resposta do assistente
        """
        if not self.vector_memory:
            return

        # Criar conteúdo combinado para melhor busca
        combined_content = f"User: {user_msg}\nAssistant: {assistant_msg}"

        # Adicionar à memória vetorial
        await self.vector_memory.add_memory(
            session_id=self.session_id,
            content=combined_content,
            role="exchange"
        )

        logger.info(f"🧠 Exchange adicionado à memória vetorial")

    async def search_relevant_memories(self, query: str) -> List[Dict[str, Any]]:
        """
        Buscar memórias relevantes antigas usando RAG

        Args:
            query: Query atual do usuário

        Returns:
            Lista de memórias relevantes
        """
        if not self.vector_memory:
            return []

        self.stats['rag_searches'] += 1

        # Buscar memórias relevantes
        memories = await self.vector_memory.search_memories(
            query=query,
            user_id=self.user_id,
            top_k=self.rag_top_k,
            threshold=self.rag_threshold
        )

        # Filtrar memórias que já estão no sliding window (evitar duplicação)
        window_contents = {msg.content for msg in self.sliding_window}
        filtered_memories = [
            mem for mem in memories
            if mem['content'] not in window_contents
        ]

        if filtered_memories:
            logger.info(f"🔍 RAG encontrou {len(filtered_memories)} memórias relevantes")

        return filtered_memories

    def get_sliding_window_context(self, max_tokens: Optional[int] = None) -> List[Message]:
        """
        Obter contexto do sliding window respeitando limite de tokens

        Args:
            max_tokens: Limite de tokens (usa self.max_tokens se None)

        Returns:
            Lista de mensagens do sliding window
        """
        max_tokens = max_tokens or self.max_tokens

        # Começar do mais recente e ir adicionando até o limite
        context = []
        total_tokens = 0

        for message in reversed(self.sliding_window):
            if total_tokens + message.tokens > max_tokens:
                break
            context.insert(0, message)
            total_tokens += message.tokens

        self.stats['window_hits'] += 1

        return context

    async def build_context(self,
                           user_message: str,
                           include_rag: bool = True) -> Dict[str, Any]:
        """
        Construir contexto completo: Sliding Window + RAG

        Args:
            user_message: Mensagem atual do usuário
            include_rag: Se deve incluir busca RAG

        Returns:
            Dicionário com contexto estruturado
        """
        context_parts = {}
        total_tokens = 0

        # 1. Estimar tokens da mensagem atual
        user_tokens = self.estimate_tokens(user_message)
        total_tokens += user_tokens

        # 2. Buscar memórias relevantes via RAG (se habilitado)
        relevant_memories = []
        if include_rag and self.vector_memory:
            relevant_memories = await self.search_relevant_memories(user_message)

            # Limitar memórias RAG a ~30% do contexto
            rag_token_limit = int(self.max_tokens * 0.3)
            rag_tokens = 0
            filtered_memories = []

            for mem in relevant_memories:
                mem_tokens = self.estimate_tokens(mem['content'])
                if rag_tokens + mem_tokens > rag_token_limit:
                    break
                filtered_memories.append(mem)
                rag_tokens += mem_tokens

            relevant_memories = filtered_memories
            total_tokens += rag_tokens

            if relevant_memories:
                context_parts['relevant_memories'] = relevant_memories

        # 3. Obter sliding window (usar tokens restantes)
        remaining_tokens = self.max_tokens - total_tokens
        sliding_context = self.get_sliding_window_context(remaining_tokens)
        context_parts['recent_messages'] = sliding_context

        # 4. Estatísticas
        window_tokens = sum(msg.tokens for msg in sliding_context)
        total_tokens += window_tokens

        self.stats['avg_context_tokens'] = (
            self.stats['avg_context_tokens'] * 0.9 + total_tokens * 0.1
        )

        # 5. Construir contexto final
        context = {
            'user_message': user_message,
            'recent_messages': sliding_context,
            'relevant_memories': relevant_memories,
            'metadata': {
                'total_tokens': total_tokens,
                'window_messages': len(sliding_context),
                'rag_memories': len(relevant_memories),
                'session_id': self.session_id,
                'user_id': self.user_id
            }
        }

        logger.info(
            f"📋 Contexto construído: "
            f"{len(sliding_context)} msgs recentes, "
            f"{len(relevant_memories)} memórias RAG, "
            f"{total_tokens} tokens total"
        )

        return context

    def format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """
        Formatar contexto para envio ao LLM

        Args:
            context: Contexto estruturado

        Returns:
            String formatada para o prompt
        """
        parts = []

        # 1. Memórias relevantes (se houver)
        if context.get('relevant_memories'):
            parts.append("=== Relevant Previous Conversations ===")
            for mem in context['relevant_memories']:
                timestamp = time.strftime('%Y-%m-%d %H:%M',
                                         time.localtime(mem.get('timestamp', 0)))
                parts.append(f"[{timestamp}] {mem['content']}")
            parts.append("")

        # 2. Conversa recente
        if context.get('recent_messages'):
            parts.append("=== Recent Conversation ===")
            for msg in context['recent_messages']:
                role = msg.role.capitalize()
                parts.append(f"{role}: {msg.content}")
            parts.append("")

        # 3. Mensagem atual
        parts.append(f"User: {context['user_message']}")

        return "\n".join(parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Obter estatísticas do gerenciador"""
        return {
            **self.stats,
            'window_size': len(self.sliding_window),
            'max_window_size': self.window_size,
            'session_id': self.session_id,
            'user_id': self.user_id
        }

    def clear_window(self):
        """Limpar sliding window (manter memórias vetoriais)"""
        self.sliding_window.clear()
        logger.info("🧹 Sliding window limpo")

    async def save_session(self):
        """Salvar estado da sessão"""
        # Vector memory já salva automaticamente
        # Aqui podemos adicionar persistência adicional se necessário
        pass


# Singleton para fácil acesso
_context_managers: Dict[str, ContextManager] = {}


def get_context_manager(session_id: str) -> Optional[ContextManager]:
    """Obter context manager para uma sessão"""
    return _context_managers.get(session_id)


async def create_context_manager(session_id: str, user_id: str, **kwargs) -> ContextManager:
    """Criar novo context manager para sessão"""
    cm = ContextManager(**kwargs)
    await cm.initialize(session_id, user_id)
    _context_managers[session_id] = cm
    return cm


def cleanup_old_managers(max_age_hours: int = 24):
    """Limpar context managers antigos"""
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    to_remove = []
    for session_id, cm in _context_managers.items():
        if cm.sliding_window:
            last_msg_time = cm.sliding_window[-1].timestamp
            if current_time - last_msg_time > max_age_seconds:
                to_remove.append(session_id)

    for session_id in to_remove:
        del _context_managers[session_id]
        logger.info(f"🧹 Context manager removido: {session_id[:12]}...")

    return len(to_remove)