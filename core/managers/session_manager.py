#!/usr/bin/env python3
"""
Sistema de gerenciamento de sessões para manter contexto de conversas
Cada usuário pode ter múltiplas sessões, cada uma com seu próprio histórico
"""

import uuid
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
import pickle
from pathlib import Path

from loguru import logger


@dataclass
class Message:
    """Representa uma mensagem no histórico da conversa"""
    role: str  # 'user' ou 'assistant'
    content: str
    timestamp: float
    audio_duration_ms: Optional[float] = None
    voice_used: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Session:
    """Representa uma sessão de conversa"""
    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    messages: List[Message]
    metadata: Dict[str, Any]

    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Adiciona mensagem ao histórico"""
        self.messages.append(Message(
            role=role,
            content=content,
            timestamp=time.time(),
            **kwargs
        ))
        self.last_activity = time.time()

    def get_context(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Retorna contexto formatado para o LLM"""
        # Pega as últimas N mensagens
        recent_messages = self.messages[-max_messages:] if max_messages else self.messages

        # Formata para o LLM
        context = []
        for msg in recent_messages:
            context.append({
                "role": msg.role,
                "content": msg.content
            })

        return context

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Session':
        messages = [Message(**msg) for msg in data.get("messages", [])]
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            created_at=data["created_at"],
            last_activity=data["last_activity"],
            messages=messages,
            metadata=data.get("metadata", {})
        )


class SessionManager:
    """Gerenciador centralizado de sessões"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> 'SessionManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, '_initialized'):
            self._initialized = True

            # Armazenamento em memória
            self.users: Dict[str, Dict[str, Any]] = {}  # user_id -> user_data
            self.sessions: Dict[str, Session] = {}  # session_id -> Session
            self.user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]

            # Configurações
            self.session_timeout_hours = 24  # Sessões expiram após 24h de inatividade
            self.max_sessions_per_user = 100  # Máximo de sessões por usuário
            self.persistence_dir = Path("./data/sessions")
            self.persistence_dir.mkdir(parents=True, exist_ok=True)

            # Thread lock para operações
            self._data_lock = threading.Lock()

            # Carregar sessões persistidas
            self._load_sessions()

            logger.info(f"📝 SessionManager inicializado com {len(self.sessions)} sessões")

    def create_user(self, user_id: str = None, metadata: Dict[str, Any] = None) -> str:
        """Cria ou retorna um usuário"""
        if user_id is None:
            user_id = f"user_{uuid.uuid4().hex[:8]}"

        with self._data_lock:
            if user_id not in self.users:
                self.users[user_id] = {
                    "user_id": user_id,
                    "created_at": time.time(),
                    "metadata": metadata or {}
                }
                self.user_sessions[user_id] = []
                logger.info(f"👤 Novo usuário criado: {user_id}")

        return user_id

    def create_session(self, user_id: str = None, metadata: Dict[str, Any] = None) -> Session:
        """Cria uma nova sessão para um usuário"""
        # Criar usuário se não existir
        if user_id is None:
            user_id = "test_user"  # Usuário padrão para testes

        user_id = self.create_user(user_id)

        # Gerar ID único para sessão
        session_id = f"session_{uuid.uuid4().hex[:12]}"

        # Criar sessão
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=time.time(),
            last_activity=time.time(),
            messages=[],
            metadata=metadata or {}
        )

        with self._data_lock:
            # Armazenar sessão
            self.sessions[session_id] = session

            # Associar ao usuário
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = []
            self.user_sessions[user_id].append(session_id)

            # Limitar número de sessões por usuário
            if len(self.user_sessions[user_id]) > self.max_sessions_per_user:
                # Remover sessão mais antiga
                old_session_id = self.user_sessions[user_id].pop(0)
                del self.sessions[old_session_id]
                logger.info(f"🗑️ Sessão antiga removida: {old_session_id}")

        logger.info(f"📝 Nova sessão criada: {session_id} para usuário {user_id}")

        # Persistir
        self._save_session(session)

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Recupera uma sessão pelo ID"""
        with self._data_lock:
            session = self.sessions.get(session_id)

            if session:
                # Verificar se não expirou
                if self._is_session_expired(session):
                    logger.info(f"⏰ Sessão expirada: {session_id}")
                    self._remove_session(session_id)
                    return None

                # Atualizar última atividade
                session.last_activity = time.time()

            return session

    def get_or_create_session(self, session_id: str = None, user_id: str = None) -> Session:
        """Recupera sessão existente ou cria uma nova"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session

        # Criar nova sessão
        return self.create_session(user_id)

    def add_interaction(self,
                       session_id: str,
                       user_message: str,
                       assistant_response: str,
                       audio_duration_ms: Optional[float] = None,
                       voice_used: Optional[str] = None) -> bool:
        """Adiciona uma interação completa ao histórico da sessão"""
        session = self.get_session(session_id)

        if not session:
            logger.error(f"❌ Sessão não encontrada: {session_id}")
            return False

        # Adicionar mensagem do usuário
        session.add_message(
            role="user",
            content=user_message,
            audio_duration_ms=audio_duration_ms
        )

        # Adicionar resposta do assistente
        session.add_message(
            role="assistant",
            content=assistant_response,
            voice_used=voice_used
        )

        # Persistir
        self._save_session(session)

        logger.info(f"💬 Interação adicionada à sessão {session_id}")
        return True

    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Retorna todas as sessões de um usuário"""
        with self._data_lock:
            session_ids = self.user_sessions.get(user_id, [])
            sessions = []

            for sid in session_ids:
                session = self.sessions.get(sid)
                if session and not self._is_session_expired(session):
                    sessions.append(session)

            return sessions

    def get_session_context(self, session_id: str, max_messages: int = 10) -> List[Dict[str, str]]:
        """Retorna contexto formatado de uma sessão para o LLM"""
        session = self.get_session(session_id)

        if not session:
            return []

        return session.get_context(max_messages)

    def cleanup_expired_sessions(self) -> None:
        """Remove sessões expiradas"""
        with self._data_lock:
            expired = []

            for sid, session in self.sessions.items():
                if self._is_session_expired(session):
                    expired.append(sid)

            for sid in expired:
                self._remove_session(sid)

            if expired:
                logger.info(f"🧹 {len(expired)} sessões expiradas removidas")

    def _is_session_expired(self, session: Session) -> bool:
        """Verifica se uma sessão expirou"""
        timeout_seconds = self.session_timeout_hours * 3600
        return (time.time() - session.last_activity) > timeout_seconds

    def _remove_session(self, session_id: str) -> None:
        """Remove uma sessão (interno)"""
        if session_id in self.sessions:
            session = self.sessions[session_id]

            # Remover da lista do usuário
            user_id = session.user_id
            if user_id in self.user_sessions:
                if session_id in self.user_sessions[user_id]:
                    self.user_sessions[user_id].remove(session_id)

            # Remover sessão
            del self.sessions[session_id]

            # Remover arquivo persistido
            session_file = self.persistence_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()

    def _save_session(self, session: Session) -> None:
        """Persiste sessão em disco"""
        try:
            session_file = self.persistence_dir / f"{session.session_id}.json"

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"❌ Erro ao salvar sessão: {e}")

    def _load_sessions(self) -> None:
        """Carrega sessões do disco"""
        try:
            loaded_count = 0

            for session_file in self.persistence_dir.glob("*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    session = Session.from_dict(data)

                    # Só carregar se não expirou
                    if not self._is_session_expired(session):
                        self.sessions[session.session_id] = session

                        # Reconstruir mapeamento de usuário
                        if session.user_id not in self.user_sessions:
                            self.user_sessions[session.user_id] = []
                        self.user_sessions[session.user_id].append(session.session_id)

                        # Reconstruir dados do usuário
                        if session.user_id not in self.users:
                            self.users[session.user_id] = {
                                "user_id": session.user_id,
                                "created_at": session.created_at,
                                "metadata": {}
                            }

                        loaded_count += 1
                    else:
                        # Remover arquivo de sessão expirada
                        session_file.unlink()

                except Exception as e:
                    logger.error(f"❌ Erro ao carregar sessão {session_file}: {e}")

            if loaded_count > 0:
                logger.info(f"📂 {loaded_count} sessões carregadas do disco")

        except Exception as e:
            logger.error(f"❌ Erro ao carregar sessões: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do gerenciador"""
        with self._data_lock:
            total_messages = sum(len(s.messages) for s in self.sessions.values())

            return {
                "total_users": len(self.users),
                "total_sessions": len(self.sessions),
                "total_messages": total_messages,
                "active_sessions": sum(
                    1 for s in self.sessions.values()
                    if (time.time() - s.last_activity) < 3600  # Ativa na última hora
                ),
                "sessions_per_user": {
                    uid: len(sids)
                    for uid, sids in self.user_sessions.items()
                }
            }


# Singleton global
session_manager = SessionManager()


# Funções helper para fácil acesso
def create_session(user_id: str = None, metadata: Dict[str, Any] = None) -> Session:
    """Cria uma nova sessão"""
    return session_manager.create_session(user_id, metadata)


def get_session(session_id: str) -> Optional[Session]:
    """Recupera uma sessão"""
    return session_manager.get_session(session_id)


def get_or_create_session(session_id: str = None, user_id: str = None) -> Session:
    """Recupera ou cria uma sessão"""
    return session_manager.get_or_create_session(session_id, user_id)


def add_interaction(session_id: str,
                   user_message: str,
                   assistant_response: str,
                   **kwargs) -> bool:
    """Adiciona interação a uma sessão"""
    return session_manager.add_interaction(
        session_id, user_message, assistant_response, **kwargs
    )


def get_context(session_id: str, max_messages: int = 10) -> List[Dict[str, str]]:
    """Recupera contexto de uma sessão"""
    return session_manager.get_session_context(session_id, max_messages)


if __name__ == "__main__":
    # Teste do sistema
    print("🧪 Testando SessionManager...")

    # Criar sessão para usuário teste
    session1 = create_session("test_user", {"device": "web"})
    print(f"✅ Sessão criada: {session1.session_id}")

    # Adicionar interações
    add_interaction(
        session1.session_id,
        "Olá, como você está?",
        "Olá! Estou funcionando perfeitamente. Como posso ajudá-lo?",
        voice_used="pf_dora"
    )

    add_interaction(
        session1.session_id,
        "Qual é a capital do Brasil?",
        "A capital do Brasil é Brasília.",
        voice_used="pf_dora"
    )

    # Recuperar contexto
    context = get_context(session1.session_id)
    print(f"\n📝 Contexto da sessão:")
    for msg in context:
        print(f"  {msg['role']}: {msg['content'][:50]}...")

    # Criar segunda sessão para mesmo usuário
    session2 = create_session("test_user", {"device": "mobile"})
    print(f"\n✅ Segunda sessão criada: {session2.session_id}")

    # Estatísticas
    stats = session_manager.get_statistics()
    print(f"\n📊 Estatísticas:")
    print(f"  Usuários: {stats['total_users']}")
    print(f"  Sessões: {stats['total_sessions']}")
    print(f"  Mensagens: {stats['total_messages']}")
    print(f"  Sessões ativas: {stats['active_sessions']}")

    print("\n✅ Teste concluído!")