"""
Database models for Student Model Service
SQLAlchemy models for tracking student knowledge and progress
"""

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone
import os
from pathlib import Path

Base = declarative_base()


class User(Base):
    """Perfil do estudante"""
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True, index=True)
    cefr_level = Column(String, default="A1")  # A1, A2, B1, B2, C1, C2
    native_language = Column(String, default="en")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    skill_masteries = relationship("SkillMastery", back_populates="user", cascade="all, delete-orphan")
    interaction_history = relationship("InteractionHistory", back_populates="user", cascade="all, delete-orphan")


class Skill(Base):
    """Catálogo de habilidades linguísticas"""
    __tablename__ = "skills"
    
    skill_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)  # grammar, vocabulary, pronunciation
    difficulty = Column(String, default="beginner")  # beginner, intermediate, advanced
    description = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    skill_masteries = relationship("SkillMastery", back_populates="skill", cascade="all, delete-orphan")


class SkillMastery(Base):
    """Progresso do estudante em uma habilidade específica"""
    __tablename__ = "skill_mastery"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    skill_id = Column(String, ForeignKey("skills.skill_id"), nullable=False, index=True)
    mastery_probability = Column(Float, default=0.0)  # 0.0 to 1.0 (BKT output)
    attempts = Column(Integer, default=0)
    successes = Column(Integer, default=0)
    last_practiced = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user = relationship("User", back_populates="skill_masteries")
    skill = relationship("Skill", back_populates="skill_masteries")


class InteractionHistory(Base):
    """Log de todas as interações do estudante"""
    __tablename__ = "interaction_history"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    skill_id = Column(String, ForeignKey("skills.skill_id"), nullable=False, index=True)
    correct = Column(Boolean, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    context = Column(Text)  # JSON string with additional context
    user_text = Column(Text)  # What the user said
    ai_text = Column(Text)  # What the AI responded
    semantic_features = Column(Text)  # JSON string with structured semantic features for context-aware representations
    
    # Relationships
    user = relationship("User", back_populates="interaction_history")


def get_database_url():
    """Get database URL from environment or use default"""
    db_path = os.getenv("STUDENT_MODEL_DB_PATH", "data/student_model.db")
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def create_engine_and_session():
    """Create database engine and session factory"""
    database_url = get_database_url()
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal


def init_database():
    """Initialize database tables"""
    engine, _ = create_engine_and_session()
    Base.metadata.create_all(bind=engine)
    return engine

