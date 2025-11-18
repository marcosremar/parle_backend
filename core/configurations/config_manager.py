#!/usr/bin/env python3
"""
Configuration Manager - Sistema centralizado de gerenciamento de configurações
Controla todas as configurações do pipeline de forma unificada
Agora com suporte a .env e settings.yaml
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv

from .conversation_prompts import ConversationType, conversation_prompts

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuração geral do pipeline"""
    # Configurações de conversação
    default_conversation_type: str = "informal"
    default_language: str = "English"
    enable_context_memory: bool = True
    max_context_messages: int = 10

    # Configurações de token e resposta
    enforce_token_limits: bool = True
    validate_response_completeness: bool = True
    auto_adjust_tokens: bool = True

    # Configurações de áudio
    audio_enabled: bool = True
    default_voice_quality: str = "high"
    enable_audio_streaming: bool = True

    # Configurações de avaliação
    enable_quality_evaluation: bool = True
    evaluation_model: str = "llama-3.1-8b-instant"
    min_quality_threshold: float = 0.7


class ConfigurationManager:
    """
    Gerenciador centralizado de configurações
    Fornece acesso unificado a todas as configurações do sistema
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Inicializar gerenciador de configurações

        Args:
            config_dir: Diretório de configurações (padrão: este diretório)
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.pipeline_config = PipelineConfig()
        self.custom_settings = {}

        # Armazenar configurações do settings.yaml
        self.settings = {}

        # Carregar .env primeiro
        self._load_env()

        # Carregar settings.yaml com substituição de variáveis
        self._load_settings()

        # Carregar configurações pipeline (pipeline_config.json)
        self._load_configurations()

        logger.info(f"🔧 ConfigurationManager inicializado: {self.config_dir}")

    def _load_env(self):
        """Carrega variáveis de ambiente do arquivo .env"""
        # Procura .env na raiz do projeto
        # __file__ = /workspace/ultravox-pipeline/src/core/configurations/config_manager.py
        # parent.parent.parent.parent = /workspace/ultravox-pipeline
        root_dir = Path(__file__).parent.parent.parent.parent
        env_file = root_dir / '.env'

        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"✅ Loaded .env from {env_file}")
        else:
            logger.warning(f"⚠️ .env file not found at {env_file}")

    def _load_settings(self):
        """Carrega configurações do settings.yaml com substituição de variáveis"""
        # Procura settings.yaml na raiz do projeto
        # __file__ = /workspace/ultravox-pipeline/src/core/configurations/config_manager.py
        # parent.parent.parent.parent = /workspace/ultravox-pipeline
        root_dir = Path(__file__).parent.parent.parent.parent
        settings_file = root_dir / 'settings.yaml'

        if not settings_file.exists():
            logger.warning(f"⚠️ settings.yaml not found at {settings_file}")
            return

        try:
            with open(settings_file, 'r') as f:
                # Carrega YAML como estrutura Python
                raw_settings = yaml.safe_load(f)

            # Substitui variáveis de ambiente de forma recursiva
            self.settings = self._substitute_env_vars(raw_settings)
            logger.info(f"✅ Loaded settings from {settings_file}")

        except Exception as e:
            logger.error(f"❌ Error loading settings: {e}")
            self.settings = {}

    def _substitute_env_vars(self, data: Any) -> Any:
        """
        Substitui variáveis de ambiente de forma recursiva e segura
        Suporta formato ${VAR} e ${VAR:-default}
        """
        if isinstance(data, dict):
            # Processa dicionário recursivamente
            return {
                key: self._substitute_env_vars(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            # Processa lista recursivamente
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Substitui variáveis em strings
            return self._process_string(data)
        else:
            # Retorna outros tipos como estão
            return data

    def _process_string(self, value: str) -> Any:
        """
        Processa string para substituir variáveis de ambiente
        Sem usar regex - usa parsing simples e seguro
        """
        # Se não tem marcador de variável, retorna como está
        if '${' not in value:
            return value

        # Processa apenas strings que são puramente variáveis
        if value.startswith('${') and value.endswith('}'):
            # Remove ${ e }
            var_expr = value[2:-1]

            # Verifica se tem valor padrão
            if ':-' in var_expr:
                var_name, default_value = var_expr.split(':-', 1)
                result = os.environ.get(var_name, default_value)
            else:
                # Sem valor padrão
                result = os.environ.get(var_expr, value)

            # Converte tipos se necessário
            return self._convert_type(result)

        # Para strings mistas, não substitui (evita problemas)
        return value

    def _convert_type(self, value: str) -> Any:
        """
        Converte string para tipo apropriado
        """
        # Tenta converter para número
        if value.isdigit():
            return int(value)

        # Tenta converter para float
        try:
            return float(value)
        except ValueError:
            pass

        # Tenta converter para booleano
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False

        # Retorna como string
        return value

    def _load_configurations(self):
        """Carregar configurações de arquivos"""
        config_file = self.config_dir / "pipeline_config.json"

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                # Atualizar configurações do pipeline
                for key, value in config_data.get('pipeline', {}).items():
                    if hasattr(self.pipeline_config, key):
                        setattr(self.pipeline_config, key, value)

                # Carregar configurações customizadas
                self.custom_settings = config_data.get('custom', {})

                logger.info("✅ Configurações carregadas do arquivo")

            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar configurações: {e}")
        else:
            logger.info("📝 Usando configurações padrão")

    def save_configurations(self):
        """Salvar configurações em arquivo"""
        config_file = self.config_dir / "pipeline_config.json"

        try:
            config_data = {
                'pipeline': asdict(self.pipeline_config),
                'custom': self.custom_settings,
                'version': '1.0',
                'last_updated': str(Path(__file__).stat().st_mtime)
            }

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Configurações salvas: {config_file}")

        except Exception as e:
            logger.error(f"❌ Erro ao salvar configurações: {e}")

    def get_conversation_prompt(self,
                              conversation_type: str = None,
                              language: str = None,
                              custom_prompt: str = "") -> str:
        """
        Obter prompt de conversa otimizado

        Args:
            conversation_type: Tipo de conversa (padrão: configuração atual)
            language: Idioma (padrão: configuração atual)
            custom_prompt: Prompt personalizado

        Returns:
            Prompt completo formatado
        """
        # Usar configurações padrão se não especificado
        conv_type = conversation_type or self.pipeline_config.default_conversation_type
        lang = language or self.pipeline_config.default_language

        # Converter string para enum se necessário
        try:
            if isinstance(conv_type, str):
                conv_type_enum = ConversationType(conv_type.lower())
            else:
                conv_type_enum = conv_type
        except ValueError:
            logger.warning(f"⚠️ Tipo de conversa inválido: {conv_type}, usando informal")
            conv_type_enum = ConversationType.INFORMAL

        return conversation_prompts.get_conversation_prompt(
            conversation_type=conv_type_enum,
            language=lang,
            custom_prompt=custom_prompt
        )

    def get_token_limit(self, conversation_type: str = None) -> int:
        """
        Obter limite de tokens para tipo de conversa

        Args:
            conversation_type: Tipo de conversa

        Returns:
            Limite de tokens
        """
        conv_type = conversation_type or self.pipeline_config.default_conversation_type

        try:
            if isinstance(conv_type, str):
                conv_type_enum = ConversationType(conv_type.lower())
            else:
                conv_type_enum = conv_type
        except ValueError:
            conv_type_enum = ConversationType.INFORMAL

        base_limit = conversation_prompts.get_token_limit(conv_type_enum)

        # Ajustar baseado em configurações
        if self.pipeline_config.auto_adjust_tokens:
            # Aumentar limite se necessário baseado em contexto
            if self.pipeline_config.enable_context_memory:
                base_limit = int(base_limit * 1.2)

        return base_limit

    def get_temperature(self, conversation_type: str = None) -> float:
        """
        Obter temperatura para tipo de conversa

        Args:
            conversation_type: Tipo de conversa

        Returns:
            Temperatura recomendada
        """
        conv_type = conversation_type or self.pipeline_config.default_conversation_type

        try:
            if isinstance(conv_type, str):
                conv_type_enum = ConversationType(conv_type.lower())
            else:
                conv_type_enum = conv_type
        except ValueError:
            conv_type_enum = ConversationType.INFORMAL

        return conversation_prompts.get_temperature(conv_type_enum)

    def validate_response(self, response: str, conversation_type: str = None, voice_id: str = None) -> Dict[str, Any]:
        """
        Validar qualidade da resposta com métricas avançadas

        Args:
            response: Resposta para validar
            conversation_type: Tipo de conversa
            voice_id: ID da voz para detecção de idioma

        Returns:
            Dicionário com resultado da validação avançada
        """
        if not self.pipeline_config.validate_response_completeness:
            return {'valid': True, 'warnings': [], 'basic_validation': True}

        conv_type = conversation_type or self.pipeline_config.default_conversation_type
        max_tokens = self.get_token_limit(conv_type)

        # Converter string para enum se necessário
        try:
            if isinstance(conv_type, str):
                conv_type_enum = ConversationType(conv_type.lower())
            else:
                conv_type_enum = conv_type
        except ValueError:
            conv_type_enum = ConversationType.INFORMAL

        # Usar validação avançada com voice_id para detecção de idioma
        validation_result = conversation_prompts.advanced_response_validation(
            response=response,
            max_tokens=max_tokens,
            conversation_type=conv_type_enum,
            voice_id=voice_id
        )

        # Adicionar informações de configuração
        validation_result.update({
            'conversation_type': conv_type,
            'config_source': 'centralized',
            'validation_level': 'advanced'
        })

        return validation_result

    def get_language_from_voice(self, voice_id: str) -> str:
        """
        Detectar idioma da voz

        Args:
            voice_id: ID da voz

        Returns:
            Idioma detectado
        """
        return conversation_prompts.get_language_from_voice_id(voice_id)

    def update_setting(self, category: str, key: str, value: Any):
        """
        Atualizar configuração específica

        Args:
            category: Categoria da configuração ('pipeline' ou 'custom')
            key: Chave da configuração
            value: Novo valor
        """
        if category == 'pipeline':
            if hasattr(self.pipeline_config, key):
                setattr(self.pipeline_config, key, value)
                logger.info(f"⚙️ Configuração atualizada: {key} = {value}")
            else:
                logger.warning(f"⚠️ Configuração pipeline inválida: {key}")
        elif category == 'custom':
            self.custom_settings[key] = value
            logger.info(f"⚙️ Configuração customizada: {key} = {value}")
        else:
            logger.warning(f"⚠️ Categoria inválida: {category}")

    def get_setting(self, category: str, key: str, default: Any = None) -> Any:
        """
        Obter valor de configuração

        Args:
            category: Categoria da configuração
            key: Chave da configuração
            default: Valor padrão se não encontrado

        Returns:
            Valor da configuração
        """
        if category == 'pipeline':
            return getattr(self.pipeline_config, key, default)
        elif category == 'custom':
            return self.custom_settings.get(key, default)
        else:
            return default

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Obter todas as configurações

        Returns:
            Dicionário com todas as configurações
        """
        return {
            'pipeline': asdict(self.pipeline_config),
            'custom': self.custom_settings.copy(),
            'settings': self.settings.copy(),  # Incluir settings.yaml
            'conversation_types': [t.value for t in ConversationType],
            'available_languages': list(conversation_prompts.LANGUAGE_INSTRUCTIONS.keys())
        }

    def get_from_settings(self, path: str, default: Any = None) -> Any:
        """
        Obtém valor de configuração do settings.yaml usando caminho pontilhado

        Args:
            path: Caminho para configuração (ex: 'providers.llm.model')
            default: Valor padrão se não encontrado

        Returns:
            Valor da configuração ou padrão
        """
        keys = path.split('.')
        value = self.settings

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_provider_config(self, provider_type: str) -> Dict[str, Any]:
        """
        Obtém configuração completa de um provider do settings.yaml

        Args:
            provider_type: Tipo do provider (llm, stt, tts)

        Returns:
            Configuração do provider
        """
        return self.get_from_settings(f'providers.{provider_type}', {})

    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Obtém variável de ambiente

        Args:
            key: Nome da variável
            default: Valor padrão

        Returns:
            Valor da variável ou padrão
        """
        return os.environ.get(key, default)

    def reset_to_defaults(self):
        """Resetar todas as configurações para padrões"""
        self.pipeline_config = PipelineConfig()
        self.custom_settings = {}
        logger.info("🔄 Configurações resetadas para padrões")

    def export_config(self, filepath: str):
        """
        Exportar configurações para arquivo

        Args:
            filepath: Caminho do arquivo para exportar
        """
        try:
            config_data = self.get_all_settings()
            config_data['export_timestamp'] = str(Path(__file__).stat().st_mtime)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"📤 Configurações exportadas: {filepath}")

        except Exception as e:
            logger.error(f"❌ Erro ao exportar configurações: {e}")

    def import_config(self, filepath: str):
        """
        Importar configurações de arquivo

        Args:
            filepath: Caminho do arquivo para importar
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Atualizar configurações pipeline
            for key, value in config_data.get('pipeline', {}).items():
                if hasattr(self.pipeline_config, key):
                    setattr(self.pipeline_config, key, value)

            # Atualizar configurações customizadas
            self.custom_settings.update(config_data.get('custom', {}))

            logger.info(f"📥 Configurações importadas: {filepath}")

        except Exception as e:
            logger.error(f"❌ Erro ao importar configurações: {e}")


# Instância global do gerenciador de configurações
config_manager = ConfigurationManager()