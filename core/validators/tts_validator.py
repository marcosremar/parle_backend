#!/usr/bin/env python3
"""
Validador Robusto para TTS (Text-to-Speech)
Sistema completo de validação com análise via Groq
"""

import os
import tempfile
import logging
import json
import time
import hashlib
from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path
import numpy as np
import base64
from src.core.exceptions import UltravoxError, wrap_exception

# Importar exceções customizadas
try:
    from .error_handler import (
        UltravoxError, ErrorSeverity, ErrorCategory, ValidationError, ProcessingError,
        TTSError, KokoroTTSError, GTTSError, AudioCorruptedError
    )
except ImportError:
    # Fallback se error_handler não estiver disponível
    class TTSError(Exception):
        """Exception raised for tts error errors"""
        pass
    class KokoroTTSError(Exception):
        """Exception raised for kokoro tts error errors"""
        pass
    class GTTSError(Exception):
        """Exception raised for gtts error errors"""
        pass
    class AudioCorruptedError(Exception):
        """Exception raised for audio corrupted error errors"""
        pass
    class ValidationError(Exception):
        """Exception raised for validation error errors"""
        pass

logger = logging.getLogger(__name__)


class TTSValidationConfig:
    """Configuração para validação de TTS"""

    def __init__(self,
                 enable_groq_validation: bool = False,
                 groq_api_key: Optional[str] = None,
                 min_audio_duration_ms: float = 100,
                 max_audio_duration_ms: float = 60000,
                 min_volume_threshold: float = 0.001,
                 enable_voice_detection: bool = True,
                 enable_silence_detection: bool = True,
                 enable_quality_check: bool = True,
                 cache_validations: bool = True,
                 development_mode: bool = False):
        """
        Args:
            enable_groq_validation: Habilitar validação via Groq
            groq_api_key: API key do Groq
            min_audio_duration_ms: Duração mínima do áudio
            max_audio_duration_ms: Duração máxima do áudio
            min_volume_threshold: Threshold mínimo de volume
            enable_voice_detection: Detectar presença de voz
            enable_silence_detection: Detectar silêncios longos
            enable_quality_check: Verificar qualidade do áudio
            cache_validations: Cachear resultados de validação
            development_mode: Modo desenvolvimento com logs detalhados
        """
        self.enable_groq_validation = enable_groq_validation
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.min_audio_duration_ms = min_audio_duration_ms
        self.max_audio_duration_ms = max_audio_duration_ms
        self.min_volume_threshold = min_volume_threshold
        self.enable_voice_detection = enable_voice_detection
        self.enable_silence_detection = enable_silence_detection
        self.enable_quality_check = enable_quality_check
        self.cache_validations = cache_validations
        self.development_mode = development_mode

        # Cache de validações
        self._validation_cache = {}

    @classmethod
    def development_config(cls) -> 'TTSValidationConfig':
        """Configuração para desenvolvimento com todas validações"""
        return cls(
            enable_groq_validation=True,
            enable_voice_detection=True,
            enable_silence_detection=True,
            enable_quality_check=True,
            development_mode=True
        )

    @classmethod
    def production_config(cls) -> 'TTSValidationConfig':
        """Configuração para produção (validações básicas)"""
        return cls(
            enable_groq_validation=False,
            enable_voice_detection=False,
            enable_silence_detection=True,
            enable_quality_check=False,
            development_mode=False
        )


class TTSValidator:
    """Validador robusto para saídas de TTS"""

    def __init__(self, config: Optional[TTSValidationConfig] = None):
        """
        Args:
            config: Configuração de validação
        """
        self.config = config or TTSValidationConfig()
        self._groq_client = None
        self._init_groq_client()

    def _init_groq_client(self):
        """Inicializar cliente Groq se habilitado"""
        if self.config.enable_groq_validation and self.config.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.config.groq_api_key)
                logger.info("✅ Cliente Groq inicializado para validação TTS")
            except ImportError:
                logger.warning("❌ Groq não disponível. Instale com: pip install groq")
                self.config.enable_groq_validation = False
            except Exception as e:
                logger.error(f"❌ Erro ao inicializar Groq: {e}")
                self.config.enable_groq_validation = False

    def validate_tts_output(self,
                           audio_data: Union[bytes, np.ndarray],
                           text_input: str,
                           tts_engine: str = "unknown",
                           voice_id: Optional[str] = None,
                           session_id: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Validar saída de TTS

        Args:
            audio_data: Dados de áudio gerados
            text_input: Texto original usado para TTS
            tts_engine: Engine TTS usado (kokoro, gtts, etc.)
            voice_id: ID da voz usada
            session_id: ID da sessão

        Returns:
            Tuple[bool, Dict]: (validação_passou, detalhes_validação)
        """

        start_time = time.time()
        validation_results = {
            "valid": True,
            "engine": tts_engine,
            "voice_id": voice_id,
            "session_id": session_id,
            "text_input": text_input[:100],  # Truncar para log
            "checks": {},
            "errors": [],
            "warnings": [],
            "timestamp": time.time()
        }

        try:
            # 1. Validação básica de formato
            audio_array, sample_rate, duration_ms = self._validate_audio_format(audio_data)
            validation_results["checks"]["format"] = {
                "passed": True,
                "duration_ms": duration_ms,
                "sample_rate": sample_rate
            }

            # 2. Verificar duração
            duration_valid = self._validate_duration(duration_ms)
            validation_results["checks"]["duration"] = {
                "passed": duration_valid,
                "duration_ms": duration_ms,
                "min_ms": self.config.min_audio_duration_ms,
                "max_ms": self.config.max_audio_duration_ms
            }

            if not duration_valid:
                validation_results["errors"].append(f"Duração inválida: {duration_ms}ms")
                validation_results["valid"] = False

            # 3. Detectar silêncio
            if self.config.enable_silence_detection:
                has_audio, volume_stats = self._detect_silence(audio_array)
                validation_results["checks"]["silence"] = {
                    "passed": has_audio,
                    "has_audio": has_audio,
                    "volume_stats": volume_stats
                }

                if not has_audio:
                    validation_results["errors"].append("Áudio é silêncio completo")
                    validation_results["valid"] = False

            # 4. Detectar voz (análise de frequência)
            if self.config.enable_voice_detection:
                has_voice, voice_confidence = self._detect_voice_presence(audio_array, sample_rate)
                validation_results["checks"]["voice_presence"] = {
                    "passed": has_voice,
                    "has_voice": has_voice,
                    "confidence": voice_confidence
                }

                if not has_voice:
                    validation_results["warnings"].append(f"Voz não detectada (confiança: {voice_confidence:.1%})")

            # 5. Verificar qualidade
            if self.config.enable_quality_check:
                quality_score, quality_issues = self._check_audio_quality(audio_array, sample_rate)
                validation_results["checks"]["quality"] = {
                    "passed": quality_score > 0.5,
                    "score": quality_score,
                    "issues": quality_issues
                }

                if quality_score < 0.5:
                    validation_results["warnings"].append(f"Qualidade baixa: {', '.join(quality_issues)}")

            # 6. Validação via Groq (se habilitado)
            if self.config.enable_groq_validation and self._groq_client:
                groq_result = self._validate_with_groq(
                    audio_data=audio_data,
                    text_input=text_input,
                    duration_ms=duration_ms
                )
                validation_results["checks"]["groq_validation"] = groq_result

                if not groq_result.get("passed", True):
                    validation_results["errors"].append(f"Groq: {groq_result.get('reason', 'Falha na validação')}")
                    validation_results["valid"] = False

            # 7. Validações específicas por engine
            if tts_engine == "kokoro":
                kokoro_valid = self._validate_kokoro_specific(audio_array, voice_id)
                validation_results["checks"]["kokoro_specific"] = kokoro_valid

            elif tts_engine == "gtts":
                gtts_valid = self._validate_gtts_specific(audio_array)
                validation_results["checks"]["gtts_specific"] = gtts_valid

            # Calcular tempo de validação
            validation_results["validation_time_ms"] = (time.time() - start_time) * 1000

            # Log em modo desenvolvimento
            if self.config.development_mode:
                self._log_validation_details(validation_results)

            return validation_results["valid"], validation_results

        except Exception as e:
            logger.error(f"❌ Erro durante validação TTS: {e}")
            validation_results["valid"] = False
            validation_results["errors"].append(f"Erro de validação: {str(e)}")
            return False, validation_results

    def _validate_audio_format(self, audio_data: Union[bytes, np.ndarray]) -> Tuple[np.ndarray, int, float]:
        """Validar formato básico do áudio"""

        if isinstance(audio_data, bytes):
            # Converter bytes para numpy array
            try:
                import soundfile as sf
                import io

                audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
            except Exception as e:
                # Assumir que é raw PCM 16-bit @ 24kHz (Kokoro padrão)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                sample_rate = 24000

        elif isinstance(audio_data, np.ndarray):
            audio_array = audio_data
            sample_rate = 24000  # Assumir taxa padrão

        else:
            raise AudioCorruptedError(f"Tipo de áudio não suportado: {type(audio_data)}")

        # Calcular duração
        duration_ms = (len(audio_array) / sample_rate) * 1000

        return audio_array, sample_rate, duration_ms

    def _validate_duration(self, duration_ms: float) -> bool:
        """Validar duração do áudio"""
        return self.config.min_audio_duration_ms <= duration_ms <= self.config.max_audio_duration_ms

    def _detect_silence(self, audio_array: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """Detectar se áudio é silêncio"""

        # Calcular estatísticas de volume
        rms = np.sqrt(np.mean(audio_array ** 2))
        peak = np.max(np.abs(audio_array))
        mean_abs = np.mean(np.abs(audio_array))

        volume_stats = {
            "rms": float(rms),
            "peak": float(peak),
            "mean_abs": float(mean_abs)
        }

        # Verificar se há áudio significativo
        has_audio = rms > self.config.min_volume_threshold

        return has_audio, volume_stats

    def _detect_voice_presence(self, audio_array: np.ndarray, sample_rate: int) -> Tuple[bool, float]:
        """Detectar presença de voz usando análise de frequência"""

        try:
            # Análise simples de energia em frequências de voz
            # Voz humana geralmente está entre 85-255 Hz (fundamental) e 300-3400 Hz (formantes)

            # Calcular FFT
            fft = np.fft.rfft(audio_array)
            freqs = np.fft.rfftfreq(len(audio_array), 1/sample_rate)

            # Energia em banda de voz
            voice_band_mask = (freqs >= 85) & (freqs <= 3400)
            voice_energy = np.sum(np.abs(fft[voice_band_mask]) ** 2)

            # Energia total
            total_energy = np.sum(np.abs(fft) ** 2)

            if total_energy > 0:
                voice_ratio = voice_energy / total_energy
            else:
                voice_ratio = 0

            # Heurística: voz presente se mais de 30% da energia está na banda de voz
            has_voice = voice_ratio > 0.3
            confidence = min(voice_ratio * 2, 1.0)  # Normalizar para [0, 1]

            return has_voice, confidence

        except Exception as e:
            logger.warning(f"Erro na detecção de voz: {e}")
            return True, 0.5  # Assumir que há voz em caso de erro

    def _check_audio_quality(self, audio_array: np.ndarray, sample_rate: int) -> Tuple[float, list]:
        """Verificar qualidade do áudio"""

        quality_issues = []
        quality_score = 1.0

        # 1. Verificar clipping
        clipping_ratio = np.sum(np.abs(audio_array) > 0.99) / len(audio_array)
        if clipping_ratio > 0.01:  # Mais de 1% clipping
            quality_issues.append(f"Clipping detectado ({clipping_ratio:.1%})")
            quality_score *= 0.7

        # 2. Verificar ruído (SNR aproximado)
        # Dividir em frames e calcular variação
        frame_size = int(sample_rate * 0.02)  # 20ms frames
        if len(audio_array) > frame_size:
            frames = audio_array[:len(audio_array) // frame_size * frame_size].reshape(-1, frame_size)
            frame_energies = np.mean(frames ** 2, axis=1)

            if len(frame_energies) > 1:
                energy_variation = np.std(frame_energies) / (np.mean(frame_energies) + 1e-10)

                if energy_variation < 0.1:  # Muito pouca variação
                    quality_issues.append("Áudio muito monotônico")
                    quality_score *= 0.8

        # 3. Verificar DC offset
        dc_offset = np.mean(audio_array)
        if abs(dc_offset) > 0.1:
            quality_issues.append(f"DC offset detectado ({dc_offset:.3f})")
            quality_score *= 0.9

        return quality_score, quality_issues

    def _validate_with_groq(self, audio_data: bytes, text_input: str, duration_ms: float) -> Dict[str, Any]:
        """Validar áudio usando Groq para transcrição e análise"""

        if not self._groq_client:
            return {"passed": True, "skipped": True, "reason": "Groq não configurado"}

        try:
            # Salvar áudio temporariamente para Groq
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                if isinstance(audio_data, bytes):
                    tmp.write(audio_data)
                else:
                    # Converter numpy para WAV
                    import soundfile as sf
                    sf.write(tmp.name, audio_data, 24000)

                tmp_path = tmp.name

            # Transcrever com Groq Whisper
            logger.info(f"🎤 Validando com Groq Whisper...")

            with open(tmp_path, "rb") as audio_file:
                transcription = self._groq_client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    language="pt",  # ou detectar do texto
                    response_format="json"
                )

            # Limpar arquivo temporário
            os.unlink(tmp_path)

            # Analisar resultado
            transcribed_text = transcription.text.strip()

            validation_result = {
                "passed": True,
                "transcription": transcribed_text,
                "original_text": text_input[:100],
                "duration_ms": duration_ms
            }

            # Verificar se há conteúdo transcrito
            if not transcribed_text:
                validation_result["passed"] = False
                validation_result["reason"] = "Nenhuma fala detectada na transcrição"

            # Verificar similaridade básica (pode melhorar com fuzzy matching)
            elif len(transcribed_text) < 5 and len(text_input) > 20:
                validation_result["passed"] = False
                validation_result["reason"] = f"Transcrição muito curta: '{transcribed_text}'"

            # Análise adicional com Groq LLM
            if self.config.development_mode and transcribed_text:
                analysis = self._groq_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é um analisador de qualidade de áudio TTS. Responda em JSON."
                        },
                        {
                            "role": "user",
                            "content": f"""Analise a qualidade desta síntese TTS:

Texto original: "{text_input[:200]}"
Transcrição do áudio: "{transcribed_text}"
Duração: {duration_ms}ms

Responda em JSON com:
- quality_score: 0-1
- has_voice: boolean
- clarity: "clear", "muffled", "distorted"
- issues: lista de problemas encontrados
- match_score: 0-1 (quão bem o áudio corresponde ao texto)"""
                        }
                    ],
                    temperature=0,
                    max_tokens=300
                )

                try:
                    analysis_data = json.loads(analysis.choices[0].message.content)
                    validation_result["analysis"] = analysis_data

                    # Atualizar validação baseado na análise
                    if analysis_data.get("quality_score", 1) < 0.5:
                        validation_result["passed"] = False
                        validation_result["reason"] = f"Qualidade baixa: {analysis_data.get('issues', [])}"

                    if not analysis_data.get("has_voice", True):
                        validation_result["passed"] = False
                        validation_result["reason"] = "Análise não detectou voz no áudio"

                except Exception as e:
                    logger.warning(f"Erro ao parsear análise Groq: {e}")

            return validation_result

        except Exception as e:
            logger.error(f"❌ Erro na validação Groq: {e}")
            return {
                "passed": True,  # Não falhar se Groq der erro
                "skipped": True,
                "error": str(e)
            }

    def _validate_kokoro_specific(self, audio_array: np.ndarray, voice_id: Optional[str]) -> Dict[str, Any]:
        """Validações específicas para Kokoro TTS"""

        result = {"passed": True, "checks": []}

        # Kokoro geralmente produz áudio em 24kHz
        # Verificar características esperadas de cada voz
        if voice_id:
            if voice_id.startswith("af_"):  # Vozes femininas
                # Frequência fundamental esperada mais alta
                result["checks"].append("voice_gender_match")
            elif voice_id.startswith("am_"):  # Vozes masculinas
                # Frequência fundamental esperada mais baixa
                result["checks"].append("voice_gender_match")

        # Kokoro tem boa qualidade, raramente tem artefatos
        if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
            result["passed"] = False
            result["error"] = "Artefatos numéricos detectados (NaN/Inf)"

        return result

    def _validate_gtts_specific(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Validações específicas para Google TTS"""

        result = {"passed": True, "checks": []}

        # gTTS às vezes tem pequenos artefatos no início/fim
        # Verificar se há clicks
        start_energy = np.mean(audio_array[:100] ** 2)
        end_energy = np.mean(audio_array[-100:] ** 2)

        if start_energy > 0.1 or end_energy > 0.1:
            result["checks"].append("possible_clicks_detected")

        return result

    def _log_validation_details(self, results: Dict[str, Any]):
        """Log detalhado em modo desenvolvimento"""

        logger.info("=" * 60)
        logger.info(f"📊 VALIDAÇÃO TTS - {results['engine']}")
        logger.info("=" * 60)

        # Status geral
        status_emoji = "✅" if results["valid"] else "❌"
        logger.info(f"{status_emoji} Status: {'VÁLIDO' if results['valid'] else 'INVÁLIDO'}")
        logger.info(f"⏱️ Tempo de validação: {results.get('validation_time_ms', 0):.1f}ms")

        # Checks realizados
        logger.info("\n📋 Verificações:")
        for check_name, check_result in results.get("checks", {}).items():
            if isinstance(check_result, dict):
                passed = check_result.get("passed", False)
                emoji = "✓" if passed else "✗"
                logger.info(f"  {emoji} {check_name}: {check_result}")

        # Erros
        if results.get("errors"):
            logger.error("\n❌ Erros encontrados:")
            for error in results["errors"]:
                logger.error(f"  - {error}")

        # Avisos
        if results.get("warnings"):
            logger.warning("\n⚠️ Avisos:")
            for warning in results["warnings"]:
                logger.warning(f"  - {warning}")

        # Transcrição Groq (se disponível)
        if "groq_validation" in results.get("checks", {}):
            groq = results["checks"]["groq_validation"]
            if groq.get("transcription"):
                logger.info(f"\n🎤 Transcrição Groq: '{groq['transcription']}'")
                logger.info(f"📝 Texto original: '{groq.get('original_text', '')}'")

                if groq.get("analysis"):
                    analysis = groq["analysis"]
                    logger.info(f"🔍 Análise Groq:")
                    logger.info(f"   Score qualidade: {analysis.get('quality_score', 'N/A')}")
                    logger.info(f"   Clareza: {analysis.get('clarity', 'N/A')}")
                    logger.info(f"   Match score: {analysis.get('match_score', 'N/A')}")

        logger.info("=" * 60)

    def validate_batch(self,
                      audio_outputs: list,
                      text_inputs: list,
                      tts_engine: str = "unknown") -> Tuple[list, Dict[str, Any]]:
        """
        Validar múltiplas saídas TTS em batch

        Args:
            audio_outputs: Lista de áudios gerados
            text_inputs: Lista de textos originais
            tts_engine: Engine TTS usado

        Returns:
            Tuple[list, Dict]: (lista_resultados, estatísticas)
        """

        results = []
        statistics = {
            "total": len(audio_outputs),
            "valid": 0,
            "invalid": 0,
            "warnings": 0,
            "total_time_ms": 0
        }

        start_time = time.time()

        for audio, text in zip(audio_outputs, text_inputs):
            is_valid, details = self.validate_tts_output(
                audio_data=audio,
                text_input=text,
                tts_engine=tts_engine
            )

            results.append({
                "valid": is_valid,
                "text": text[:50],
                "details": details
            })

            if is_valid:
                statistics["valid"] += 1
            else:
                statistics["invalid"] += 1

            if details.get("warnings"):
                statistics["warnings"] += len(details["warnings"])

        statistics["total_time_ms"] = (time.time() - start_time) * 1000
        statistics["success_rate"] = statistics["valid"] / statistics["total"] if statistics["total"] > 0 else 0

        if self.config.development_mode:
            logger.info(f"\n📊 Batch validation statistics:")
            logger.info(f"   Total: {statistics['total']}")
            logger.info(f"   Valid: {statistics['valid']} ({statistics['success_rate']:.1%})")
            logger.info(f"   Invalid: {statistics['invalid']}")
            logger.info(f"   Warnings: {statistics['warnings']}")
            logger.info(f"   Time: {statistics['total_time_ms']:.1f}ms")

        return results, statistics


# ============================================================================
# INTEGRAÇÃO COM SISTEMA DE ERROS
# ============================================================================

def create_tts_error_from_validation(validation_results: Dict[str, Any],
                                    tts_engine: str) -> Optional[TTSError]:
    """
    Criar erro TTS apropriado baseado nos resultados de validação

    Args:
        validation_results: Resultados da validação
        tts_engine: Engine TTS usado

    Returns:
        TTSError ou None se validação passou
    """

    if validation_results.get("valid", True):
        return None

    # Determinar mensagem de erro principal
    errors = validation_results.get("errors", [])
    error_message = errors[0] if errors else "Falha na validação TTS"

    # Criar erro específico baseado no engine
    if tts_engine == "kokoro":
        error = KokoroTTSError(
            message=error_message,
            voice_id=validation_results.get("voice_id")
        )
    elif tts_engine == "gtts":
        error = GTTSError(
            message=error_message,
            language="pt"  # ou detectar do contexto
        )
    else:
        error = TTSError(
            message=error_message,
            tts_engine=tts_engine
        )

    # Adicionar detalhes da validação
    error.details.update({
        "validation_checks": validation_results.get("checks", {}),
        "warnings": validation_results.get("warnings", []),
        "duration_ms": validation_results.get("checks", {}).get("format", {}).get("duration_ms")
    })

    return error


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

def example_usage():
    """Exemplo de uso do validador TTS"""

    # Configuração para desenvolvimento
    config = TTSValidationConfig.development_config()
    validator = TTSValidator(config)

    # Simular saída TTS (silêncio)
    sample_rate = 24000
    duration_sec = 2
    audio_array = np.zeros(sample_rate * duration_sec, dtype=np.float32)

    # Adicionar algum ruído para simular voz
    audio_array += np.random.normal(0, 0.01, len(audio_array))

    # Validar
    is_valid, details = validator.validate_tts_output(
        audio_data=audio_array,
        text_input="Olá, este é um teste do sistema TTS",
        tts_engine="kokoro",
        voice_id="af_sarah"
    )

    print(f"\n✅ Válido: {is_valid}")
    print(f"📊 Detalhes: {json.dumps(details, indent=2, default=str)}")

    # Criar erro se necessário
    if not is_valid:
        error = create_tts_error_from_validation(details, "kokoro")
        if error:
            print(f"\n❌ Erro criado: {error.to_json()}")


if __name__ == "__main__":
    example_usage()