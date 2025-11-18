#!/usr/bin/env python3
"""
Validador Completo de Áudio para Pipeline Speech-to-Speech
Valida tanto áudio de entrada do usuário quanto saída do TTS
Integração com Groq para transcrição e análise em modo desenvolvimento
"""

import os
import logging
import tempfile
import numpy as np
import soundfile as sf
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path

# Importar exceções customizadas
from .error_handler import UltravoxError, ErrorSeverity, ErrorCategory, ValidationError, ProcessingError
    TTSError, TTSNoVoiceError, TTSSilenceError, TTSQualityError,
    ValidationError, AudioTooShortError, AudioTooLongError,
    AudioCorruptedError, InvalidAudioDataTypeError
)

# Configurar logging
logger = logging.getLogger(__name__)

# Carregar configurações do ambiente


class AudioPipelineValidator:
    """Validador completo para áudio de entrada e saída da pipeline Speech-to-Speech"""

    def __init__(self, development_mode: Optional[bool] = None):
        """
        Inicializar validador

        Args:
            development_mode: Se None, lê do .env, senão usa valor fornecido
        """
        # Ler configurações do ambiente
        self.development_mode = development_mode if development_mode is not None else (
            os.getenv("ENVIRONMENT", "production") == "development"
        )
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # Configurações de desenvolvimento
        self.enable_transcription = self.development_mode and os.getenv("DEV_TTS_TRANSCRIPTION", "false").lower() == "true"
        self.enable_metrics = self.development_mode and os.getenv("DEV_SEND_METRICS", "false").lower() == "true"
        self.verbose_logs = self.development_mode and os.getenv("DEV_VERBOSE_LOGS", "false").lower() == "true"

        # Cliente Groq (apenas em modo desenvolvimento)
        self.groq_client = None
        if self.development_mode and self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info("✅ Groq cliente inicializado para validação em desenvolvimento")
            except ImportError:
                logger.warning("❌ Groq não instalado. Instale com: pip install groq")
            except Exception as e:
                logger.error(f"❌ Erro ao inicializar Groq: {e}")

        # Configurações de validação
        self.min_duration_ms = 100  # Mínimo de 100ms de áudio
        self.max_duration_ms = 60000  # Máximo de 60 segundos
        self.target_sample_rate = 16000  # Taxa de amostragem alvo
        self.min_voice_frequency = 85  # Hz mínimo para voz humana
        self.max_voice_frequency = 3000  # Hz máximo para voz humana
        self.silence_threshold = 0.01  # Threshold para detectar silêncio

        if self.verbose_logs:
            logger.info(f"🔧 AudioPipelineValidator inicializado")
            logger.info(f"   Modo: {'DESENVOLVIMENTO' if self.development_mode else 'PRODUÇÃO'}")
            logger.info(f"   Groq: {'Habilitado' if self.groq_client else 'Desabilitado'}")
            logger.info(f"   Transcrição: {'Habilitada' if self.enable_transcription else 'Desabilitada'}")

    def validate_user_input(self,
                           audio_data: Union[np.ndarray, bytes, str],
                           session_id: Optional[str] = None,
                           expected_language: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validar áudio de entrada do usuário

        Args:
            audio_data: Dados de áudio (numpy array, bytes ou caminho)
            session_id: ID da sessão
            expected_language: Idioma esperado (para validação com Groq)

        Returns:
            Tuple com (áudio validado, metadados)
        """

        if self.verbose_logs:
            logger.info(f"🎤 Validando entrada do usuário (sessão: {session_id})")

        # Converter para numpy array se necessário
        audio_array, sample_rate = self._convert_to_array(audio_data)

        # Validar duração
        duration_ms = len(audio_array) / sample_rate * 1000

        if duration_ms < self.min_duration_ms:
            raise AudioTooShortError(duration_ms, self.min_duration_ms)

        if duration_ms > self.max_duration_ms:
            raise AudioTooLongError(duration_ms, self.max_duration_ms)

        # Verificar se tem voz (análise básica)
        has_voice, voice_confidence = self._detect_voice_basic(audio_array, sample_rate)

        # Converter taxa de amostragem se necessário
        if sample_rate != self.target_sample_rate:
            import scipy.signal
            num_samples = int(len(audio_array) * self.target_sample_rate / sample_rate)
            audio_array = scipy.signal.resample(audio_array, num_samples)
            sample_rate = self.target_sample_rate

            if self.verbose_logs:
                logger.info(f"   Reamostragem: {sample_rate}Hz → {self.target_sample_rate}Hz")

        # Normalizar áudio
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val

        # Metadados básicos
        metadata = {
            "duration_ms": duration_ms,
            "sample_rate": sample_rate,
            "has_voice_basic": has_voice,
            "voice_confidence": voice_confidence,
            "normalized": True,
            "session_id": session_id
        }

        # Validação avançada com Groq em modo desenvolvimento
        if self.development_mode and self.groq_client and self.enable_transcription:
            groq_result = self._validate_with_groq(
                audio_array,
                expected_text=None,
                duration_ms=duration_ms,
                context="user_input",
                expected_language=expected_language
            )
            metadata["groq_validation"] = groq_result

            # Verificar se foi detectada voz
            if not groq_result.get("has_voice", False):
                logger.warning(f"⚠️ Groq não detectou voz no áudio do usuário")
                if self.development_mode:
                    # Em desenvolvimento, apenas loga mas não bloqueia
                    metadata["warning"] = "No voice detected by Groq"
                else:
                    raise TTSNoVoiceError(transcription=groq_result.get("transcription", ""))

        if self.verbose_logs:
            logger.info(f"   ✅ Entrada validada: {duration_ms:.1f}ms, voz: {has_voice}")

        return audio_array, metadata

    def validate_tts_output(self,
                           audio_data: Union[np.ndarray, bytes],
                           text_input: str,
                           tts_engine: str = "unknown",
                           voice_id: Optional[str] = None,
                           session_id: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validar saída do TTS

        Args:
            audio_data: Dados de áudio gerado pelo TTS
            text_input: Texto original usado para gerar o áudio
            tts_engine: Engine TTS usado (kokoro, gtts, etc.)
            voice_id: ID da voz usada
            session_id: ID da sessão

        Returns:
            Tuple com (áudio validado, metadados)
        """

        if self.verbose_logs:
            logger.info(f"🔊 Validando saída do TTS (engine: {tts_engine}, voz: {voice_id})")

        # Converter para numpy array
        audio_array, sample_rate = self._convert_to_array(audio_data)

        # Validar duração
        duration_ms = len(audio_array) / sample_rate * 1000

        if duration_ms < 50:  # TTS muito curto é suspeito
            logger.error(f"❌ TTS gerou áudio muito curto: {duration_ms}ms")
            raise TTSError(f"TTS output too short: {duration_ms}ms", engine=tts_engine)

        # Detectar silêncio
        is_silent = self._detect_silence(audio_array)
        if is_silent:
            logger.error(f"❌ TTS gerou apenas silêncio")
            raise TTSSilenceError(tts_engine, voice_id)

        # Verificar qualidade básica
        quality_issues = self._check_audio_quality(audio_array, sample_rate)

        # Metadados básicos
        metadata = {
            "duration_ms": duration_ms,
            "sample_rate": sample_rate,
            "tts_engine": tts_engine,
            "voice_id": voice_id,
            "text_input": text_input[:100],  # Primeiros 100 chars
            "is_silent": is_silent,
            "quality_issues": quality_issues,
            "session_id": session_id
        }

        # Validação avançada com Groq em modo desenvolvimento
        if self.development_mode and self.groq_client and self.enable_transcription:
            groq_result = self._validate_with_groq(
                audio_array,
                expected_text=text_input,
                duration_ms=duration_ms,
                context="tts_output",
                tts_engine=tts_engine,
                voice_id=voice_id
            )
            metadata["groq_validation"] = groq_result

            # Verificar qualidade
            if groq_result.get("quality_score", 0) < 3:
                logger.warning(f"⚠️ Baixa qualidade detectada pelo Groq: {groq_result.get('quality_analysis')}")
                if not self.development_mode:  # Em produção, lançar erro
                    raise TTSQualityError(
                        quality_score=groq_result.get("quality_score", 0),
                        issues=groq_result.get("quality_analysis", "Unknown issues")
                    )

        # Normalizar se necessário
        max_val = np.max(np.abs(audio_array))
        if max_val > 1.0:
            audio_array = audio_array / max_val
            metadata["normalized"] = True

        if self.verbose_logs:
            logger.info(f"   ✅ TTS validado: {duration_ms:.1f}ms, qualidade: OK")

        return audio_array, metadata

    def _convert_to_array(self, audio_data: Union[np.ndarray, bytes, str]) -> Tuple[np.ndarray, int]:
        """Converter diferentes formatos para numpy array"""

        if isinstance(audio_data, np.ndarray):
            # Assumir 16kHz se já é array
            return audio_data, self.target_sample_rate

        elif isinstance(audio_data, (str, Path)):
            # Carregar de arquivo
            try:
                audio, sr = sf.read(audio_data)
                return audio, sr
            except Exception as e:
                raise AudioCorruptedError(f"Failed to read audio file: {e}")

        elif isinstance(audio_data, bytes):
            # Converter bytes para array
            try:
                # Salvar temporariamente e carregar
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_data)
                    temp_path = f.name

                audio, sr = sf.read(temp_path)
                os.unlink(temp_path)
                return audio, sr
            except Exception as e:
                raise AudioCorruptedError(f"Failed to decode audio bytes: {e}")

        else:
            raise InvalidAudioDataTypeError(type(audio_data).__name__, "numpy.ndarray, bytes, or str")

    def _detect_voice_basic(self, audio: np.ndarray, sample_rate: int) -> Tuple[bool, float]:
        """Detecção básica de voz usando análise de frequência"""

        # Análise FFT
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)

        # Encontrar frequências dominantes
        voice_range_mask = (freqs >= self.min_voice_frequency) & (freqs <= self.max_voice_frequency)
        voice_energy = np.sum(fft[voice_range_mask])
        total_energy = np.sum(fft)

        if total_energy > 0:
            voice_ratio = voice_energy / total_energy
        else:
            voice_ratio = 0

        # Considerar voz se mais de 30% da energia está na faixa de voz
        has_voice = voice_ratio > 0.3
        confidence = min(voice_ratio * 2, 1.0)  # Escalar para 0-1

        return has_voice, confidence

    def _detect_silence(self, audio: np.ndarray) -> bool:
        """Detectar se o áudio é apenas silêncio"""

        # RMS do sinal
        rms = np.sqrt(np.mean(audio**2))
        return rms < self.silence_threshold

    def _check_audio_quality(self, audio: np.ndarray, sample_rate: int) -> list:
        """Verificar problemas de qualidade no áudio"""

        issues = []

        # Verificar clipping
        if np.any(np.abs(audio) >= 0.99):
            issues.append("clipping_detected")

        # Verificar DC offset
        dc_offset = np.mean(audio)
        if abs(dc_offset) > 0.1:
            issues.append(f"dc_offset_{dc_offset:.2f}")

        # Verificar se é monotônico (apenas um valor)
        if np.std(audio) < 0.001:
            issues.append("monotonic_signal")

        return issues

    def validate_ultravox_response(self,
                                  user_question: str,
                                  ultravox_response: str,
                                  session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validar resposta do Ultravox usando Groq LLM

        Args:
            user_question: Pergunta original do usuário
            ultravox_response: Resposta gerada pelo Ultravox
            session_id: ID da sessão

        Returns:
            Dict com análise de coerência e qualidade
        """
        metadata = {
            "validation_type": "ultravox_response",
            "session_id": session_id,
            "user_question": user_question[:100],  # Primeiros 100 chars
            "response_length": len(ultravox_response)
        }

        if self.verbose_logs:
            logger.info(f"🤖 Validando resposta Ultravox (sessão: {session_id})")
            logger.info(f"   Pergunta: {user_question[:50]}...")
            logger.info(f"   Resposta: {ultravox_response[:50]}...")

        # Validar com Groq LLM se disponível
        if self.development_mode and self.groq_client:
            try:
                # Usar o modelo mais inteligente do Groq: llama-3.3-70b-versatile
                import json
                completion = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": """Você é um validador de respostas de IA. Analise a pergunta do usuário e a resposta fornecida.
                            Avalie:
                            1. Coerência: A resposta faz sentido com a pergunta?
                            2. Completude: A resposta responde adequadamente a pergunta?
                            3. Qualidade: A resposta é útil e bem formulada?

                            Responda em formato JSON com:
                            - coherence_score: 0-10
                            - completeness_score: 0-10
                            - quality_score: 0-10
                            - is_valid: true/false (se a resposta é aceitável)
                            - issues: lista de problemas encontrados (se houver)
                            - analysis: breve análise em português"""
                        },
                        {
                            "role": "user",
                            "content": f"Pergunta: {user_question}\n\nResposta: {ultravox_response}"
                        }
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )

                # Parsear resposta JSON
                llm_analysis = json.loads(completion.choices[0].message.content)

                metadata["llm_validation"] = {
                    "model": "llama-3.3-70b-versatile",
                    "coherence_score": llm_analysis.get("coherence_score", 0),
                    "completeness_score": llm_analysis.get("completeness_score", 0),
                    "quality_score": llm_analysis.get("quality_score", 0),
                    "is_valid": llm_analysis.get("is_valid", False),
                    "issues": llm_analysis.get("issues", []),
                    "analysis": llm_analysis.get("analysis", "")
                }

                # Calcular score geral
                scores = [
                    llm_analysis.get("coherence_score", 0),
                    llm_analysis.get("completeness_score", 0),
                    llm_analysis.get("quality_score", 0)
                ]
                metadata["overall_score"] = sum(scores) / len(scores) if scores else 0
                metadata["is_valid"] = llm_analysis.get("is_valid", False)

                if self.verbose_logs:
                    logger.info(f"   🤖 [Groq LLM] Análise de resposta:")
                    logger.info(f"      Coerência: {llm_analysis.get('coherence_score', 0)}/10")
                    logger.info(f"      Completude: {llm_analysis.get('completeness_score', 0)}/10")
                    logger.info(f"      Qualidade: {llm_analysis.get('quality_score', 0)}/10")
                    logger.info(f"      Válida: {llm_analysis.get('is_valid', False)}")
                    if llm_analysis.get("issues"):
                        logger.info(f"      Problemas: {', '.join(llm_analysis['issues'])}")
                    logger.info(f"      Análise: {llm_analysis.get('analysis', '')[:100]}")

            except Exception as e:
                logger.error(f"❌ Erro na validação LLM Groq: {e}")
                metadata["llm_validation"] = {"error": str(e)}
        else:
            if self.verbose_logs:
                logger.info("   ⚠️ Validação LLM não disponível (requer ENVIRONMENT=development e GROQ_API_KEY)")

        return metadata

    def _validate_with_groq(self,
                           audio: np.ndarray,
                           expected_text: Optional[str] = None,
                           duration_ms: float = 0,
                           context: str = "unknown",
                           **kwargs) -> Dict[str, Any]:
        """Validar áudio usando Groq API"""

        if not self.groq_client:
            return {"error": "Groq client not initialized"}

        result = {
            "context": context,
            "duration_ms": duration_ms
        }

        try:
            # Salvar áudio temporariamente para enviar ao Groq
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, self.target_sample_rate)
                temp_path = f.name

            # Transcrever com Whisper
            with open(temp_path, "rb") as audio_file:
                transcription = self.groq_client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    response_format="json",
                    language=kwargs.get("expected_language")  # Se fornecido
                )

            os.unlink(temp_path)

            # Processar transcrição
            transcribed_text = transcription.text.strip()
            result["transcription"] = transcribed_text
            result["has_voice"] = len(transcribed_text) > 0

            # Análise com LLM se foi transcrito algo
            if transcribed_text and expected_text:
                # Comparar com texto esperado
                analysis_prompt = f"""Analyze this TTS output quality:

Expected text: "{expected_text}"
Transcribed text: "{transcribed_text}"
Duration: {duration_ms}ms
Context: {context}

Please provide:
1. Quality score (1-5, where 5 is perfect)
2. Brief analysis of any issues
3. Whether the transcription matches the expected text

Format your response as JSON with keys: quality_score, matches_expected, analysis"""

                llm_response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )

                import json
                analysis = json.loads(llm_response.choices[0].message.content)
                result.update(analysis)

            elif context == "user_input" and transcribed_text:
                # Para entrada do usuário, apenas verificar se tem conteúdo válido
                result["quality_score"] = 5 if len(transcribed_text) > 3 else 3
                result["analysis"] = "User input detected with voice"

            elif not transcribed_text:
                # Nenhuma voz detectada
                result["quality_score"] = 0
                result["analysis"] = "No voice detected in audio"
                result["has_voice"] = False

            # Log em desenvolvimento
            if self.verbose_logs:
                logger.info(f"   🤖 Groq análise ({context}):")
                logger.info(f"      Transcrição: {transcribed_text[:50]}..." if transcribed_text else "      Sem transcrição")
                if "quality_score" in result:
                    logger.info(f"      Qualidade: {result['quality_score']}/5")
                if "analysis" in result:
                    logger.info(f"      Análise: {result['analysis']}")

        except Exception as e:
            logger.error(f"❌ Erro na validação Groq: {e}")
            result["error"] = str(e)

        return result


# Função de conveniência para criar validador global
_global_validator: Optional[AudioPipelineValidator] = None


def get_audio_validator() -> AudioPipelineValidator:
    """Obter instância global do validador de áudio"""
    global _global_validator
    if _global_validator is None:
        _global_validator = AudioPipelineValidator()
    return _global_validator


def set_audio_validator(validator: AudioPipelineValidator):
    """Definir instância global do validador"""
    global _global_validator
    _global_validator = validator
    logger.info("🎵 Validador de áudio global definido")