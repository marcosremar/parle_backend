#!/usr/bin/env python3
"""
Validador Robusto de Formato de Áudio
Sistema completo de validação para pipeline Speech-to-Speech
"""

import os
import tempfile
import logging
from typing import Union, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
import base64
from src.core.exceptions import UltravoxError, wrap_exception

# Importar exceções customizadas
try:
    from .error_handler import (
        UltravoxError, ErrorSeverity, ErrorCategory, ValidationError, ProcessingError,
        InvalidAudioDataTypeError, AudioTooShortError, AudioTooLongError,
        AudioCorruptedError, UnsupportedValidationError
    )
except ImportError:
    # Fallback se error_handler não estiver disponível
    class ValidationError(Exception):
        """Exception raised for validation error errors"""
        pass
    class InvalidAudioDataTypeError(Exception):
        """Exception raised for invalid audio data type error errors"""
        pass
    class AudioTooShortError(Exception):
        """Exception raised for audio too short error errors"""
        pass
    class AudioTooLongError(Exception):
        """Exception raised for audio too long error errors"""
        pass
    class AudioCorruptedError(Exception):
        """Exception raised for audio corrupted error errors"""
        pass
    class UnsupportedValidationError(Exception):
        """Exception raised for unsupported validation error errors"""
        pass

# Bibliotecas de áudio (com fallbacks)
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioValidationConfig:
    """Configuração para validação de áudio"""

    def __init__(self,
                 target_sample_rate: int = 16000,
                 min_duration_ms: float = 100,
                 max_duration_ms: float = 30000,
                 target_dtype: str = "float32",
                 allowed_channels: tuple = (1, 2),
                 auto_convert: bool = True,
                 strict_mode: bool = False):
        """
        Args:
            target_sample_rate: Taxa de amostragem alvo (Hz)
            min_duration_ms: Duração mínima em milissegundos
            max_duration_ms: Duração máxima em milissegundos
            target_dtype: Tipo de dados alvo
            allowed_channels: Canais permitidos (1=mono, 2=stereo)
            auto_convert: Converter automaticamente se possível
            strict_mode: Modo estrito (falha se conversão necessária)
        """
        self.target_sample_rate = target_sample_rate
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.target_dtype = target_dtype
        self.allowed_channels = allowed_channels
        self.auto_convert = auto_convert
        self.strict_mode = strict_mode

    @classmethod
    def ultravox_config(cls) -> 'AudioValidationConfig':
        """Configuração otimizada para Ultravox"""
        return cls(
            target_sample_rate=16000,
            min_duration_ms=100,
            max_duration_ms=30000,
            target_dtype="float32",
            allowed_channels=(1,),  # Apenas mono
            auto_convert=True,
            strict_mode=False
        )

    @classmethod
    def strict_config(cls) -> 'AudioValidationConfig':
        """Configuração estrita (sem conversões automáticas)"""
        return cls(
            target_sample_rate=16000,
            min_duration_ms=100,
            max_duration_ms=30000,
            target_dtype="float32",
            allowed_channels=(1,),
            auto_convert=False,
            strict_mode=True
        )


class AudioValidator:
    """Validador robusto de formato de áudio"""

    # Formatos suportados
    SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.opus'}
    SUPPORTED_DTYPES = {'float32', 'float64', 'int16', 'int32'}

    def __init__(self, config: Optional[AudioValidationConfig] = None):
        """
        Args:
            config: Configuração de validação
        """
        self.config = config or AudioValidationConfig.ultravox_config()
        self._check_dependencies()

    def _check_dependencies(self):
        """Verificar dependências disponíveis"""
        if not any([SOUNDFILE_AVAILABLE, PYDUB_AVAILABLE, LIBROSA_AVAILABLE]):
            raise ValidationError(
                "Nenhuma biblioteca de áudio disponível. Instale: pip install soundfile pydub librosa",
                field="audio_libraries",
                value="None"
            )

    def validate_and_convert(self,
                           audio_input: Union[str, np.ndarray, bytes],
                           session_id: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validar e converter áudio para formato padrão

        Args:
            audio_input: Entrada de áudio (arquivo, array, bytes, base64)
            session_id: ID da sessão para logging

        Returns:
            Tuple[np.ndarray, Dict]: (áudio processado, metadados)

        Raises:
            ValidationError: Erro de formato de áudio
            ValidationError: Erro de validação
        """
        logger.info(f"🔍 Validando áudio (sessão: {session_id or 'unknown'})")

        try:
            # 1. Detectar tipo de entrada
            input_type = self._detect_input_type(audio_input)
            logger.debug(f"Tipo de entrada detectado: {input_type}")

            # 2. Carregar áudio
            audio_data, original_sr, metadata = self._load_audio(audio_input, input_type)

            # 3. Validar propriedades básicas
            self._validate_basic_properties(audio_data, original_sr, metadata)

            # 4. Converter para formato alvo
            processed_audio = self._convert_to_target_format(audio_data, original_sr)

            # 5. Validações finais
            final_metadata = self._validate_final_audio(processed_audio)

            # 6. Retornar resultado
            result_metadata = {
                **metadata,
                **final_metadata,
                "input_type": input_type,
                "validation_passed": True,
                "conversions_applied": self._get_conversions_applied(metadata, final_metadata)
            }

            logger.info(f"✅ Áudio validado com sucesso (duração: {final_metadata['duration_ms']:.1f}ms)")
            return processed_audio, result_metadata

        except Exception as e:
            if isinstance(e, (ValidationError, ValidationError)):
                raise
            else:
                # Converter erro genérico em erro específico
                raise AudioCorruptedError(f"Erro inesperado durante validação: {str(e)}")

    def _detect_input_type(self, audio_input: Union[str, np.ndarray, bytes]) -> str:
        """Detectar tipo de entrada"""

        if isinstance(audio_input, np.ndarray):
            return "numpy_array"

        if isinstance(audio_input, bytes):
            return "raw_bytes"

        if isinstance(audio_input, str):
            # Verificar se é arquivo
            if os.path.isfile(audio_input):
                ext = Path(audio_input).suffix.lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    return "audio_file"
                else:
                    raise UnsupportedValidationError(ext, list(self.SUPPORTED_EXTENSIONS))

            # Verificar se é base64
            if self._is_base64(audio_input):
                return "base64"

            # Se chegou aqui, formato não reconhecido
            raise UnsupportedValidationError("string_unknown", ["audio_file", "base64"])

        raise UnsupportedValidationError(str(type(audio_input)), ["str", "np.ndarray", "bytes"])

    def _is_base64(self, s: str) -> bool:
        """Verificar se string é base64 válida"""
        try:
            if len(s) < 100:  # Base64 de áudio deve ser longo
                return False
            decoded = base64.b64decode(s)
            return base64.b64encode(decoded).decode() == s
        except Exception as e:
            return False

    def _load_audio(self, audio_input: Union[str, np.ndarray, bytes], input_type: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Carregar áudio baseado no tipo de entrada"""

        if input_type == "numpy_array":
            return self._handle_numpy_array(audio_input)

        elif input_type == "raw_bytes":
            return self._handle_raw_bytes(audio_input)

        elif input_type == "audio_file":
            return self._handle_audio_file(audio_input)

        elif input_type == "base64":
            # Decodificar base64 e tratar como bytes
            audio_bytes = base64.b64decode(audio_input)
            return self._handle_raw_bytes(audio_bytes)

        else:
            raise UnsupportedValidationError(input_type, ["numpy_array", "raw_bytes", "audio_file", "base64"])

    def _handle_numpy_array(self, audio: np.ndarray) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Processar numpy array"""

        metadata = {
            "original_shape": audio.shape,
            "original_dtype": str(audio.dtype),
            "estimated_sample_rate": self.config.target_sample_rate  # Assumir taxa padrão
        }

        # Verificar formato
        if audio.ndim > 2:
            raise ValidationError(f"Array com muitas dimensões: {audio.ndim}. Máximo: 2")

        if audio.ndim == 2 and audio.shape[1] > 2:
            raise ValidationError(f"Muitos canais: {audio.shape[1]}. Máximo: 2")

        # Converter para mono se stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
            metadata["converted_to_mono"] = True

        return audio, self.config.target_sample_rate, metadata

    def _handle_raw_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Processar bytes de áudio"""

        # Salvar em arquivo temporário para processamento
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self._handle_audio_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def _handle_audio_file(self, file_path: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Processar arquivo de áudio"""

        metadata = {
            "file_path": file_path,
            "file_size_bytes": os.path.getsize(file_path),
            "file_extension": Path(file_path).suffix.lower()
        }

        # Tentar carregar com diferentes bibliotecas
        if SOUNDFILE_AVAILABLE:
            try:
                audio, sr = sf.read(file_path)
                metadata["loader_used"] = "soundfile"
                return self._process_loaded_audio(audio, sr, metadata)
            except Exception as e:
                logger.debug(f"Soundfile falhou: {e}")

        if LIBROSA_AVAILABLE:
            try:
                audio, sr = librosa.load(file_path, sr=None)
                metadata["loader_used"] = "librosa"
                return self._process_loaded_audio(audio, sr, metadata)
            except Exception as e:
                logger.debug(f"Librosa falhou: {e}")

        if PYDUB_AVAILABLE:
            try:
                audio_segment = AudioSegment.from_file(file_path)
                sr = audio_segment.frame_rate
                audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

                # Normalizar para [-1, 1]
                if audio_segment.sample_width == 2:  # 16-bit
                    audio = audio / 32768.0
                elif audio_segment.sample_width == 4:  # 32-bit
                    audio = audio / 2147483648.0

                # Converter stereo para mono
                if audio_segment.channels == 2:
                    audio = audio.reshape((-1, 2)).mean(axis=1)

                metadata["loader_used"] = "pydub"
                metadata["original_channels"] = audio_segment.channels
                metadata["original_sample_width"] = audio_segment.sample_width

                return self._process_loaded_audio(audio, sr, metadata)

            except Exception as e:
                logger.debug(f"Pydub falhou: {e}")

        raise AudioCorruptedError(f"Não foi possível carregar o arquivo: {file_path}")

    def _process_loaded_audio(self, audio: np.ndarray, sr: int, metadata: Dict[str, Any]) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Processar áudio carregado"""

        metadata.update({
            "original_sample_rate": sr,
            "original_dtype": str(audio.dtype),
            "original_shape": audio.shape
        })

        # Converter para mono se necessário
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
            metadata["converted_to_mono"] = True

        return audio, sr, metadata

    def _validate_basic_properties(self, audio: np.ndarray, sample_rate: int, metadata: Dict[str, Any]):
        """Validar propriedades básicas do áudio"""

        # 1. Verificar se não está vazio
        if len(audio) == 0:
            raise AudioCorruptedError("Array de áudio vazio")

        # 2. Verificar tipo de dados
        if str(audio.dtype) not in self.SUPPORTED_DTYPES:
            if not self.config.auto_convert:
                raise InvalidAudioDataTypeError(str(audio.dtype), self.config.target_dtype)

        # 3. Verificar taxa de amostragem
        if sample_rate != self.config.target_sample_rate:
            if self.config.strict_mode:
                raise ValidationError(sample_rate, self.config.target_sample_rate)

        # 4. Verificar duração
        duration_ms = (len(audio) / sample_rate) * 1000

        if duration_ms < self.config.min_duration_ms:
            raise AudioTooShortError(duration_ms, self.config.min_duration_ms)

        if duration_ms > self.config.max_duration_ms:
            if self.config.strict_mode:
                raise AudioTooLongError(duration_ms, self.config.max_duration_ms)

        # 5. Verificar se há valores inválidos (NaN, inf)
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            raise AudioCorruptedError("Áudio contém valores NaN ou infinitos")

        logger.debug(f"Propriedades básicas validadas: {duration_ms:.1f}ms @ {sample_rate}Hz")

    def _convert_to_target_format(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        """Converter áudio para formato alvo"""

        processed_audio = audio.copy()

        # 1. Converter tipo de dados
        processed_audio = self._convert_dtype(processed_audio)

        # 2. Reamostrar se necessário
        if original_sr != self.config.target_sample_rate:
            processed_audio = self._resample_audio(processed_audio, original_sr, self.config.target_sample_rate)

        # 3. Truncar se muito longo
        max_samples = int((self.config.max_duration_ms / 1000) * self.config.target_sample_rate)
        if len(processed_audio) > max_samples:
            processed_audio = processed_audio[:max_samples]
            logger.warning(f"Áudio truncado para {self.config.max_duration_ms}ms")

        # 4. Normalizar amplitude
        processed_audio = self._normalize_amplitude(processed_audio)

        return processed_audio

    def _convert_dtype(self, audio: np.ndarray) -> np.ndarray:
        """Converter tipo de dados"""

        target_dtype = getattr(np, self.config.target_dtype)

        if audio.dtype == target_dtype:
            return audio

        # Conversões específicas para manter qualidade
        if audio.dtype == np.int16 and target_dtype == np.float32:
            return audio.astype(np.float32) / 32768.0

        elif audio.dtype == np.int32 and target_dtype == np.float32:
            return audio.astype(np.float32) / 2147483648.0

        elif audio.dtype == np.float64 and target_dtype == np.float32:
            return audio.astype(np.float32)

        else:
            # Conversão genérica
            return audio.astype(target_dtype)

    def _resample_audio(self, audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """Reamostrar áudio"""

        if original_sr == target_sr:
            return audio

        # Usar librosa se disponível (melhor qualidade)
        if LIBROSA_AVAILABLE:
            try:
                return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
            except Exception as e:
                logger.warning(f"Librosa resample falhou: {e}")

        # Fallback para interpolação linear simples
        ratio = target_sr / original_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)

    def _normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        """Normalizar amplitude do áudio"""

        # Encontrar valor máximo absoluto
        max_val = np.abs(audio).max()

        if max_val == 0:
            return audio  # Silêncio

        # Normalizar para [-1, 1] se necessário
        if max_val > 1.0:
            return audio / max_val
        else:
            return audio

    def _validate_final_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """Validações finais no áudio processado"""

        duration_ms = (len(audio) / self.config.target_sample_rate) * 1000

        metadata = {
            "final_sample_rate": self.config.target_sample_rate,
            "final_dtype": str(audio.dtype),
            "final_shape": audio.shape,
            "duration_ms": duration_ms,
            "duration_seconds": duration_ms / 1000,
            "num_samples": len(audio),
            "amplitude_range": {
                "min": float(audio.min()),
                "max": float(audio.max()),
                "mean": float(audio.mean()),
                "std": float(audio.std())
            }
        }

        # Verificar propriedades finais
        if len(audio) == 0:
            raise AudioCorruptedError("Áudio processado está vazio")

        if duration_ms < self.config.min_duration_ms:
            raise AudioTooShortError(duration_ms, self.config.min_duration_ms)

        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            raise AudioCorruptedError("Áudio processado contém valores inválidos")

        return metadata

    def _get_conversions_applied(self, original_metadata: Dict[str, Any], final_metadata: Dict[str, Any]) -> list:
        """Obter lista de conversões aplicadas"""

        conversions = []

        # Verificar resampling
        if original_metadata.get("original_sample_rate") != final_metadata["final_sample_rate"]:
            conversions.append({
                "type": "resample",
                "from": original_metadata.get("original_sample_rate"),
                "to": final_metadata["final_sample_rate"]
            })

        # Verificar conversão de tipo
        if original_metadata.get("original_dtype") != final_metadata["final_dtype"]:
            conversions.append({
                "type": "dtype_conversion",
                "from": original_metadata.get("original_dtype"),
                "to": final_metadata["final_dtype"]
            })

        # Verificar conversão para mono
        if original_metadata.get("converted_to_mono"):
            conversions.append({
                "type": "mono_conversion",
                "from": "stereo",
                "to": "mono"
            })

        return conversions

    def quick_validate(self, audio_input: Union[str, np.ndarray, bytes]) -> bool:
        """
        Validação rápida (sem conversões)

        Args:
            audio_input: Entrada de áudio

        Returns:
            bool: True se válido
        """
        try:
            self.validate_and_convert(audio_input)
            return True
        except (ValidationError, ValidationError):
            return False

    def get_audio_info(self, audio_input: Union[str, np.ndarray, bytes]) -> Dict[str, Any]:
        """
        Obter informações detalhadas do áudio sem validação

        Args:
            audio_input: Entrada de áudio

        Returns:
            Dict com informações do áudio
        """
        try:
            input_type = self._detect_input_type(audio_input)
            audio_data, original_sr, metadata = self._load_audio(audio_input, input_type)

            duration_ms = (len(audio_data) / original_sr) * 1000

            info = {
                "input_type": input_type,
                "sample_rate": original_sr,
                "dtype": str(audio_data.dtype),
                "shape": audio_data.shape,
                "duration_ms": duration_ms,
                "duration_seconds": duration_ms / 1000,
                "num_samples": len(audio_data),
                "is_mono": audio_data.ndim == 1,
                "amplitude_range": {
                    "min": float(audio_data.min()),
                    "max": float(audio_data.max())
                },
                **metadata
            }

            return info

        except Exception as e:
            return {
                "error": str(e),
                "input_type": "unknown",
                "valid": False
            }