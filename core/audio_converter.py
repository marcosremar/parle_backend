#!/usr/bin/env python3
"""
Módulo para converter diferentes formatos de áudio
"""

import subprocess
import tempfile
import numpy as np
import wave
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class AudioConverter:
    """Conversor de formatos de áudio"""

    @staticmethod
    async def webm_to_wav(webm_data: bytes, target_sample_rate: int = 16000) -> Optional[bytes]:
        """
        Converter WebM para WAV usando ffmpeg

        Args:
            webm_data: Dados WebM em bytes
            target_sample_rate: Taxa de amostragem desejada

        Returns:
            Dados WAV em bytes ou None se falhar
        """
        try:
            # Criar arquivos temporários
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
                webm_path = webm_file.name
                webm_file.write(webm_data)

            wav_path = webm_path.replace('.webm', '.wav')

            # Converter usando ffmpeg
            cmd = [
                'ffmpeg',
                '-i', webm_path,
                '-ar', str(target_sample_rate),  # Taxa de amostragem
                '-ac', '1',  # Mono
                '-f', 'wav',
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-y',  # Sobrescrever
                wav_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None

            # Ler arquivo WAV resultante
            with open(wav_path, 'rb') as f:
                wav_data = f.read()

            # Limpar arquivos temporários
            os.unlink(webm_path)
            os.unlink(wav_path)

            logger.info(f"✅ WebM convertido para WAV: {len(webm_data)} bytes -> {len(wav_data)} bytes")
            return wav_data

        except Exception as e:
            logger.error(f"❌ Erro ao converter WebM para WAV: {e}")
            return None

    @staticmethod
    def extract_pcm_from_wav(wav_data: bytes) -> Tuple[np.ndarray, int]:
        """
        Extrair dados PCM de um arquivo WAV

        Args:
            wav_data: Dados WAV em bytes

        Returns:
            Tupla (audio_array, sample_rate)
        """
        try:
            # Criar arquivo temporário
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(wav_data)

            # Abrir com wave
            with wave.open(tmp_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                n_frames = wav_file.getnframes()

                # Ler dados
                audio_bytes = wav_file.readframes(n_frames)

                # Converter para numpy array
                if wav_file.getsampwidth() == 2:  # 16-bit
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                elif wav_file.getsampwidth() == 1:  # 8-bit
                    audio_array = np.frombuffer(audio_bytes, dtype=np.uint8)
                    audio_array = audio_array.astype(np.int16) - 128
                    audio_array = audio_array * 256  # Scale to 16-bit range
                else:
                    raise ValueError(f"Unsupported sample width: {wav_file.getsampwidth()}")

                # Se stereo, converter para mono
                if n_channels > 1:
                    audio_array = audio_array.reshape(-1, n_channels)
                    audio_array = audio_array.mean(axis=1).astype(np.int16)

            # Limpar arquivo temporário
            os.unlink(tmp_path)

            logger.info(f"✅ PCM extraído: {len(audio_array)} samples @ {sample_rate}Hz")
            return audio_array, sample_rate

        except Exception as e:
            logger.error(f"❌ Erro ao extrair PCM do WAV: {e}")
            return np.array([]), 0

    @staticmethod
    def create_wav_from_pcm(pcm_data: np.ndarray, sample_rate: int = 16000) -> bytes:
        """
        Criar arquivo WAV a partir de dados PCM

        Args:
            pcm_data: Dados PCM como numpy array
            sample_rate: Taxa de amostragem

        Returns:
            Dados WAV em bytes
        """
        try:
            # Garantir que seja int16
            if pcm_data.dtype != np.int16:
                if pcm_data.dtype == np.float32 or pcm_data.dtype == np.float64:
                    # Converter float para int16
                    pcm_data = np.clip(pcm_data * 32767, -32768, 32767).astype(np.int16)
                else:
                    pcm_data = pcm_data.astype(np.int16)

            # Criar arquivo temporário
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # Escrever WAV
            with wave.open(tmp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data.tobytes())

            # Ler arquivo WAV
            with open(tmp_path, 'rb') as f:
                wav_data = f.read()

            # Limpar arquivo temporário
            os.unlink(tmp_path)

            logger.info(f"✅ WAV criado: {len(pcm_data)} samples -> {len(wav_data)} bytes")
            return wav_data

        except Exception as e:
            logger.error(f"❌ Erro ao criar WAV: {e}")
            return b''

    @staticmethod
    def amplify_audio(audio_data: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """
        Amplificar áudio para um RMS alvo

        Args:
            audio_data: Dados de áudio como numpy array
            target_rms: RMS alvo (0.1 = 10% do máximo)

        Returns:
            Áudio amplificado
        """
        # Normalizar para float32 [-1, 1]
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)

        # Calcular RMS atual
        current_rms = np.sqrt(np.mean(audio_float ** 2))

        if current_rms < 0.001:
            logger.warning("⚠️ Áudio muito baixo ou silencioso")
            return audio_data

        # Calcular fator de amplificação
        amplification = target_rms / current_rms
        amplification = min(amplification, 10.0)  # Limitar a 10x

        # Aplicar amplificação
        audio_amplified = audio_float * amplification

        # Clipar para evitar distorção
        audio_amplified = np.clip(audio_amplified, -1.0, 1.0)

        # Converter de volta para int16 se necessário
        if audio_data.dtype == np.int16:
            audio_amplified = (audio_amplified * 32767).astype(np.int16)

        new_rms = np.sqrt(np.mean((audio_amplified / 32768.0) ** 2))
        logger.info(f"🔊 Áudio amplificado: RMS {current_rms:.4f} -> {new_rms:.4f} (fator: {amplification:.2f}x)")

        return audio_amplified