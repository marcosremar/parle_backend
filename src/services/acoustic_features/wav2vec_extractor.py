"""
Wav2Vec Feature Extractor
Extracts acoustic embeddings using pre-trained Wav2Vec 2.0 models.
Based on Lee et al. (Interspeech 2024) - Multi-Embedding approach.
"""

import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from typing import Dict, Any, Optional
import os
from loguru import logger


class Wav2VecExtractor:
    """
    Extract acoustic embeddings using pre-trained Wav2Vec 2.0 models.
    
    Based on Lee et al. (Interspeech 2024):
    - Uses multiple Wav2Vec models (native + learner)
    - Extracts embeddings for multi-head attention fusion
    
    For now, uses the same base model for both.
    In future, fine-tune separate models on native vs. L2 Portuguese data.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize Wav2Vec extractor.
        
        Args:
            device: Device to use ('mps' for M1, 'cuda' for GPU, 'cpu' for CPU)
                   If None, auto-detects based on availability
        """
        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using Apple Silicon GPU (MPS)")
            elif torch.cuda.is_available():
                device = "cuda"
                logger.info("Using NVIDIA GPU (CUDA)")
            else:
                device = "cpu"
                logger.warning("No GPU available, using CPU (slow)")
        
        self.device = device
        logger.info(f"Wav2Vec Extractor initialized on device: {device}")
        
        # Model configuration
        self.model_name = "facebook/wav2vec2-large-xlsr-53"
        
        # Load processor
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            logger.info(f"Loaded processor: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise
        
        # Load models
        try:
            # Native speaker model (Portuguese)
            # For now, using same base model
            # TODO: Fine-tune on native Portuguese speech
            self.model_native = Wav2Vec2Model.from_pretrained(self.model_name)
            self.model_native = self.model_native.to(self.device)
            self.model_native.eval()
            logger.info("Loaded native Wav2Vec model")
            
            # Learner model (L2 Portuguese)
            # For now, using same base model
            # TODO: Fine-tune on L2 Portuguese speech
            self.model_learner = Wav2Vec2Model.from_pretrained(self.model_name)
            self.model_learner = self.model_learner.to(self.device)
            self.model_learner.eval()
            logger.info("Loaded learner Wav2Vec model")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_audio(self, audio_path: str, target_sr: int = 16000) -> torch.Tensor:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (Wav2Vec expects 16kHz)
            
        Returns:
            Audio tensor of shape (1, samples)
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, target_sr
                )
                waveform = resampler(waveform)
            
            # Ensure shape is (1, samples)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            
            return waveform
            
        except Exception as e:
            logger.error(f"Failed to load audio from {audio_path}: {e}")
            raise
    
    async def extract_embeddings(
        self, 
        audio_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from both native and learner models.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with:
            - native_embedding: Embedding from native model (seq_len, 768)
            - learner_embedding: Embedding from learner model (seq_len, 768)
        """
        # Load audio
        waveform = self._load_audio(audio_path)
        waveform = waveform.to(self.device)
        
        # Process audio
        try:
            audio_input = self.processor(
                waveform.squeeze().cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            audio_input = {k: v.to(self.device) for k, v in audio_input.items()}
        except Exception as e:
            logger.error(f"Failed to process audio: {e}")
            raise
        
        # Extract embeddings
        with torch.no_grad():
            try:
                # Native model
                native_output = self.model_native(audio_input.input_values)
                native_emb = native_output.last_hidden_state
                
                # Learner model
                learner_output = self.model_learner(audio_input.input_values)
                learner_emb = learner_output.last_hidden_state
                
                logger.debug(
                    f"Extracted embeddings - Native: {native_emb.shape}, "
                    f"Learner: {learner_emb.shape}"
                )
                
                return {
                    "native_embedding": native_emb,
                    "learner_embedding": learner_emb
                }
                
            except Exception as e:
                logger.error(f"Failed to extract embeddings: {e}")
                raise
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "native_model_params": sum(
                p.numel() for p in self.model_native.parameters()
            ),
            "learner_model_params": sum(
                p.numel() for p in self.model_learner.parameters()
            ),
            "native_model_size_mb": sum(
                p.numel() * 4 / (1024 * 1024)  # FP32 = 4 bytes
                for p in self.model_native.parameters()
            ),
            "learner_model_size_mb": sum(
                p.numel() * 4 / (1024 * 1024)
                for p in self.model_learner.parameters()
            )
        }
