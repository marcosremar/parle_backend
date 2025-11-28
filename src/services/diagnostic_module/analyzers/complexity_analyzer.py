"""
Complexity Analyzer - Análise de complexidade linguística e estimativa de nível CEFR
"""

import os
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, Any, Optional
from ..models import ComplexityAnalysis, CEFRLevel
from ..llm_client import DiagnosticLLMClient
from .error_rate_analyzer import ErrorRateAnalyzer
from loguru import logger


class ComplexityAnalyzer:
    """Analisador de complexidade linguística"""
    
    def __init__(self, llm_client: DiagnosticLLMClient):
        self.llm_client = llm_client
        self.error_rate_analyzer = ErrorRateAnalyzer()
        self.acoustic_features_url = os.getenv(
            "ACOUSTIC_FEATURES_URL", 
            "http://localhost:8970"
        )
    
    async def analyze(self, text: str, user_id: Optional[str] = None) -> ComplexityAnalysis:
        """
        Analisa complexidade e estima nível CEFR
        
        Args:
            text: Texto para análise
            user_id: ID do usuário (opcional, para integração com AKT)
            
        Returns:
            Análise de complexidade
        """
        # Chamar LLM para estimativa
        result = await self.llm_client.estimate_complexity(text)
        
        return ComplexityAnalysis(
            estimated_cefr_level=CEFRLevel(result.get("cefr_level", "A1")),
            confidence=result.get("confidence", 0.5),
            vocabulary_complexity=result.get("vocabulary_complexity", "basic"),
            grammar_complexity=result.get("grammar_complexity", "basic"),
            sentence_length_avg=result.get("sentence_length_avg", 8.0),
            indicators=result.get("indicators", []),
            reasoning=result.get("reasoning", "Análise automática")
        )
    
    async def analyze_with_asr(
        self, 
        text: str, 
        asr_transcription: str,
        expected_text: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze complexity with ASR error-rate features.
        Based on Do et al. (Interspeech 2024).
        
        Args:
            text: Transcribed text for analysis
            asr_transcription: Raw ASR transcription
            expected_text: Expected/correct text (if available)
            user_id: User ID for AKT integration
            
        Returns:
            Analysis dictionary with error-rate features
        """
        # Get base analysis
        analysis = await self.analyze(text, user_id)
        
        # Convert to dict for extension
        analysis_dict = {
            "estimated_cefr_level": analysis.estimated_cefr_level.value,
            "confidence": analysis.confidence,
            "vocabulary_complexity": analysis.vocabulary_complexity,
            "grammar_complexity": analysis.grammar_complexity,
            "sentence_length_avg": analysis.sentence_length_avg,
            "indicators": analysis.indicators,
            "reasoning": analysis.reasoning
        }
        
        # Add error-rate features if expected text provided
        if expected_text:
            error_features = self.error_rate_analyzer.extract_error_rate_features(
                asr_transcription, 
                expected_text
            )
            analysis_dict["error_rate_features"] = error_features
            
            # Calculate pronunciation score
            pronunciation_score = self.error_rate_analyzer.calculate_pronunciation_score(
                error_features
            )
            analysis_dict["pronunciation_score"] = pronunciation_score
            
            # Identify specific error positions
            error_positions = self.error_rate_analyzer.identify_error_positions(
                asr_transcription,
                expected_text
            )
            analysis_dict["error_positions"] = error_positions
        
        return analysis_dict
    
    async def analyze_with_audio(
        self, 
        text: str,
        audio_path: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze complexity with acoustic features from audio.
        Based on Lee et al. (Interspeech 2024) - Multi-Embedding approach.
        
        Args:
            text: Transcribed text for analysis
            audio_path: Path to audio file
            user_id: User ID for AKT integration
            
        Returns:
            Analysis dictionary with acoustic features
        """
        # Get base text analysis
        analysis = await self.analyze(text, user_id)
        
        # Convert to dict for extension
        analysis_dict = {
            "estimated_cefr_level": analysis.estimated_cefr_level.value,
            "confidence": analysis.confidence,
            "vocabulary_complexity": analysis.vocabulary_complexity,
            "grammar_complexity": analysis.grammar_complexity,
            "sentence_length_avg": analysis.sentence_length_avg,
            "indicators": analysis.indicators,
            "reasoning": analysis.reasoning
        }
        
        # Extract acoustic features from audio service
        try:
            async with aiohttp.ClientSession() as session:
                # Check if service is available
                try:
                    async with session.get(
                        f"{self.acoustic_features_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as health_resp:
                        if health_resp.status != 200:
                            logger.warning(
                                f"Acoustic features service not available "
                                f"(status {health_resp.status})"
                            )
                            return analysis_dict
                except Exception as e:
                    logger.warning(
                        f"Acoustic features service not reachable: {e}"
                    )
                    return analysis_dict
                
                # Extract features
                with open(audio_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('audio', f, filename=os.path.basename(audio_path))
                    
                    async with session.post(
                        f"{self.acoustic_features_url}/api/acoustic/extract_features",
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            acoustic_data = await resp.json()
                            analysis_dict["acoustic_features"] = acoustic_data["features"]
                            analysis_dict["acoustic_embedding_shapes"] = {
                                "native": acoustic_data["native_embedding_shape"],
                                "learner": acoustic_data["learner_embedding_shape"],
                                "fused": acoustic_data["fused_shape"]
                            }
                            logger.info("Acoustic features extracted successfully")
                        else:
                            error_text = await resp.text()
                            logger.warning(
                                f"Failed to extract acoustic features: "
                                f"{resp.status} - {error_text}"
                            )
        
        except FileNotFoundError:
            logger.warning(f"Audio file not found: {audio_path}")
        except Exception as e:
            logger.error(f"Error extracting acoustic features: {e}")
            # Continue without acoustic features
        
        return analysis_dict
    
    async def analyze_with_audio(
        self,
        text: str,
        audio_path: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze complexity with acoustic features from audio.
        Based on Lee et al. (Interspeech 2024) - Multi-Embedding approach.
        
        Args:
            text: Transcribed text for analysis
            audio_path: Path to audio file
            user_id: User ID for AKT integration
            
        Returns:
            Analysis dictionary with acoustic features
        """
        # Get base text analysis
        analysis = await self.analyze(text, user_id)
        
        # Convert to dict for extension
        analysis_dict = {
            "estimated_cefr_level": analysis.estimated_cefr_level.value,
            "confidence": analysis.confidence,
            "vocabulary_complexity": analysis.vocabulary_complexity,
            "grammar_complexity": analysis.grammar_complexity,
            "sentence_length_avg": analysis.sentence_length_avg,
            "indicators": analysis.indicators,
            "reasoning": analysis.reasoning
        }
        
        # Extract acoustic features if service available
        try:
            async with aiohttp.ClientSession() as session:
                # Check if service is available
                try:
                    async with session.get(
                        f"{self.acoustic_features_url}/health",
                        timeout=aiohttp.ClientTimeout(total=2)
                    ) as health_resp:
                        if health_resp.status == 200:
                            # Service is available, extract features
                            with open(audio_path, 'rb') as f:
                                data = aiohttp.FormData()
                                data.add_field('audio', f, filename=Path(audio_path).name)
                                
                                async with session.post(
                                    f"{self.acoustic_features_url}/api/acoustic/extract_features",
                                    data=data,
                                    timeout=aiohttp.ClientTimeout(total=30)
                                ) as resp:
                                    if resp.status == 200:
                                        acoustic_data = await resp.json()
                                        analysis_dict["acoustic_features"] = acoustic_data.get("features", [])
                                        analysis_dict["acoustic_metadata"] = {
                                            "native_embedding_shape": acoustic_data.get("native_embedding_shape"),
                                            "learner_embedding_shape": acoustic_data.get("learner_embedding_shape"),
                                            "fused_shape": acoustic_data.get("fused_shape"),
                                            "model_info": acoustic_data.get("model_info", {})
                                        }
                                        logger.info(f"Acoustic features extracted: {len(acoustic_data.get('features', []))} dimensions")
                                    else:
                                        logger.warning(f"Acoustic features service returned {resp.status}")
                        else:
                            logger.warning("Acoustic features service not healthy")
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Acoustic features service unavailable: {e}")
        except FileNotFoundError:
            logger.warning(f"Audio file not found: {audio_path}")
        except Exception as e:
            logger.error(f"Error extracting acoustic features: {e}")
            # Continue without acoustic features
        
        return analysis_dict
    
    def estimate_level_simple(self, text: str) -> CEFRLevel:
        """
        Estimativa simples baseada em heurísticas (fallback)
        
        Args:
            text: Texto para análise
            
        Returns:
            Nível CEFR estimado
        """
        # Heurísticas simples
        words = text.split()
        avg_length = sum(len(w) for w in words) / len(words) if words else 0
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = len(words) / sentence_count if sentence_count > 0 else len(words)
        
        # Classificação básica
        if avg_sentence_length < 5 and avg_length < 4:
            return CEFRLevel.A1
        elif avg_sentence_length < 8:
            return CEFRLevel.A2
        elif avg_sentence_length < 12:
            return CEFRLevel.B1
        elif avg_sentence_length < 15:
            return CEFRLevel.B2
        else:
            return CEFRLevel.C1

