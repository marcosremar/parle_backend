"""
Version Tracker - Track versions of all components in real-time
"""

from pathlib import Path
import subprocess
import json
import logging
from typing import Dict, Any, Optional
import importlib.metadata
import sys
import os

logger = logging.getLogger(__name__)


class VersionTracker:
    """
    Track versions of all components used in testing
    """

    @staticmethod
    def get_all_versions() -> Dict[str, str]:
        """Get versions of all components"""
        versions = {}

        # Get Groq API version
        versions['groq'] = VersionTracker.get_groq_version()

        # Get Whisper model version
        versions['whisper'] = VersionTracker.get_whisper_version()

        # Get Ultravox version and model
        versions['ultravox'] = VersionTracker.get_ultravox_version()
        versions['ultravox_model'] = VersionTracker.get_ultravox_model()

        # Get Kokoro TTS version
        versions['kokoro'] = VersionTracker.get_kokoro_version()

        # Get vLLM version
        versions['vllm'] = VersionTracker.get_vllm_version()

        # Get Python version
        versions['python'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Get CUDA version if available
        versions['cuda'] = VersionTracker.get_cuda_version()

        # Get GPU info
        versions['gpu'] = VersionTracker.get_gpu_info()

        # Get system info
        versions['system'] = VersionTracker.get_system_info()

        logger.info(f"📦 Component versions tracked: {json.dumps(versions, indent=2)}")
        return versions

    @staticmethod
    def get_groq_version() -> str:
        """Get Groq API client version"""
        try:
            import groq
            return groq.__version__ if hasattr(groq, '__version__') else "unknown"
        except ImportError as e:
            logger.debug(f"Groq module not available via __version__: {e}")
            try:
                version = importlib.metadata.version('groq')
                return version
            except importlib.metadata.PackageNotFoundError:
                logger.debug("Groq package not installed")
                return "not installed"

    @staticmethod
    def get_whisper_version() -> str:
        """Get Whisper model version used by Groq"""
        # Groq uses their own Whisper deployment
        return "whisper-large-v3-turbo (Groq deployment)"

    @staticmethod
    def get_ultravox_version() -> str:
        """Get Ultravox version"""
        try:
            # Check if using local or HuggingFace version
            model_path = str(Path.home() / ".cache" / "ultravox-pipeline" / "models/fixie-ai/ultravox-v0_2")
            if os.path.exists(os.path.join(model_path, "config.json")):
                with open(os.path.join(model_path, "config.json"), 'r') as f:
                    config = json.load(f)
                    return config.get('model_type', 'ultravox-v0.2')
            return "ultravox-v0.2"
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Could not read ultravox version from local config: {e}")
            return "ultravox-v0.2"

    @staticmethod
    def get_ultravox_model() -> str:
        """Get specific Ultravox model being used"""
        return "fixie-ai/ultravox-v0_2 (8B parameters)"

    @staticmethod
    def get_kokoro_version() -> str:
        """Get Kokoro TTS version"""
        try:
            import kokoro
            if hasattr(kokoro, '__version__'):
                return kokoro.__version__
            else:
                # Try to get from package metadata
                try:
                    version = importlib.metadata.version('kokoro')
                    return version
                except importlib.metadata.PackageNotFoundError:
                    logger.debug("Kokoro package metadata not found")
                    # Check if using specific model
                    return "kokoro-82M (hexgrad)"
        except ImportError as e:
            logger.debug(f"Kokoro module not installed: {e}")
            return "not installed"

    @staticmethod
    def get_vllm_version() -> str:
        """Get vLLM version"""
        try:
            import vllm
            return vllm.__version__ if hasattr(vllm, '__version__') else "unknown"
        except ImportError as e:
            logger.debug(f"vLLM module not available via __version__: {e}")
            try:
                version = importlib.metadata.version('vllm')
                return version
            except importlib.metadata.PackageNotFoundError:
                logger.debug("vLLM package not installed")
                return "not installed"

    @staticmethod
    def get_cuda_version() -> str:
        """Get CUDA version"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.version.cuda
            return "not available"
        except ImportError as e:
            logger.debug(f"PyTorch not installed, cannot get CUDA version: {e}")
            return "not installed"

    @staticmethod
    def get_gpu_info() -> str:
        """Get GPU information"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return f"{gpu_name} ({gpu_memory:.1f}GB)"
            return "no GPU"
        except (ImportError, RuntimeError) as e:
            logger.debug(f"PyTorch GPU detection failed, trying nvidia-smi fallback: {e}")
            try:
                # Fallback to nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    output = result.stdout.strip()
                    return output.replace(', ', ' - ')
                return "GPU info unavailable"
            except (FileNotFoundError, subprocess.TimeoutExpired) as fallback_err:
                logger.debug(f"nvidia-smi detection also failed: {fallback_err}")
                return "no GPU detected"

    @staticmethod
    def get_system_info() -> str:
        """Get system information"""
        try:
            import platform
            return f"{platform.system()} {platform.release()}"
        except Exception as e:
            logger.debug(f"Could not get system information: {e}")
            return "unknown"

    @staticmethod
    def format_versions_html(versions: Dict[str, str]) -> str:
        """Format versions as HTML table"""
        html = """
        <div class="version-info">
            <h3>Component Versions</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background: #f0f0f0;">
                    <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Component</th>
                    <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Version</th>
                </tr>
        """

        for component, version in versions.items():
            html += f"""
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>{component.upper()}</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd; font-family: monospace;">{version}</td>
                </tr>
            """

        html += """
            </table>
        </div>
        """
        return html