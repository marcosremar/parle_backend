"""
GPU Profile Manager
Manages different GPU allocation profiles for various use cases
"""

import os
import yaml
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ProfileType(Enum):
    """Available GPU profiles"""
    FULL_GPU = "full_gpu"
    HYBRID = "hybrid"
    MINIMAL_GPU = "minimal_gpu"


@dataclass
class ServiceConfig:
    """Service configuration within a profile"""
    enabled: bool
    device: str = "cuda"
    gpu_required: bool = False
    gpu_memory_mb: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    gpu_memory_utilization_min: Optional[float] = None
    vllm_gpu_memory_utilization: Optional[float] = None  # vLLM-specific GPU allocation
    startup_priority: int = 999
    auto_start: bool = True
    cpu_threads: Optional[int] = None
    provider: Optional[str] = None  # For external services
    model: Optional[str] = None
    api_timeout_seconds: Optional[int] = None


@dataclass
class WarmupConfig:
    """Warmup configuration for a service"""
    iterations: int
    test_audio_durations: Optional[List[int]] = None
    batch_sizes: Optional[List[int]] = None
    sequence_lengths: Optional[List[int]] = None
    text_lengths: Optional[List[int]] = None
    languages: Optional[List[str]] = None
    voices: Optional[List[str]] = None
    with_audio: bool = False


@dataclass
class ProfileWarmupConfig:
    """Complete warmup configuration for a profile"""
    enabled: bool
    parallel: bool
    timeout_seconds: int
    stt: Optional[WarmupConfig] = None
    llm: Optional[WarmupConfig] = None
    tts: Optional[WarmupConfig] = None
    external_stt: Optional[WarmupConfig] = None


@dataclass
class PerformanceTargets:
    """Expected performance targets for a profile"""
    stt_latency_ms: int
    llm_first_token_ms: int
    llm_tokens_per_second: int
    tts_latency_ms: int
    total_pipeline_ms: int


@dataclass
class ResourceLimits:
    """Resource limits for a profile"""
    total_gpu_memory_mb: int
    gpu_memory_buffer_mb: int
    estimated_cpu_memory_mb: int


@dataclass
class GPUProfile:
    """Complete GPU profile configuration"""
    name: str
    description: str
    use_case: str
    services: Dict[str, ServiceConfig]
    warmup: ProfileWarmupConfig
    performance_targets: PerformanceTargets
    resources: ResourceLimits


class GPUProfileManager:
    """
    Manages GPU profiles for the service manager

    Features:
    - Load and validate profiles from YAML
    - Switch between profiles at runtime
    - Apply profile configurations to services
    - Validate profile compatibility with hardware
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize GPU Profile Manager

        Args:
            config_path: Path to gpu_profiles.yaml
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "gpu_profiles.yaml"

        self.config_path = Path(config_path)
        self.profiles: Dict[str, GPUProfile] = {}
        self.active_profile: Optional[str] = None
        self.config: Dict[str, Any] = {}

        self._load_config()
        logger.info(f"📋 GPU Profile Manager initialized with {len(self.profiles)} profiles")

    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                logger.error(f"❌ GPU profiles config not found: {self.config_path}")
                raise FileNotFoundError(f"GPU profiles config not found: {self.config_path}")

            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            logger.info(f"📋 Loaded GPU profiles config from {self.config_path}")

            # Load profiles
            self._load_profiles()

            # Set active profile
            self.active_profile = self.config.get('active_profile', 'minimal_gpu')
            logger.info(f"🎯 Active profile: {self.active_profile}")

        except Exception as e:
            logger.error(f"❌ Failed to load GPU profiles config: {e}")
            raise

    def _load_profiles(self):
        """Load all profiles from config"""
        profiles_config = self.config.get('profiles', {})

        for profile_id, profile_data in profiles_config.items():
            try:
                # Parse service configurations
                services = {}
                for service_id, service_data in profile_data.get('services', {}).items():
                    services[service_id] = ServiceConfig(
                        enabled=service_data.get('enabled', True),
                        device=service_data.get('device', 'cuda'),
                        gpu_required=service_data.get('gpu_required', False),
                        gpu_memory_mb=service_data.get('gpu_memory_mb'),
                        gpu_memory_utilization=service_data.get('gpu_memory_utilization'),
                        gpu_memory_utilization_min=service_data.get('gpu_memory_utilization_min'),
                        vllm_gpu_memory_utilization=service_data.get('vllm_gpu_memory_utilization'),
                        startup_priority=service_data.get('startup_priority', 999),
                        auto_start=service_data.get('auto_start', True),
                        cpu_threads=service_data.get('cpu_threads'),
                        provider=service_data.get('provider'),
                        model=service_data.get('model'),
                        api_timeout_seconds=service_data.get('api_timeout_seconds')
                    )

                # Parse warmup configuration
                warmup_data = profile_data.get('warmup', {})
                warmup = ProfileWarmupConfig(
                    enabled=warmup_data.get('enabled', True),
                    parallel=warmup_data.get('parallel', False),
                    timeout_seconds=warmup_data.get('timeout_seconds', 300)
                )

                # Parse individual service warmup configs
                for service_name in ['stt', 'llm', 'tts', 'external_stt']:
                    service_warmup_data = warmup_data.get(service_name, {})
                    if service_warmup_data:
                        service_warmup = WarmupConfig(
                            iterations=service_warmup_data.get('iterations', 0),
                            test_audio_durations=service_warmup_data.get('test_audio_durations'),
                            batch_sizes=service_warmup_data.get('batch_sizes'),
                            sequence_lengths=service_warmup_data.get('sequence_lengths'),
                            text_lengths=service_warmup_data.get('text_lengths'),
                            languages=service_warmup_data.get('languages'),
                            voices=service_warmup_data.get('voices'),
                            with_audio=service_warmup_data.get('with_audio', False)
                        )
                        setattr(warmup, service_name, service_warmup)

                # Parse performance targets
                perf_data = profile_data.get('performance_targets', {})
                performance_targets = PerformanceTargets(
                    stt_latency_ms=perf_data.get('stt_latency_ms', 500),
                    llm_first_token_ms=perf_data.get('llm_first_token_ms', 100),
                    llm_tokens_per_second=perf_data.get('llm_tokens_per_second', 300),
                    tts_latency_ms=perf_data.get('tts_latency_ms', 150),
                    total_pipeline_ms=perf_data.get('total_pipeline_ms', 800)
                )

                # Parse resource limits
                res_data = profile_data.get('resources', {})
                resources = ResourceLimits(
                    total_gpu_memory_mb=res_data.get('total_gpu_memory_mb', 20000),
                    gpu_memory_buffer_mb=res_data.get('gpu_memory_buffer_mb', 3500),
                    estimated_cpu_memory_mb=res_data.get('estimated_cpu_memory_mb', 8000)
                )

                # Create profile
                profile = GPUProfile(
                    name=profile_data.get('name', profile_id),
                    description=profile_data.get('description', ''),
                    use_case=profile_data.get('use_case', ''),
                    services=services,
                    warmup=warmup,
                    performance_targets=performance_targets,
                    resources=resources
                )

                self.profiles[profile_id] = profile
                logger.info(f"  ✓ Loaded profile: {profile_id} - {profile.name}")

            except Exception as e:
                logger.error(f"❌ Failed to load profile {profile_id}: {e}")
                raise

    def get_profile(self, profile_id: str) -> Optional[GPUProfile]:
        """Get a profile by ID"""
        return self.profiles.get(profile_id)

    def get_active_profile(self) -> Optional[GPUProfile]:
        """Get the currently active profile"""
        if self.active_profile:
            return self.profiles.get(self.active_profile)
        return None

    def list_profiles(self) -> List[Dict[str, str]]:
        """List all available profiles"""
        return [
            {
                'id': profile_id,
                'name': profile.name,
                'description': profile.description,
                'use_case': profile.use_case,
                'active': profile_id == self.active_profile
            }
            for profile_id, profile in self.profiles.items()
        ]

    def validate_profile(self, profile_id: str) -> tuple[bool, Optional[str]]:
        """
        Validate if a profile can be activated

        Returns:
            (is_valid, error_message)
        """
        profile = self.get_profile(profile_id)
        if not profile:
            return False, f"Profile '{profile_id}' not found"

        # Check GPU availability
        try:
            import torch
            if not torch.cuda.is_available():
                # Check if profile requires GPU
                for service_id, service_config in profile.services.items():
                    if service_config.gpu_required:
                        return False, f"Profile '{profile_id}' requires GPU but CUDA is not available"

            # Check GPU memory
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                required_memory = profile.resources.total_gpu_memory_mb

                if total_memory < required_memory:
                    return False, f"Insufficient GPU memory: {total_memory}MB < {required_memory}MB required"

        except Exception as e:
            logger.warning(f"Could not validate GPU requirements: {e}")

        return True, None

    def activate_profile(self, profile_id: str, backup: bool = True) -> bool:
        """
        Activate a profile

        Args:
            profile_id: Profile to activate
            backup: Whether to backup current configuration

        Returns:
            Success status
        """
        # Validate profile
        is_valid, error_msg = self.validate_profile(profile_id)
        if not is_valid:
            logger.error(f"❌ Cannot activate profile '{profile_id}': {error_msg}")
            return False

        # Backup current config if requested
        if backup:
            self._backup_config()

        # Update active profile
        old_profile = self.active_profile
        self.active_profile = profile_id

        # Update config file
        try:
            self.config['active_profile'] = profile_id
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"✅ Activated profile: {profile_id} (was: {old_profile})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save active profile: {e}")
            self.active_profile = old_profile  # Rollback
            return False

    def _backup_config(self):
        """Backup current configuration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.config_path.parent / f"gpu_profiles.{timestamp}.yaml.bak"

            shutil.copy2(self.config_path, backup_path)
            logger.info(f"📦 Backed up config to: {backup_path}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to backup config: {e}")

    def get_service_config(self, profile_id: str, service_id: str) -> Optional[ServiceConfig]:
        """Get service configuration for a specific profile"""
        profile = self.get_profile(profile_id)
        if profile:
            return profile.services.get(service_id)
        return None

    def get_warmup_config(self, profile_id: Optional[str] = None) -> Optional[ProfileWarmupConfig]:
        """Get warmup configuration for a profile (defaults to active)"""
        if profile_id is None:
            profile_id = self.active_profile

        profile = self.get_profile(profile_id)
        if profile:
            return profile.warmup
        return None

    def export_profile_to_service_execution_config(self, profile_id: str) -> Dict[str, Any]:
        """
        Export profile to service_execution.yaml format

        Returns:
            Dictionary that can be merged into service_execution.yaml
        """
        profile = self.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Profile '{profile_id}' not found")

        service_config = {}

        for service_id, service_data in profile.services.items():
            config = {
                'execution_mode': 'external',  # All GPU services are external
                'locked': service_data.gpu_required,
                'auto_start': service_data.auto_start
            }

            if service_data.gpu_required:
                config.update({
                    'gpu_required': True,
                    'gpu_memory_mb': service_data.gpu_memory_mb,
                    'gpu_memory_utilization': service_data.gpu_memory_utilization,
                    'gpu_memory_utilization_min': service_data.gpu_memory_utilization_min,
                    'startup_priority': service_data.startup_priority
                })

            service_config[service_id] = config

        return {'services': service_config}

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization testing configuration"""
        return self.config.get('optimization', {})

    def get_profile_summary(self, profile_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive summary of a profile

        Args:
            profile_id: Profile ID (defaults to active)

        Returns:
            Summary dictionary
        """
        if profile_id is None:
            profile_id = self.active_profile

        profile = self.get_profile(profile_id)
        if not profile:
            return {}

        # Count enabled services
        enabled_services = [sid for sid, svc in profile.services.items() if svc.enabled]
        gpu_services = [sid for sid, svc in profile.services.items() if svc.gpu_required and svc.enabled]
        cpu_services = [sid for sid, svc in profile.services.items() if not svc.gpu_required and svc.enabled]

        return {
            'profile_id': profile_id,
            'name': profile.name,
            'description': profile.description,
            'use_case': profile.use_case,
            'active': profile_id == self.active_profile,
            'services': {
                'total': len(enabled_services),
                'gpu': len(gpu_services),
                'cpu': len(cpu_services),
                'enabled': enabled_services,
                'gpu_services': gpu_services,
                'cpu_services': cpu_services
            },
            'resources': {
                'total_gpu_memory_mb': profile.resources.total_gpu_memory_mb,
                'gpu_memory_buffer_mb': profile.resources.gpu_memory_buffer_mb,
                'estimated_cpu_memory_mb': profile.resources.estimated_cpu_memory_mb
            },
            'performance_targets': {
                'stt_latency_ms': profile.performance_targets.stt_latency_ms,
                'llm_first_token_ms': profile.performance_targets.llm_first_token_ms,
                'llm_tokens_per_second': profile.performance_targets.llm_tokens_per_second,
                'tts_latency_ms': profile.performance_targets.tts_latency_ms,
                'total_pipeline_ms': profile.performance_targets.total_pipeline_ms
            },
            'warmup': {
                'enabled': profile.warmup.enabled,
                'parallel': profile.warmup.parallel,
                'timeout_seconds': profile.warmup.timeout_seconds
            }
        }

    def create_temp_profile_variant(
        self,
        base_profile_id: str,
        vllm_utilization: float,
        variant_suffix: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a temporary profile variant with specific vLLM GPU utilization

        Args:
            base_profile_id: Base profile to clone
            vllm_utilization: vLLM GPU memory utilization (0.0-1.0)
            variant_suffix: Optional suffix for profile ID (defaults to GPU%)

        Returns:
            New profile ID or None if creation failed
        """
        base_profile = self.get_profile(base_profile_id)
        if not base_profile:
            logger.error(f"Base profile '{base_profile_id}' not found")
            return None

        # Generate variant ID
        if variant_suffix is None:
            variant_suffix = f"{int(vllm_utilization * 100)}"
        variant_id = f"{base_profile_id}_variant_{variant_suffix}"

        # Clone services config
        services = {}
        for service_id, service_config in base_profile.services.items():
            # Create copy of service config
            new_config = ServiceConfig(
                enabled=service_config.enabled,
                device=service_config.device,
                gpu_required=service_config.gpu_required,
                gpu_memory_mb=service_config.gpu_memory_mb,
                gpu_memory_utilization=service_config.gpu_memory_utilization,
                gpu_memory_utilization_min=service_config.gpu_memory_utilization_min,
                vllm_gpu_memory_utilization=service_config.vllm_gpu_memory_utilization,
                startup_priority=service_config.startup_priority,
                auto_start=service_config.auto_start,
                cpu_threads=service_config.cpu_threads,
                provider=service_config.provider,
                model=service_config.model,
                api_timeout_seconds=service_config.api_timeout_seconds
            )

            # Override vLLM GPU utilization for LLM service
            if service_id == 'llm' and new_config.gpu_required:
                new_config.vllm_gpu_memory_utilization = vllm_utilization
                new_config.gpu_memory_utilization = vllm_utilization

            services[service_id] = new_config

        # Create variant profile
        variant_profile = GPUProfile(
            name=f"{base_profile.name} (Variant {int(vllm_utilization * 100)}%)",
            description=f"Temporary variant with {vllm_utilization:.2f} vLLM GPU utilization",
            use_case=f"Benchmarking variant of {base_profile_id}",
            services=services,
            warmup=base_profile.warmup,
            performance_targets=base_profile.performance_targets,
            resources=base_profile.resources
        )

        # Register variant (in-memory only, not persisted to YAML)
        self.profiles[variant_id] = variant_profile
        logger.info(f"✅ Created temp profile variant: {variant_id} (vLLM GPU: {vllm_utilization:.2f})")

        return variant_id

    def can_restart_safely(self, service_id: str, new_gpu_util: float) -> tuple[bool, Optional[str]]:
        """
        Check if service can be restarted safely with new GPU utilization

        Args:
            service_id: Service to check (e.g., 'llm')
            new_gpu_util: Desired GPU utilization (0.0-1.0)

        Returns:
            (can_restart, error_message)
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return False, "CUDA not available"

            # Get GPU memory info
            gpu_props = torch.cuda.get_device_properties(0)
            total_mb = gpu_props.total_memory // (1024 * 1024)
            torch.cuda.empty_cache()
            free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)

            # Calculate required memory
            required_mb = int(total_mb * new_gpu_util)

            # Add safety buffer (500MB)
            safety_buffer_mb = 500
            needed_mb = required_mb + safety_buffer_mb

            if free_mb < needed_mb:
                return False, f"Insufficient GPU memory: {free_mb}MB free < {needed_mb}MB needed ({required_mb}MB + {safety_buffer_mb}MB buffer)"

            logger.info(f"✅ Safe to restart {service_id} with {new_gpu_util:.2f} GPU utilization")
            logger.info(f"   Free: {free_mb}MB, Required: {required_mb}MB, Buffer: {safety_buffer_mb}MB")

            return True, None

        except Exception as e:
            logger.error(f"Error checking restart safety: {e}")
            return False, str(e)

    def estimate_gpu_memory_needed(self, profile_id: str) -> Optional[int]:
        """
        Estimate GPU memory needed for a profile

        Args:
            profile_id: Profile to estimate

        Returns:
            Estimated memory in MB or None if profile not found
        """
        profile = self.get_profile(profile_id)
        if not profile:
            return None

        # Sum up GPU memory for all GPU services
        total_mb = 0
        for service_id, service_config in profile.services.items():
            if service_config.gpu_required and service_config.enabled:
                if service_config.gpu_memory_mb:
                    total_mb += service_config.gpu_memory_mb

        return total_mb if total_mb > 0 else profile.resources.total_gpu_memory_mb

    def get_or_create_speed_profile(self, gpu_percent: int) -> Optional[str]:
        """
        Get or create a speed profile for specific GPU percentage

        Args:
            gpu_percent: GPU utilization percentage (70-90)

        Returns:
            Profile ID or None if invalid
        """
        if not (70 <= gpu_percent <= 90):
            logger.error(f"Invalid GPU percentage: {gpu_percent} (must be 70-90)")
            return None

        profile_id = f"speed_{gpu_percent}"

        # Return if profile already exists
        if profile_id in self.profiles:
            return profile_id

        # Create dynamic profile if not in 5% increments
        if gpu_percent % 5 != 0:
            # Find closest base profile
            base_percent = (gpu_percent // 5) * 5
            base_profile_id = f"speed_{base_percent}"

            if base_profile_id not in self.profiles:
                logger.error(f"Base profile '{base_profile_id}' not found")
                return None

            # Create variant
            variant_id = self.create_temp_profile_variant(
                base_profile_id=base_profile_id,
                vllm_utilization=gpu_percent / 100.0,
                variant_suffix=str(gpu_percent)
            )

            return variant_id

        return profile_id


# Singleton instance
_profile_manager: Optional[GPUProfileManager] = None


def get_gpu_profile_manager(reload: bool = False) -> GPUProfileManager:
    """
    Get global GPU profile manager instance

    Args:
        reload: Force reload configuration

    Returns:
        GPUProfileManager instance
    """
    global _profile_manager

    if _profile_manager is None or reload:
        _profile_manager = GPUProfileManager()

    return _profile_manager
