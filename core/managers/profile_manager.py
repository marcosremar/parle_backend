"""
Profile Manager
Manages service execution profiles for different environments
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation message severity"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationMessage:
    """Validation message"""
    severity: ValidationSeverity
    message: str
    service_id: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of profile validation"""
    valid: bool
    messages: List[ValidationMessage] = field(default_factory=list)

    def add_error(self, message: str, service_id: Optional[str] = None) -> None:
        """Add error message"""
        self.valid = False
        self.messages.append(ValidationMessage(ValidationSeverity.ERROR, message, service_id))

    def add_warning(self, message: str, service_id: Optional[str] = None) -> None:
        """Add warning message"""
        self.messages.append(ValidationMessage(ValidationSeverity.WARNING, message, service_id))

    def add_info(self, message: str, service_id: Optional[str] = None) -> None:
        """Add info message"""
        self.messages.append(ValidationMessage(ValidationSeverity.INFO, message, service_id))


@dataclass
class ProfileRestrictions:
    """Profile restrictions"""
    max_gpu_memory_mb: int = 24000
    allow_remote_services: bool = True
    allow_gpu_services: bool = True
    require_gpu_services_remote: bool = False
    require_all_internal: bool = False
    allow_composite: bool = True
    max_child_modules: int = 10
    max_concurrent_services: int = 50


@dataclass
class Profile:
    """Service execution profile"""
    name: str
    description: str
    enabled_services: List[str]
    service_overrides: Dict[str, Dict[str, Any]]
    composite_services: Dict[str, Dict[str, Any]]
    restrictions: ProfileRestrictions


class ProfileManager:
    """
    Manages service execution profiles

    Features:
    - Load profiles from YAML
    - Apply profile to service configuration
    - Validate profile restrictions
    - Switch profiles at runtime
    """

    def __init__(self, profiles_path: Optional[str] = None) -> None:
        """
        Initialize ProfileManager

        Args:
            profiles_path: Path to profiles.yaml (default: config/profiles.yaml)
        """
        if profiles_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            profiles_path = project_root / "config" / "profiles.yaml"

        self.profiles_path = Path(profiles_path)
        self.profiles: Dict[str, Profile] = {}
        self.active_profile: Optional[Profile] = None
        self.config: Dict[str, Any] = {}

        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load profiles from YAML"""
        try:
            if not self.profiles_path.exists():
                logger.warning(f"⚠️  Profiles config not found: {self.profiles_path}")
                logger.info("   Using default profile")
                self._create_default_profile()
                return

            with open(self.profiles_path, 'r') as f:
                self.config = yaml.safe_load(f)

            logger.info(f"📋 Loaded profiles config from {self.profiles_path}")

            # Load all profiles
            profiles_data = self.config.get('profiles', {})
            for profile_name, profile_data in profiles_data.items():
                try:
                    self.profiles[profile_name] = self._parse_profile(profile_name, profile_data)
                except Exception as e:
                    logger.error(f"❌ Failed to load profile {profile_name}: {e}")

            # Load active profile
            active_profile_name = self.config.get('active_profile', 'development')
            if active_profile_name in self.profiles:
                self.active_profile = self.profiles[active_profile_name]
                logger.info(f"✅ Active profile: {active_profile_name}")
            else:
                logger.warning(f"⚠️  Active profile {active_profile_name} not found")
                if self.profiles:
                    self.active_profile = list(self.profiles.values())[0]
                    logger.info(f"   Using first available profile: {self.active_profile.name}")

        except Exception as e:
            logger.error(f"❌ Failed to load profiles: {e}")
            self._create_default_profile()

    def _create_default_profile(self) -> None:
        """Create default profile"""
        self.profiles['default'] = Profile(
            name='default',
            description='Default profile - all services enabled',
            enabled_services=[],
            service_overrides={},
            composite_services={},
            restrictions=ProfileRestrictions()
        )
        self.active_profile = self.profiles['default']

    def _parse_profile(self, name: str, data: Dict[str, Any]) -> Profile:
        """Parse profile from YAML data"""
        # Parse restrictions
        restrictions_data = data.get('restrictions', {})
        restrictions = ProfileRestrictions(
            max_gpu_memory_mb=restrictions_data.get('max_gpu_memory_mb', 24000),
            allow_remote_services=restrictions_data.get('allow_remote_services', True),
            allow_gpu_services=restrictions_data.get('allow_gpu_services', True),
            require_gpu_services_remote=restrictions_data.get('require_gpu_services_remote', False),
            require_all_internal=restrictions_data.get('require_all_internal', False),
            allow_composite=restrictions_data.get('allow_composite', True),
            max_child_modules=restrictions_data.get('max_child_modules', 10),
            max_concurrent_services=restrictions_data.get('max_concurrent_services', 50)
        )

        return Profile(
            name=name,
            description=data.get('description', ''),
            enabled_services=data.get('enabled_services', []),
            service_overrides=data.get('service_overrides', {}),
            composite_services=data.get('composite_services', {}),
            restrictions=restrictions
        )

    def get_profile(self, name: str) -> Optional[Profile]:
        """Get profile by name"""
        return self.profiles.get(name)

    def list_profiles(self) -> List[str]:
        """List all available profile names"""
        return list(self.profiles.keys())

    def get_active_profile(self) -> Optional[Profile]:
        """Get currently active profile"""
        return self.active_profile

    def set_active_profile(self, name: str, persist: bool = False) -> bool:
        """
        Set active profile

        Args:
            name: Profile name
            persist: Save to YAML file

        Returns:
            bool: True if successful
        """
        if name not in self.profiles:
            logger.error(f"❌ Profile not found: {name}")
            return False

        self.active_profile = self.profiles[name]
        logger.info(f"✅ Active profile set to: {name}")

        # Persist to YAML if requested
        if persist:
            try:
                self.config['active_profile'] = name
                with open(self.profiles_path, 'w') as f:
                    yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)
                logger.info(f"💾 Active profile persisted to {self.profiles_path}")
            except Exception as e:
                logger.error(f"❌ Failed to persist active profile: {e}")
                return False

        return True

    def get_enabled_services(self, profile_name: Optional[str] = None) -> List[str]:
        """
        Get list of enabled services for a profile

        Args:
            profile_name: Profile name (uses active if None)

        Returns:
            List of enabled service IDs
        """
        profile = self.profiles.get(profile_name) if profile_name else self.active_profile
        if not profile:
            return []
        return profile.enabled_services

    def get_service_config(self, service_id: str, profile_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get service configuration with profile overrides applied

        Args:
            service_id: Service ID
            profile_name: Profile name (uses active if None)

        Returns:
            Service configuration dict or None
        """
        profile = self.profiles.get(profile_name) if profile_name else self.active_profile
        if not profile:
            return None

        return profile.service_overrides.get(service_id)

    def is_service_enabled(self, service_id: str, profile_name: Optional[str] = None) -> bool:
        """
        Check if service is enabled in profile

        Args:
            service_id: Service ID
            profile_name: Profile name (uses active if None)

        Returns:
            bool: True if enabled
        """
        profile = self.profiles.get(profile_name) if profile_name else self.active_profile
        if not profile:
            return True  # Default to enabled if no profile

        return service_id in profile.enabled_services

    def validate_profile(self, profile_name: str) -> ValidationResult:
        """
        Validate a profile

        Args:
            profile_name: Profile name

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        profile = self.profiles.get(profile_name)
        if not profile:
            result.add_error(f"Profile not found: {profile_name}")
            return result

        # Validate restrictions
        self._validate_restrictions(profile, result)

        # Validate service configurations
        self._validate_services(profile, result)

        # Validate composite services
        self._validate_composite_services(profile, result)

        return result

    def _validate_restrictions(self, profile: Profile, result: ValidationResult) -> None:
        """Validate profile restrictions"""
        r = profile.restrictions

        # GPU memory validation
        if r.max_gpu_memory_mb < 0:
            result.add_error("max_gpu_memory_mb cannot be negative")

        # Logical validation
        if r.require_gpu_services_remote and not r.allow_remote_services:
            result.add_error("require_gpu_services_remote=true requires allow_remote_services=true")

        if r.require_all_internal and r.allow_remote_services:
            result.add_error("require_all_internal=true conflicts with allow_remote_services=true")

        if not r.allow_gpu_services and r.max_gpu_memory_mb > 0:
            result.add_warning("allow_gpu_services=false but max_gpu_memory_mb > 0")

    def _validate_services(self, profile: Profile, result: ValidationResult) -> None:
        """Validate service configurations"""
        # Check for services in overrides but not in enabled_services
        for service_id in profile.service_overrides.keys():
            if service_id not in profile.enabled_services:
                result.add_warning(
                    f"Service {service_id} has overrides but is not in enabled_services",
                    service_id
                )

    def _validate_composite_services(self, profile: Profile, result: ValidationResult) -> None:
        """Validate composite services"""
        if not profile.restrictions.allow_composite and profile.composite_services:
            result.add_error("Composite services defined but allow_composite=false")

        for comp_id, comp_config in profile.composite_services.items():
            # Validate child_modules count
            child_modules = comp_config.get('child_modules', [])
            if len(child_modules) > profile.restrictions.max_child_modules:
                result.add_error(
                    f"Composite service {comp_id} has {len(child_modules)} child_modules "
                    f"(max: {profile.restrictions.max_child_modules})",
                    comp_id
                )

            # Check if child modules are enabled
            for child_id in child_modules:
                if child_id not in profile.enabled_services:
                    result.add_warning(
                        f"Child module {child_id} not in enabled_services",
                        comp_id
                    )

    def get_restrictions(self, profile_name: Optional[str] = None) -> Optional[ProfileRestrictions]:
        """Get profile restrictions"""
        profile = self.profiles.get(profile_name) if profile_name else self.active_profile
        if not profile:
            return None
        return profile.restrictions

    def check_restriction(self, restriction_name: str, value: Any = None, profile_name: Optional[str] = None) -> bool:
        """
        Check if a restriction allows an action

        Args:
            restriction_name: Restriction name (e.g., 'allow_gpu_services')
            value: Optional value to check against (e.g., GPU memory amount)
            profile_name: Profile name (uses active if None)

        Returns:
            bool: True if allowed
        """
        restrictions = self.get_restrictions(profile_name)
        if not restrictions:
            return True  # No restrictions

        if not hasattr(restrictions, restriction_name):
            logger.warning(f"⚠️  Unknown restriction: {restriction_name}")
            return True

        restriction_value = getattr(restrictions, restriction_name)

        # Boolean restrictions
        if isinstance(restriction_value, bool):
            return restriction_value

        # Numeric restrictions (max values)
        if isinstance(restriction_value, int) and value is not None:
            return value <= restriction_value

        return True

    def _get_config_value(self, service_config: Any, key: str, default: Any = None) -> Any:
        """
        Safely get value from service_config (works with both dict and object)

        Args:
            service_config: Service configuration (dict or object)
            key: Key/attribute name
            default: Default value if not found

        Returns:
            Value or default
        """
        if isinstance(service_config, dict):
            value = service_config.get(key, default)
        else:
            value = getattr(service_config, key, default)

        # Handle None values - return default instead
        return value if value is not None else default

    def can_start_service(self, service_id: str, service_config: Dict[str, Any], profile_name: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """
        Check if a service can be started based on profile restrictions

        Args:
            service_id: Service ID
            service_config: Service configuration dict
            profile_name: Profile name (uses active if None)

        Returns:
            tuple: (can_start: bool, reason: Optional[str])
        """
        profile = self.profiles.get(profile_name) if profile_name else self.active_profile
        if not profile:
            return (True, None)  # No profile, allow everything

        restrictions = profile.restrictions

        # Check if service is enabled in profile
        if service_id not in profile.enabled_services:
            return (False, f"Service '{service_id}' is not enabled in profile '{profile.name}'")

        # Check GPU service restrictions
        gpu_required = self._get_config_value(service_config, 'gpu_required', False)
        gpu_memory_mb = self._get_config_value(service_config, 'gpu_memory_mb', 0)
        is_gpu_service = gpu_required or gpu_memory_mb > 0
        if is_gpu_service and not restrictions.allow_gpu_services:
            return (False, f"GPU services not allowed in profile '{profile.name}' (allow_gpu_services=false)")

        # Check remote service restrictions
        execution_mode = self._get_config_value(service_config, 'execution_mode', None)
        runpod_pod_id = self._get_config_value(service_config, 'runpod_pod_id', None)
        is_remote = execution_mode == 'external' and runpod_pod_id
        if is_remote and not restrictions.allow_remote_services:
            return (False, f"Remote services not allowed in profile '{profile.name}' (allow_remote_services=false)")

        # Check if GPU service must be remote
        if is_gpu_service and restrictions.require_gpu_services_remote and not is_remote:
            return (False, f"GPU services must be remote in profile '{profile.name}' (require_gpu_services_remote=true)")

        # Check GPU memory limit
        if gpu_memory_mb > restrictions.max_gpu_memory_mb:
            return (False, f"Service requires {gpu_memory_mb}MB GPU but profile allows max {restrictions.max_gpu_memory_mb}MB")

        # Check composite service restrictions
        is_composite = self._get_config_value(service_config, 'composite', False)
        if is_composite and not restrictions.allow_composite:
            return (False, f"Composite services not allowed in profile '{profile.name}' (allow_composite=false)")

        # Check child modules count for composite
        if is_composite:
            child_modules = self._get_config_value(service_config, 'child_modules', [])
            if len(child_modules) > restrictions.max_child_modules:
                return (False, f"Composite service has {len(child_modules)} child_modules but profile allows max {restrictions.max_child_modules}")

        return (True, None)

    def calculate_total_gpu_memory(self, services_config: Dict[str, Dict[str, Any]], profile_name: Optional[str] = None) -> int:
        """
        Calculate total GPU memory required by enabled services

        Args:
            services_config: Dict of service_id -> service_config
            profile_name: Profile name (uses active if None)

        Returns:
            int: Total GPU memory in MB
        """
        profile = self.profiles.get(profile_name) if profile_name else self.active_profile
        if not profile:
            return 0

        total_mb = 0
        for service_id in profile.enabled_services:
            if service_id in services_config:
                service_config = services_config[service_id]
                gpu_mb = service_config.get('gpu_memory_mb', 0)
                if gpu_mb:
                    total_mb += gpu_mb
                    logger.debug(f"   {service_id}: {gpu_mb}MB")

        return total_mb

    def validate_gpu_memory_total(self, services_config: Dict[str, Dict[str, Any]], profile_name: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """
        Validate that total GPU memory of enabled services doesn't exceed profile limit

        Args:
            services_config: Dict of service_id -> service_config
            profile_name: Profile name (uses active if None)

        Returns:
            tuple: (is_valid: bool, error_message: Optional[str])
        """
        profile = self.profiles.get(profile_name) if profile_name else self.active_profile
        if not profile:
            return (True, None)

        total_mb = self.calculate_total_gpu_memory(services_config, profile_name)
        max_mb = profile.restrictions.max_gpu_memory_mb

        if total_mb > max_mb:
            return (False, f"Total GPU memory required ({total_mb}MB) exceeds profile limit ({max_mb}MB)")

        return (True, None)

    def validate_service_dependencies(
        self,
        service_dependencies: Dict[str, List[str]],
        profile_name: Optional[str] = None
    ) -> tuple[bool, List[str]]:
        """
        Validate that service dependencies are satisfied in profile

        Args:
            service_dependencies: Dict of service_id -> list of required services
            profile_name: Profile name (uses active if None)

        Returns:
            tuple: (all_satisfied: bool, missing_dependencies: List[str])
        """
        profile = self.profiles.get(profile_name) if profile_name else self.active_profile
        if not profile:
            return (True, [])  # No profile, skip validation

        enabled = set(profile.enabled_services)
        missing = []

        for service_id in profile.enabled_services:
            if service_id in service_dependencies:
                required_deps = service_dependencies[service_id]
                for dep in required_deps:
                    if dep not in enabled:
                        missing.append(f"{service_id} requires {dep} (not enabled in profile)")
                        logger.warning(f"⚠️  {service_id} requires {dep} but it's not enabled in profile {profile.name}")

        return (len(missing) == 0, missing)


# Singleton instance
_profile_manager_instance: Optional[ProfileManager] = None


def get_profile_manager(reload: bool = False) -> ProfileManager:
    """
    Get global ProfileManager instance (singleton)

    Args:
        reload: Force reload from disk

    Returns:
        ProfileManager instance
    """
    global _profile_manager_instance

    if _profile_manager_instance is None or reload:
        _profile_manager_instance = ProfileManager()

    return _profile_manager_instance
