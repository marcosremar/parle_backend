"""
Systemd Service Manager
D-Bus integration for controlling external services via systemd
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class SystemdServiceState(Enum):
    """Systemd service states"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ACTIVATING = "activating"
    DEACTIVATING = "deactivating"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class SystemdServiceStatus:
    """Systemd service status information"""
    name: str
    state: SystemdServiceState
    sub_state: str
    pid: Optional[int]
    memory_bytes: Optional[int]
    cpu_usage_percent: Optional[float]
    uptime_seconds: Optional[int]
    restart_count: int
    last_start: Optional[datetime]


class SystemdServiceManager:
    """
    Manage external services via systemd instead of subprocess

    Uses D-Bus API to communicate with systemd for process management.
    Falls back to systemctl CLI if D-Bus is not available.
    """

    def __init__(self, use_dbus: bool = True):
        self.use_dbus = use_dbus and self._check_dbus_available()
        self.service_prefix = "ultravox-"

        if self.use_dbus:
            try:
                import dbus
                self.bus = dbus.SystemBus()
                self.systemd = self.bus.get_object(
                    'org.freedesktop.systemd1',
                    '/org/freedesktop/systemd1'
                )
                self.manager = dbus.Interface(
                    self.systemd,
                    'org.freedesktop.systemd1.Manager'
                )
                logger.info("✅ D-Bus connection to systemd established")
            except Exception as e:
                logger.warning(f"⚠️ D-Bus not available, falling back to CLI: {e}")
                self.use_dbus = False

        if not self.use_dbus:
            logger.info("📋 Using systemctl CLI for service management")

    def _check_dbus_available(self) -> bool:
        """Check if D-Bus Python bindings are available"""
        try:
            import dbus
            return True
        except ImportError:
            return False

    def _get_unit_name(self, service_id: str) -> str:
        """Convert service ID to systemd unit name"""
        return f"{self.service_prefix}{service_id}.service"

    def _systemctl(self, *args) -> Tuple[int, str, str]:
        """Execute systemctl command"""
        import subprocess

        cmd = ['systemctl'] + list(args)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)

    def start_service(self, service_id: str) -> Dict[str, any]:
        """Start a service via systemd"""
        unit_name = self._get_unit_name(service_id)

        try:
            if self.use_dbus:
                # Use D-Bus API
                import dbus
                self.manager.StartUnit(unit_name, 'replace')
                logger.info(f"✅ Started {unit_name} via D-Bus")
            else:
                # Use CLI
                returncode, stdout, stderr = self._systemctl('start', unit_name)
                if returncode != 0:
                    raise Exception(f"systemctl start failed: {stderr}")
                logger.info(f"✅ Started {unit_name} via systemctl")

            return {
                "success": True,
                "service": service_id,
                "unit": unit_name,
                "method": "dbus" if self.use_dbus else "cli",
                "status": "starting"
            }

        except Exception as e:
            logger.error(f"❌ Failed to start {service_id}: {e}")
            return {
                "success": False,
                "service": service_id,
                "error": str(e)
            }

    def stop_service(self, service_id: str) -> Dict[str, any]:
        """Stop a service via systemd"""
        unit_name = self._get_unit_name(service_id)

        try:
            if self.use_dbus:
                import dbus
                self.manager.StopUnit(unit_name, 'replace')
                logger.info(f"🛑 Stopped {unit_name} via D-Bus")
            else:
                returncode, stdout, stderr = self._systemctl('stop', unit_name)
                if returncode != 0:
                    raise Exception(f"systemctl stop failed: {stderr}")
                logger.info(f"🛑 Stopped {unit_name} via systemctl")

            return {
                "success": True,
                "service": service_id,
                "unit": unit_name,
                "method": "dbus" if self.use_dbus else "cli",
                "status": "stopping"
            }

        except Exception as e:
            logger.error(f"❌ Failed to stop {service_id}: {e}")
            return {
                "success": False,
                "service": service_id,
                "error": str(e)
            }

    def restart_service(self, service_id: str) -> Dict[str, any]:
        """Restart a service via systemd"""
        unit_name = self._get_unit_name(service_id)

        try:
            if self.use_dbus:
                import dbus
                self.manager.RestartUnit(unit_name, 'replace')
                logger.info(f"🔄 Restarted {unit_name} via D-Bus")
            else:
                returncode, stdout, stderr = self._systemctl('restart', unit_name)
                if returncode != 0:
                    raise Exception(f"systemctl restart failed: {stderr}")
                logger.info(f"🔄 Restarted {unit_name} via systemctl")

            return {
                "success": True,
                "service": service_id,
                "unit": unit_name,
                "method": "dbus" if self.use_dbus else "cli",
                "status": "restarting"
            }

        except Exception as e:
            logger.error(f"❌ Failed to restart {service_id}: {e}")
            return {
                "success": False,
                "service": service_id,
                "error": str(e)
            }

    def get_service_status(self, service_id: str) -> SystemdServiceStatus:
        """Get detailed service status from systemd"""
        unit_name = self._get_unit_name(service_id)

        try:
            if self.use_dbus:
                return self._get_status_dbus(unit_name)
            else:
                return self._get_status_cli(unit_name)
        except Exception as e:
            logger.error(f"❌ Failed to get status for {service_id}: {e}")
            return SystemdServiceStatus(
                name=service_id,
                state=SystemdServiceState.UNKNOWN,
                sub_state="unknown",
                pid=None,
                memory_bytes=None,
                cpu_usage_percent=None,
                uptime_seconds=None,
                restart_count=0,
                last_start=None
            )

    def _get_status_dbus(self, unit_name: str) -> SystemdServiceStatus:
        """Get status via D-Bus"""
        import dbus

        # Get unit object
        unit_path = self.manager.LoadUnit(unit_name)
        unit = self.bus.get_object('org.freedesktop.systemd1', unit_path)
        unit_interface = dbus.Interface(unit, 'org.freedesktop.DBus.Properties')

        # Get properties
        active_state = str(unit_interface.Get('org.freedesktop.systemd1.Unit', 'ActiveState'))
        sub_state = str(unit_interface.Get('org.freedesktop.systemd1.Unit', 'SubState'))

        # Get service-specific properties
        try:
            main_pid = int(unit_interface.Get('org.freedesktop.systemd1.Service', 'MainPID'))
        except Exception as e:
            logger.debug(f"Could not get MainPID from systemd: {e}")
            main_pid = None

        # Map to enum
        state_map = {
            'active': SystemdServiceState.ACTIVE,
            'inactive': SystemdServiceState.INACTIVE,
            'activating': SystemdServiceState.ACTIVATING,
            'deactivating': SystemdServiceState.DEACTIVATING,
            'failed': SystemdServiceState.FAILED,
        }
        state = state_map.get(active_state, SystemdServiceState.UNKNOWN)

        return SystemdServiceStatus(
            name=unit_name.replace('.service', '').replace(self.service_prefix, ''),
            state=state,
            sub_state=sub_state,
            pid=main_pid,
            memory_bytes=None,  # Would need cgroups API
            cpu_usage_percent=None,
            uptime_seconds=None,
            restart_count=0,
            last_start=None
        )

    def _get_status_cli(self, unit_name: str) -> SystemdServiceStatus:
        """Get status via systemctl CLI"""
        returncode, stdout, stderr = self._systemctl('show', unit_name, '--no-pager')

        if returncode != 0:
            raise Exception(f"systemctl show failed: {stderr}")

        # Parse output
        props = {}
        for line in stdout.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                props[key] = value

        # Map state
        active_state = props.get('ActiveState', 'unknown')
        state_map = {
            'active': SystemdServiceState.ACTIVE,
            'inactive': SystemdServiceState.INACTIVE,
            'activating': SystemdServiceState.ACTIVATING,
            'deactivating': SystemdServiceState.DEACTIVATING,
            'failed': SystemdServiceState.FAILED,
        }
        state = state_map.get(active_state, SystemdServiceState.UNKNOWN)

        # Get PID
        main_pid = None
        if 'MainPID' in props and props['MainPID'] != '0':
            main_pid = int(props['MainPID'])

        return SystemdServiceStatus(
            name=unit_name.replace('.service', '').replace(self.service_prefix, ''),
            state=state,
            sub_state=props.get('SubState', 'unknown'),
            pid=main_pid,
            memory_bytes=None,
            cpu_usage_percent=None,
            uptime_seconds=None,
            restart_count=int(props.get('NRestarts', 0)),
            last_start=None
        )

    def list_services(self) -> List[str]:
        """List all Ultravox services"""
        try:
            if self.use_dbus:
                import dbus
                units = self.manager.ListUnits()
                service_names = []
                for unit in units:
                    name = str(unit[0])
                    if name.startswith(self.service_prefix) and name.endswith('.service'):
                        service_id = name.replace('.service', '').replace(self.service_prefix, '')
                        service_names.append(service_id)
                return service_names
            else:
                returncode, stdout, stderr = self._systemctl(
                    'list-units', f'{self.service_prefix}*.service', '--no-legend'
                )
                if returncode != 0:
                    return []

                service_names = []
                for line in stdout.split('\n'):
                    if line.strip():
                        parts = line.split()
                        if parts:
                            unit_name = parts[0]
                            if unit_name.startswith(self.service_prefix):
                                service_id = unit_name.replace('.service', '').replace(self.service_prefix, '')
                                service_names.append(service_id)
                return service_names
        except Exception as e:
            logger.error(f"❌ Failed to list services: {e}")
            return []

    def is_service_active(self, service_id: str) -> bool:
        """Check if a service is active"""
        status = self.get_service_status(service_id)
        return status.state == SystemdServiceState.ACTIVE

    def reload_daemon(self) -> bool:
        """Reload systemd daemon configuration"""
        try:
            if self.use_dbus:
                import dbus
                self.manager.Reload()
                logger.info("🔄 Systemd daemon reloaded via D-Bus")
            else:
                returncode, stdout, stderr = self._systemctl('daemon-reload')
                if returncode != 0:
                    raise Exception(f"daemon-reload failed: {stderr}")
                logger.info("🔄 Systemd daemon reloaded via systemctl")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to reload daemon: {e}")
            return False
