#!/usr/bin/env python3
"""
Proto Compiler
Automatically compiles .proto files to Python stubs

Compiles protobuf files on service startup, ensuring stubs are always up-to-date.
Handles both service-specific protos and generic fallback protos.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, List
import os

logger = logging.getLogger(__name__)


class ProtoCompiler:
    """
    Automatically compiles .proto files to Python/gRPC stubs

    Features:
    - Auto-compiles on service startup
    - Handles dependencies between protos
    - Fixes imports in generated files
    - Creates __init__.py for proper imports
    - Validates grpc-tools installation
    """

    def __init__(
        self,
        proto_dir: str = "protos",
        output_dir: str = "protos/generated"
    ):
        self.proto_dir = Path(proto_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔧 ProtoCompiler initialized")
        logger.info(f"   Proto dir: {self.proto_dir}")
        logger.info(f"   Output dir: {self.output_dir}")

    def ensure_grpc_tools_installed(self) -> bool:
        """
        Ensure grpcio-tools is installed

        Returns:
            bool: True if installed or successfully installed
        """
        try:
            import grpc_tools
            logger.debug("✅ grpcio-tools already installed")
            return True
        except ImportError:
            logger.info("📦 Installing grpcio-tools...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "grpcio-tools", "grpcio"],
                    check=True,
                    capture_output=True
                )
                logger.info("✅ grpcio-tools installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install grpcio-tools: {e}")
                return False

    def compile_proto(self, proto_file: Path) -> bool:
        """
        Compile a single .proto file

        Args:
            proto_file: Path to .proto file

        Returns:
            bool: True if compilation successful
        """
        try:
            if not proto_file.exists():
                logger.error(f"❌ Proto file not found: {proto_file}")
                return False

            logger.info(f"🔨 Compiling {proto_file.name}...")

            # Ensure grpc-tools is installed
            if not self.ensure_grpc_tools_installed():
                return False

            # Compile proto
            cmd = [
                sys.executable, "-m", "grpc_tools.protoc",
                f"-I{self.proto_dir}",
                f"--python_out={self.output_dir}",
                f"--grpc_python_out={self.output_dir}",
                str(proto_file)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info(f"✅ Compiled {proto_file.name}")

                # Fix imports in generated files
                self._fix_imports(proto_file.stem)

                return True
            else:
                logger.error(f"❌ Compilation failed for {proto_file.name}")
                logger.error(f"   stdout: {result.stdout}")
                logger.error(f"   stderr: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Compilation timeout for {proto_file.name}")
            return False
        except Exception as e:
            logger.error(f"❌ Compilation error for {proto_file.name}: {e}")
            return False

    def compile_all_protos(self) -> int:
        """
        Compile all .proto files in proto_dir

        Returns:
            int: Number of successfully compiled files
        """
        logger.info("🔨 Compiling all .proto files...")

        # Find all .proto files
        proto_files = list(self.proto_dir.glob("**/*.proto"))

        if not proto_files:
            logger.warning(f"⚠️  No .proto files found in {self.proto_dir}")
            return 0

        # Compile each file
        success_count = 0
        for proto_file in proto_files:
            if self.compile_proto(proto_file):
                success_count += 1

        # Create __init__.py files
        self._create_init_files()

        logger.info(f"✅ Compiled {success_count}/{len(proto_files)} proto files")
        return success_count

    def _fix_imports(self, proto_stem: str):
        """
        Fix imports in generated _pb2_grpc.py files

        Changes absolute imports to relative imports for proper packaging

        Args:
            proto_stem: Proto file stem (name without extension)
        """
        try:
            grpc_file = self.output_dir / f"{proto_stem}_pb2_grpc.py"
            if not grpc_file.exists():
                return

            # Read content
            content = grpc_file.read_text()

            # Fix import: "import foo_pb2" -> "from . import foo_pb2"
            # Handle both simple and aliased imports
            import_pattern1 = f"import {proto_stem}_pb2 as"
            import_pattern2 = f"import {proto_stem}_pb2\n"

            fixed_content = content

            # Fix aliased import (e.g., "import ultravox_services_pb2 as ultravox__services__pb2")
            if import_pattern1 in fixed_content:
                fixed_content = fixed_content.replace(
                    import_pattern1,
                    f"from . import {proto_stem}_pb2 as"
                )
                logger.debug(f"🔧 Fixed aliased import in {grpc_file.name}")

            # Fix simple import (e.g., "import ultravox_services_pb2")
            elif import_pattern2 in fixed_content:
                fixed_content = fixed_content.replace(
                    import_pattern2,
                    f"from . import {proto_stem}_pb2\n"
                )
                logger.debug(f"🔧 Fixed simple import in {grpc_file.name}")

            # Write back if changed
            if fixed_content != content:
                grpc_file.write_text(fixed_content)
                logger.info(f"✅ Fixed imports in {grpc_file.name}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to fix imports in {proto_stem}: {e}")

    def _create_init_files(self):
        """Create __init__.py files for proper Python packaging"""
        try:
            # Create __init__.py in proto_dir
            init_file = self.proto_dir / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                logger.debug(f"📝 Created {init_file}")

            # Create __init__.py in output_dir
            init_file = self.output_dir / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                logger.debug(f"📝 Created {init_file}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to create __init__.py: {e}")

    def compile_service_proto(self, service_name: str) -> bool:
        """
        Compile proto for a specific service

        Args:
            service_name: Service identifier (e.g., 'external_stt')

        Returns:
            bool: True if compilation successful
        """
        # Try generated proto first
        proto_file = self.output_dir / f"{service_name}.proto"
        if proto_file.exists():
            return self.compile_proto(proto_file)

        # Try main proto dir
        proto_file = self.proto_dir / f"{service_name}.proto"
        if proto_file.exists():
            return self.compile_proto(proto_file)

        logger.warning(f"⚠️  No proto file found for {service_name}")
        return False

    def is_compiled(self, proto_stem: str) -> bool:
        """
        Check if a proto file has been compiled

        Args:
            proto_stem: Proto file stem (name without .proto)

        Returns:
            bool: True if compiled stubs exist
        """
        pb2_file = self.output_dir / f"{proto_stem}_pb2.py"
        grpc_file = self.output_dir / f"{proto_stem}_pb2_grpc.py"

        return pb2_file.exists() and grpc_file.exists()

    def get_compiled_services(self) -> List[str]:
        """
        Get list of services with compiled protos

        Returns:
            List of service names (proto stems)
        """
        services = []

        for pb2_file in self.output_dir.glob("*_pb2.py"):
            stem = pb2_file.stem.replace('_pb2', '')
            if self.is_compiled(stem):
                services.append(stem)

        return services
