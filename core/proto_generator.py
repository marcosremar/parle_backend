#!/usr/bin/env python3
"""
Proto Generator
Automatically generates .proto files from FastAPI endpoints

Analyzes FastAPI routes and Pydantic models to create gRPC service definitions.
Auto-generates protobuf files for each service, enabling type-safe gRPC communication.
"""

import logging
import inspect
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from fastapi.routing import APIRoute
from pydantic import ValidationError, BaseModel
import re
from src.core.exceptions import UltravoxError, wrap_exception

logger = logging.getLogger(__name__)


class ProtoGenerator:
    """
    Automatically generates .proto files from FastAPI services

    Features:
    - Analyzes FastAPI routes and Pydantic models
    - Generates typed .proto messages from Pydantic schemas
    - Creates service definitions with RPCs
    - Handles nested models and common types
    - Auto-detects message types (Request/Response patterns)
    """

    def __init__(self, output_dir: str = "protos/generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Type mapping: Python -> Protobuf
        self.type_map = {
            'str': 'string',
            'int': 'int32',
            'float': 'float',
            'bool': 'bool',
            'bytes': 'bytes',
            'List[str]': 'repeated string',
            'List[int]': 'repeated int32',
            'List[float]': 'repeated float',
            'Optional[str]': 'string',
            'Optional[int]': 'int32',
            'Optional[float]': 'float',
            'Optional[bool]': 'bool',
        }

        logger.info(f"🏗️  ProtoGenerator initialized (output: {self.output_dir})")

    def generate_proto_for_service(
        self,
        service_name: str,
        service_instance,
        package_name: str = "ultravox"
    ) -> Optional[Path]:
        """
        Generate .proto file for a service

        Args:
            service_name: Service identifier (e.g., 'external_stt')
            service_instance: BaseService instance
            package_name: Proto package name

        Returns:
            Path to generated .proto file or None if failed
        """
        try:
            logger.info(f"🔨 Generating .proto for {service_name}...")

            router = service_instance.get_router()

            # Analyze routes and collect models
            messages = {}
            rpcs = []
            processed_models = set()

            for route in router.routes:
                if isinstance(route, APIRoute):
                    # Extract method info
                    path = route.path
                    method_name = self._path_to_method_name(path)

                    # Get request/response types from handler signature
                    handler = route.endpoint
                    sig = inspect.signature(handler)

                    # Extract request type (first parameter)
                    request_type = None
                    request_msg_name = None
                    params = list(sig.parameters.values())
                    if params:
                        first_param = params[0]
                        if hasattr(first_param.annotation, '__mro__') and BaseModel in first_param.annotation.__mro__:
                            request_type = first_param.annotation
                            request_msg_name = request_type.__name__

                    # Extract response type from return annotation
                    response_type = None
                    response_msg_name = None
                    if sig.return_annotation and sig.return_annotation != inspect.Signature.empty:
                        return_anno = sig.return_annotation
                        # Handle Dict, Optional, etc.
                        if hasattr(return_anno, '__origin__'):
                            # Generic type like Dict[str, Any]
                            response_msg_name = f"{method_name}Response"
                        elif hasattr(return_anno, '__mro__') and BaseModel in return_anno.__mro__:
                            response_type = return_anno
                            response_msg_name = response_type.__name__

                    # Generate messages for request/response
                    if request_type and request_msg_name not in processed_models:
                        messages[request_msg_name] = self._generate_message(request_type)
                        processed_models.add(request_msg_name)

                    if response_type and response_msg_name not in processed_models:
                        messages[response_msg_name] = self._generate_message(response_type)
                        processed_models.add(response_msg_name)

                    # Create RPC definition
                    if request_msg_name and response_msg_name:
                        rpcs.append({
                            'name': method_name,
                            'request': request_msg_name,
                            'response': response_msg_name
                        })

            # Generate .proto file
            if rpcs:
                proto_content = self._generate_proto_content(
                    service_name,
                    package_name,
                    messages,
                    rpcs
                )

                proto_file = self.output_dir / f"{service_name}.proto"
                proto_file.write_text(proto_content)

                logger.info(f"✅ Generated {proto_file}")
                logger.info(f"   Messages: {len(messages)}, RPCs: {len(rpcs)}")
                return proto_file
            else:
                logger.warning(f"⚠️  No RPCs found for {service_name} - skipping proto generation")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to generate proto for {service_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _path_to_method_name(self, path: str) -> str:
        """
        Convert FastAPI path to gRPC method name

        Examples:
            /transcribe -> Transcribe
            /health -> Health
            /api/generate -> Generate
        """
        # Remove leading slash and split
        parts = path.strip('/').split('/')
        # Take last part (most specific)
        name = parts[-1]
        # Convert to PascalCase
        return ''.join(word.capitalize() for word in name.split('_'))

    def _generate_message(self, model_class: type) -> List[Dict[str, Any]]:
        """
        Generate proto message fields from Pydantic model

        Args:
            model_class: Pydantic model class

        Returns:
            List of field definitions
        """
        fields = []
        field_number = 1

        # Get Pydantic fields
        if hasattr(model_class, 'model_fields'):
            # Pydantic v2
            model_fields = model_class.model_fields
        elif hasattr(model_class, '__fields__'):
            # Pydantic v1
            model_fields = model_class.__fields__
        else:
            return fields

        for field_name, field_info in model_fields.items():
            # Get field type
            if hasattr(field_info, 'annotation'):
                field_type = field_info.annotation
            else:
                field_type = field_info.outer_type_ if hasattr(field_info, 'outer_type_') else str

            # Convert to proto type
            proto_type = self._python_type_to_proto(field_type)

            fields.append({
                'name': field_name,
                'type': proto_type,
                'number': field_number
            })
            field_number += 1

        return fields

    def _python_type_to_proto(self, python_type) -> str:
        """
        Convert Python type annotation to protobuf type

        Args:
            python_type: Python type annotation

        Returns:
            Protobuf type string
        """
        # Handle string representation
        type_str = str(python_type).replace('typing.', '').replace('<class \'', '').replace('\'>', '')

        # Check direct mapping
        if type_str in self.type_map:
            return self.type_map[type_str]

        # Handle Optional[T]
        if 'Optional[' in type_str:
            inner = type_str.replace('Optional[', '').replace(']', '')
            return self._python_type_to_proto(inner)

        # Handle List[T]
        if 'List[' in type_str or 'list[' in type_str:
            inner = re.search(r'[Ll]ist\[(.*?)\]', type_str)
            if inner:
                inner_type = self._python_type_to_proto(inner.group(1))
                return f"repeated {inner_type}"

        # Handle Dict
        if 'Dict' in type_str or 'dict' in type_str:
            return "map<string, string>"  # Simplified

        # Handle bytes
        if python_type == bytes or 'bytes' in type_str:
            return 'bytes'

        # Default fallbacks
        if 'str' in type_str.lower():
            return 'string'
        if 'int' in type_str.lower():
            return 'int32'
        if 'float' in type_str.lower():
            return 'float'
        if 'bool' in type_str.lower():
            return 'bool'

        # Unknown type - use string
        logger.warning(f"⚠️  Unknown type {type_str}, using string")
        return 'string'

    def _generate_proto_content(
        self,
        service_name: str,
        package_name: str,
        messages: Dict[str, List[Dict[str, Any]]],
        rpcs: List[Dict[str, str]]
    ) -> str:
        """
        Generate complete .proto file content

        Args:
            service_name: Service name
            package_name: Proto package name
            messages: Message definitions
            rpcs: RPC definitions

        Returns:
            Proto file content as string
        """
        lines = [
            'syntax = "proto3";',
            '',
            f'package {package_name}.{service_name};',
            '',
            '// Auto-generated by ProtoGenerator',
            '// DO NOT EDIT MANUALLY - will be regenerated on service startup',
            '',
        ]

        # Generate messages
        for msg_name, fields in messages.items():
            lines.append(f'message {msg_name} {{')
            for field in fields:
                lines.append(f'  {field["type"]} {field["name"]} = {field["number"]};')
            lines.append('}')
            lines.append('')

        # Generate service
        service_class_name = ''.join(word.capitalize() for word in service_name.split('_'))
        lines.append(f'service {service_class_name}Service {{')
        for rpc in rpcs:
            lines.append(f'  rpc {rpc["name"]}({rpc["request"]}) returns ({rpc["response"]});')
        lines.append('}')
        lines.append('')

        return '\n'.join(lines)
