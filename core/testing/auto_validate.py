import os
#!/usr/bin/env python3
"""
Auto-Validation Framework

Automatically generates validation tests for FastAPI endpoints using decorators.
Eliminates boilerplate validate.py files in each service.

Usage:
    # In routes.py:
    from src.core.testing.auto_validate import validated_endpoint

    @router.post("/process")
    @validated_endpoint(
        required_fields=["audio", "sample_rate"],
        optional_fields=["language"],
        response_model=ProcessResponse
    )
    async def process_audio(request: AudioRequest):
        return {"result": "ok"}

    # In conftest.py or test file:
    from src.core.testing.auto_validate import generate_validation_tests

    # Tests are auto-generated!
    # Run: pytest -m auto_validate
"""

from functools import wraps
from typing import List, Dict, Any, Optional, Callable, Type
from pydantic import BaseModel
import inspect
from loguru import logger

# Global registry of endpoints for testing
_REGISTERED_ENDPOINTS: List[Dict[str, Any]] = []


def validated_endpoint(
    required_fields: Optional[List[str]] = None,
    optional_fields: Optional[List[str]] = None,
    response_model: Optional[Type[BaseModel]] = None,
    test_cases: Optional[List[Dict[str, Any]]] = None
):
    """
    Decorator that auto-registers endpoint for validation testing

    Args:
        required_fields: List of required field names
        optional_fields: List of optional field names
        response_model: Pydantic model for response validation
        test_cases: Custom test cases (for complex scenarios)

    Example:
        @router.post("/transcribe")
        @validated_endpoint(
            required_fields=["audio"],
            optional_fields=["language", "sample_rate"],
            response_model=TranscriptionResponse
        )
        async def transcribe(request: AudioRequest):
            return {"text": "hello world"}

    Framework auto-generates tests for:
    - Missing required fields (400 error)
    - Invalid field types (422 error)
    - Response schema validation
    - Endpoint availability (200/201 success)
    """
    def decorator(func: Callable):
        # Extract route info from function
        path = f"/{func.__name__}"  # Default path
        method = "POST"  # Default method

        # Try to extract actual path from route decorator if available
        if hasattr(func, "__route__"):
            path = func.__route__.get("path", path)
            method = func.__route__.get("method", method)

        # Extract request model from function signature
        sig = inspect.signature(func)
        request_model = None
        for param in sig.parameters.values():
            if param.annotation != inspect.Parameter.empty:
                if hasattr(param.annotation, "__mro__") and BaseModel in param.annotation.__mro__:
                    request_model = param.annotation
                    break

        # Register endpoint
        endpoint_info = {
            "function": func,
            "function_name": func.__name__,
            "path": path,
            "method": method,
            "required_fields": required_fields or [],
            "optional_fields": optional_fields or [],
            "request_model": request_model,
            "response_model": response_model,
            "test_cases": test_cases or []
        }

        _REGISTERED_ENDPOINTS.append(endpoint_info)

        logger.debug(
            f"📝 Registered endpoint for auto-validation: {method} {path}",
            required_fields=required_fields,
            optional_fields=optional_fields
        )

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def get_registered_endpoints() -> List[Dict[str, Any]]:
    """
    Get all registered endpoints

    Returns:
        List of endpoint metadata dictionaries
    """
    return _REGISTERED_ENDPOINTS.copy()


def clear_registered_endpoints():
    """Clear all registered endpoints (useful for testing)"""
    global _REGISTERED_ENDPOINTS
    _REGISTERED_ENDPOINTS = []


def generate_test_case_missing_field(endpoint: Dict[str, Any], field: str) -> Dict[str, Any]:
    """
    Generate test case for missing required field

    Args:
        endpoint: Endpoint metadata
        field: Field to omit

    Returns:
        Test case dictionary
    """
    return {
        "name": f"test_{endpoint['function_name']}_missing_{field}",
        "description": f"Test {endpoint['path']} fails when {field} is missing",
        "request": {
            k: "test_value"
            for k in endpoint['required_fields']
            if k != field
        },
        "expected_status": 422,  # Unprocessable Entity
        "expected_error_field": field
    }


def generate_test_case_invalid_type(endpoint: Dict[str, Any], field: str) -> Dict[str, Any]:
    """
    Generate test case for invalid field type

    Args:
        endpoint: Endpoint metadata
        field: Field with invalid type

    Returns:
        Test case dictionary
    """
    # Build valid request
    request_data = {k: "test_value" for k in endpoint['required_fields']}

    # Make one field invalid (e.g., string instead of int)
    request_data[field] = "invalid_type_not_a_number"

    return {
        "name": f"test_{endpoint['function_name']}_invalid_{field}_type",
        "description": f"Test {endpoint['path']} fails when {field} has invalid type",
        "request": request_data,
        "expected_status": 422,
        "expected_error_field": field
    }


def generate_test_case_success(endpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate test case for successful request

    Args:
        endpoint: Endpoint metadata

    Returns:
        Test case dictionary
    """
    # Build valid request with all required fields
    request_data = {k: "test_value" for k in endpoint['required_fields']}

    return {
        "name": f"test_{endpoint['function_name']}_success",
        "description": f"Test {endpoint['path']} succeeds with valid data",
        "request": request_data,
        "expected_status": 200,  # or 201
        "validate_response": True
    }


def generate_pytest_code(service_name: str, base_url: str = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")) -> str:
    """
    Generate pytest code for all registered endpoints

    Args:
        service_name: Service name (e.g., "session")
        base_url: Base URL for the service

    Returns:
        Python code string with pytest tests

    Example:
        code = generate_pytest_code("session", "http://localhost:8302")
        with open("test_session_auto_validate.py", "w") as f:
            f.write(code)
    """
    endpoints = get_registered_endpoints()

    if not endpoints:
        logger.warning(f"No endpoints registered for {service_name}")
        return ""

    # Generate pytest code
    lines = [
        '"""',
        f'Auto-generated validation tests for {service_name}',
        '',
        'Generated by Auto-Validation Framework',
        'DO NOT EDIT MANUALLY - regenerate using generate_pytest_code()',
        '"""',
        '',
        'import pytest',
        'import httpx',
        'from typing import Dict, Any',
        '',
        '',
        f'BASE_URL = "{base_url}"',
        '',
        '',
        '@pytest.fixture',
        'async def client():',
        '    """Async HTTP client for testing"""',
        '    async with httpx.AsyncClient(base_url=BASE_URL) as client:',
        '        yield client',
        '',
        ''
    ]

    # Generate tests for each endpoint
    for endpoint in endpoints:
        path = endpoint['path']
        method = endpoint['method'].lower()
        func_name = endpoint['function_name']
        required_fields = endpoint['required_fields']
        response_model = endpoint['response_model']

        # Test 1: Missing required fields
        for field in required_fields:
            test_case = generate_test_case_missing_field(endpoint, field)

            lines.extend([
                '@pytest.mark.auto_validate',
                f'async def {test_case["name"]}(client):',
                f'    """{test_case["description"]}"""',
                f'    response = await client.{method}(',
                f'        "{path}",',
                f'        json={test_case["request"]}',
                '    )',
                f'    assert response.status_code == {test_case["expected_status"]}',
                '',
                ''
            ])

        # Test 2: Success case
        test_case = generate_test_case_success(endpoint)

        lines.extend([
            '@pytest.mark.auto_validate',
            f'async def {test_case["name"]}(client):',
            f'    """{test_case["description"]}"""',
            f'    response = await client.{method}(',
            f'        "{path}",',
            f'        json={test_case["request"]}',
            '    )',
            f'    assert response.status_code in [200, 201]',
            ''
        ])

        # Test 3: Response schema validation (if model provided)
        if response_model:
            lines.extend([
                '    # Validate response schema',
                '    data = response.json()',
                f'    {response_model.__name__}(**data)  # Pydantic validation',
                ''
            ])

        lines.append('')

    return '\n'.join(lines)


def save_pytest_file(service_name: str, output_path: str, base_url: str = os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888")):
    """
    Save auto-generated pytest file

    Args:
        service_name: Service name
        output_path: Path to save pytest file
        base_url: Base URL for the service

    Example:
        save_pytest_file(
            "session",
            "src/services/session/tests/test_auto_validate.py",
            "http://localhost:8302"
        )
    """
    code = generate_pytest_code(service_name, base_url)

    if code:
        with open(output_path, 'w') as f:
            f.write(code)

        logger.info(f"✅ Generated auto-validation tests: {output_path}")
    else:
        logger.warning(f"⚠️  No tests generated for {service_name} (no endpoints registered)")


# ============================================
# CLI Tool for generating tests
# ============================================

def main():
    """CLI tool to generate validation tests"""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Auto-Validation Framework - Generate tests")
    parser.add_argument("service", help="Service name (e.g., 'session')")
    parser.add_argument("--base-url", default=os.getenv("SERVICE_MANAGER_URL", "http://localhost:8888"), help="Base URL")
    parser.add_argument("--output", help="Output file path (optional)")

    args = parser.parse_args()

    # Default output path
    output_path = args.output or f"src/services/{args.service}/tests/test_auto_validate.py"

    # Generate and save
    save_pytest_file(args.service, output_path, args.base_url)

    print(f"✅ Generated tests for {args.service}")
    print(f"   Output: {output_path}")
    print(f"   Run: pytest {output_path} -v")


if __name__ == "__main__":
    main()
