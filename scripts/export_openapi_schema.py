#!/usr/bin/env python3
"""
Script para exportar schema OpenAPI do FastAPI
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import app

def export_openapi_schema():
    """Export OpenAPI schema to JSON file"""
    schema = app.openapi()
    
    # Create docs directory if it doesn't exist
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # Write schema to file
    schema_file = docs_dir / "openapi.json"
    with open(schema_file, "w") as f:
        json.dump(schema, f, indent=2)
    
    print(f"✅ OpenAPI schema exported to {schema_file}")
    print(f"   Version: {schema.get('info', {}).get('version', 'unknown')}")
    print(f"   Endpoints: {len(schema.get('paths', {}))}")
    
    return schema_file

if __name__ == "__main__":
    export_openapi_schema()
