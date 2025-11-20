"""
Viber Gateway - Standalone runner
For development and testing
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from .service import ViberGatewayService
from .config import get_config


def main():
    """Run Viber Gateway Service"""
    config = get_config()

    # Create service instance
    service = ViberGatewayService(config=config)

    # Run with uvicorn
    uvicorn.run(
        service.app,
        host=config['service']['host'],
        port=config['service']['port'],
        log_level="info"
    )


if __name__ == "__main__":
    main()
