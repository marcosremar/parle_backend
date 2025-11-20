#!/usr/bin/env python3
"""
Fixture data loader for User Service
Automatically loads test users during installation

Note: User service uses in-memory storage, so fixtures are loaded
into the users_db dictionary and optionally saved to a JSON file
for persistence across service restarts.
"""

import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.user.storage import users_db


# Test Users
FIXTURE_USERS = [
    {
        "id": "test-user",
        "email": "test@ultravox.ai",
        "name": "Test User",
        "language": "pt-BR",
        "metadata": {
            "role": "test",
            "created_by": "fixtures",
            "description": "Default test user for integration tests"
        }
    },
    {
        "id": "demo-user",
        "email": "demo@ultravox.ai",
        "name": "Demo User",
        "language": "pt-BR",
        "metadata": {
            "role": "demo",
            "created_by": "fixtures",
            "description": "Demo user for demonstrations"
        }
    }
]


def load_fixtures() -> Dict[str, Any]:
    """
    Load fixture users into in-memory storage

    Returns:
        Dict with statistics about loaded fixtures
    """
    stats = {
        "total": len(FIXTURE_USERS),
        "created": 0,
        "skipped": 0,
        "errors": 0,
        "users": {}
    }

    for user_data in FIXTURE_USERS:
        try:
            user_id = user_data["id"]

            # Check if already exists
            if user_id in users_db:
                stats["skipped"] += 1
                stats["users"][user_id] = {
                    "status": "skipped",
                    "email": users_db[user_id]["email"]
                }
                continue

            # Create user directly in storage
            users_db[user_id] = {
                "id": user_id,
                "email": user_data["email"],
                "name": user_data["name"],
                "language": user_data.get("language", "pt-BR"),
                "metadata": user_data.get("metadata", {}),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            stats["created"] += 1
            stats["users"][user_id] = {
                "status": "created",
                "email": user_data["email"]
            }

        except Exception as e:
            stats["errors"] += 1
            stats["users"][user_data["id"]] = {
                "status": "error",
                "error": str(e)
            }

    return stats


def main():
    """Main entry point for fixture loading"""
    print("\n" + "="*80)
    print("USER SERVICE - FIXTURE LOADER")
    print("="*80 + "\n")

    print("📊 Loading fixture users into in-memory storage...")
    stats = load_fixtures()

    print(f"\n✅ Fixture loading complete!")
    print(f"   Total: {stats['total']}")
    print(f"   Created: {stats['created']}")
    print(f"   Skipped: {stats['skipped']}")
    print(f"   Errors: {stats['errors']}\n")

    if stats['created'] > 0:
        print("👤 Created users:")
        for user_id, info in stats['users'].items():
            if info['status'] == 'created':
                print(f"   ✅ {user_id}: {info['email']}")

    if stats['errors'] > 0:
        print("\n❌ Errors:")
        for user_id, info in stats['users'].items():
            if info['status'] == 'error':
                print(f"   ❌ {user_id}: {info['error']}")
        return 1

    # Save to persistent file (for persistence across restarts)
    try:
        fixture_file = Path(__file__).parent / "data" / "fixture_users.json"
        fixture_file.parent.mkdir(parents=True, exist_ok=True)

        with open(fixture_file, 'w') as f:
            json.dump(dict(users_db), f, indent=2)

        print(f"\n💾 Fixture users saved to: {fixture_file}")
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to save fixtures to file: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
