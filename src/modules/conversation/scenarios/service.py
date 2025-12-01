"""
Scenarios Service - Simple in-memory scenarios storage
"""

from datetime import datetime
import secrets
from typing import Any


class ScenariosService:
    """Simple scenarios service with in-memory storage"""

    def __init__(self):
        self.scenarios = {}

    async def create_scenario(
        self, name: str, description: str, system_prompt: str
    ) -> dict[str, Any]:
        """Create a new scenario"""
        scenario_id = f"scenario_{secrets.token_hex(8)}"
        scenario = {
            "scenario_id": scenario_id,
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "created_at": datetime.now().isoformat(),
        }
        self.scenarios[scenario_id] = scenario
        return scenario

    async def get_scenario(self, scenario_id: str) -> dict[str, Any] | None:
        """Get scenario by ID"""
        return self.scenarios.get(scenario_id)

    async def list_scenarios(self) -> list[dict[str, Any]]:
        """List all scenarios"""
        return list(self.scenarios.values())
