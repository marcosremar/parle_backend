"""
Scenarios Module - Direct Python calls for Scenario management
"""

from typing import Dict, Optional, Any, List
from loguru import logger

from src.modules.base_module import BaseModule


class ScenariosModule(BaseModule):
    """Scenarios Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("scenarios")
        self.scenarios_db = {}
    
    async def _initialize(self) -> bool:
        """Initialize scenarios storage"""
        try:
            # Try to import scenarios service
            from src.services.scenarios.service import ScenariosService
            
            # For now, use in-memory storage
            # In the future, can integrate with ScenariosService
            self.scenarios_db = {}
            
            self.logger.info("✅ Scenarios Module initialized")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  Scenarios service not available: {e}")
            # Use in-memory storage
            self.scenarios_db = {}
            return True
    
    async def create_scenario(
        self,
        name: str,
        description: str,
        system_prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new scenario"""
        if not self.initialized:
            await self.initialize()
        
        try:
            import secrets
            from datetime import datetime
            
            scenario_id = f"scenario_{secrets.token_hex(8)}"
            scenario = {
                "scenario_id": scenario_id,
                "name": name,
                "description": description,
                "system_prompt": system_prompt,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                **kwargs
            }
            self.scenarios_db[scenario_id] = scenario
            return scenario
        except Exception as e:
            self.logger.error(f"❌ Scenario creation failed: {e}")
            raise
    
    async def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get scenario by ID"""
        if not self.initialized:
            await self.initialize()
        
        return self.scenarios_db.get(scenario_id)
    
    async def list_scenarios(self) -> List[Dict[str, Any]]:
        """List all scenarios"""
        if not self.initialized:
            await self.initialize()
        
        return list(self.scenarios_db.values())
    
    async def update_scenario(
        self,
        scenario_id: str,
        **updates
    ) -> Optional[Dict[str, Any]]:
        """Update scenario"""
        if not self.initialized:
            await self.initialize()
        
        if scenario_id in self.scenarios_db:
            from datetime import datetime
            self.scenarios_db[scenario_id].update(updates)
            self.scenarios_db[scenario_id]["updated_at"] = datetime.now().isoformat()
            return self.scenarios_db[scenario_id]
        return None
    
    async def delete_scenario(self, scenario_id: str) -> bool:
        """Delete scenario"""
        if not self.initialized:
            await self.initialize()
        
        if scenario_id in self.scenarios_db:
            del self.scenarios_db[scenario_id]
            return True
        return False
