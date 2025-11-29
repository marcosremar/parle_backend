"""
Scenarios Module - Direct Python calls for scenario management
"""

from typing import Dict, Optional, Any, List

from src.modules.base_module import BaseModule


class ScenariosModule(BaseModule):
    """Scenarios Module for direct Python calls"""
    
    def __init__(self):
        super().__init__("scenarios")
        self.service = None
    
    async def _initialize(self) -> bool:
        """Initialize scenarios service"""
        try:
            from .service import ScenariosService
            self.service = ScenariosService()
            self.logger.info("✅ Scenarios Module initialized")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  Scenarios service not available: {e}")
            # Fallback to in-memory
            self.scenarios_db = {}
            return True
    
    async def create_scenario(
        self,
        name: str,
        description: str,
        system_prompt: str
    ) -> Dict[str, Any]:
        """Create a new scenario"""
        if not self.initialized:
            await self.initialize()
        
        if self.service and hasattr(self.service, 'create_scenario'):
            try:
                return await self.service.create_scenario(
                    name=name,
                    description=description,
                    system_prompt=system_prompt
                )
            except Exception as e:
                self.logger.warning(f"Service create_scenario failed: {e}, using fallback")
        
        # Fallback to in-memory
        import secrets
        from datetime import datetime
        scenario_id = f"scenario_{secrets.token_hex(8)}"
        scenario = {
            "scenario_id": scenario_id,
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "created_at": datetime.now().isoformat()
        }
        if not hasattr(self, 'scenarios_db'):
            self.scenarios_db = {}
        self.scenarios_db[scenario_id] = scenario
        return scenario
    
    async def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get scenario by ID"""
        if not self.initialized:
            await self.initialize()
        
        if self.service and hasattr(self.service, 'get_scenario'):
            try:
                return await self.service.get_scenario(scenario_id)
            except Exception as e:
                self.logger.warning(f"Service get_scenario failed: {e}, using fallback")
        
        # Fallback to in-memory
        if not hasattr(self, 'scenarios_db'):
            self.scenarios_db = {}
        return self.scenarios_db.get(scenario_id)
    
    async def list_scenarios(self) -> List[Dict[str, Any]]:
        """List all scenarios"""
        if not self.initialized:
            await self.initialize()
        
        if self.service and hasattr(self.service, 'list_scenarios'):
            try:
                return await self.service.list_scenarios()
            except Exception as e:
                self.logger.warning(f"Service list_scenarios failed: {e}, using fallback")
        
        # Fallback to in-memory
        if not hasattr(self, 'scenarios_db'):
            self.scenarios_db = {}
        return list(self.scenarios_db.values())
