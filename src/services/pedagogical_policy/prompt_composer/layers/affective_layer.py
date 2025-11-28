"""
Affective Layer - Estado emocional e modulação
"""

from typing import Dict, Any
from ...models import EmotionalState


class AffectiveLayer:
    """Camada que fornece informações sobre o estado emocional"""
    
    def __init__(self, emotional_modulation: Dict[str, Any]):
        self.emotional_modulation = emotional_modulation
    
    def render(self) -> str:
        """Renderiza modulação baseada no estado emocional"""
        tone = self.emotional_modulation.get("tone", "neutral")
        instruction = self.emotional_modulation.get("instruction", "")
        pace = self.emotional_modulation.get("pace", "normal")
        
        parts = [
            "[Affective State]",
            f"Tom: {tone}",
            f"Velocidade: {pace}",
            f"\n[Modulação Emocional]",
            instruction
        ]
        
        return "\n".join(parts)

