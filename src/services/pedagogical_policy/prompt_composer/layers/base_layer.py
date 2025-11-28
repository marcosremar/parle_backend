"""
Base Layer - Contexto base do cenário
"""

from typing import Dict, Any, Optional


class BaseLayer:
    """Camada base que fornece contexto do cenário"""
    
    def __init__(self, scenario: Optional[Dict[str, Any]] = None, cefr_level: str = "A1"):
        self.scenario = scenario or {}
        self.cefr_level = cefr_level
    
    def render(self) -> str:
        """Renderiza a camada base"""
        if not self.scenario:
            parts = [
                "[System Role]",
                "Você é um professor paciente e atencioso de português."
            ]
        else:
            scenario_name = self.scenario.get("name", "conversação")
            system_prompt = self.scenario.get("system_prompt", "")
            ai_role = self.scenario.get("ai_role", "professor")
            
            parts = [
                "[System Role]",
                f"Você é um {ai_role} em um cenário de {scenario_name}."
            ]
            
            if system_prompt:
                parts.append(f"\n[Contexto do Cenário]\n{system_prompt}")
        
        # Adicionar instrução de adaptação ao nível CEFR
        parts.append(f"\n[Adaptação ao Nível {self.cefr_level}]")
        parts.append(f"IMPORTANTE: Adapte TODAS as suas respostas ao nível {self.cefr_level} do aluno.")
        parts.append("Isso inclui:")
        parts.append("  - Descrição do cenário (se necessário)")
        parts.append("  - Instruções e explicações")
        parts.append("  - Todas as falas do seu personagem")
        parts.append("  - Vocabulário e estruturas gramaticais")
        parts.append(f"\nSe o aluno está no nível {self.cefr_level}, você DEVE usar apenas linguagem apropriada para esse nível.")
        
        return "\n".join(parts)

