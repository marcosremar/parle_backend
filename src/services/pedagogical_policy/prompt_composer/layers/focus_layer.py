"""
Focus Layer - Habilidade em foco e objetivos de aprendizado
"""

from typing import Optional, Dict, Any


class FocusLayer:
    """Camada que fornece informações sobre a habilidade em foco"""
    
    def __init__(self, target_skill: Optional[Dict[str, Any]] = None):
        self.target_skill = target_skill or {}
    
    def render(self) -> str:
        """Renderiza informações sobre a habilidade em foco"""
        if not self.target_skill:
            return "[Focus Instruction]\nNão há habilidade específica em foco. Mantenha a conversa natural."
        
        skill_id = self.target_skill.get("skill_id", "unknown")
        skill_name = self.target_skill.get("name", skill_id)
        mastery = self.target_skill.get("mastery_probability", 0.0)
        category = self.target_skill.get("category", "unknown")
        
        parts = [
            "[Focus Instruction]",
            f"Habilidade em foco: {skill_name} ({skill_id})",
            f"Categoria: {category}",
            f"Domínio atual: {mastery:.0%}"
        ]
        
        # Adicionar objetivo baseado no nível de domínio
        if mastery < 0.3:
            parts.append(f"\nObjetivo: Introduzir e ensinar o conceito de '{skill_name}'.")
            parts.append("- Use exemplos claros e simples.")
            parts.append("- Repita o conceito de diferentes formas.")
        elif mastery < 0.7:
            parts.append(f"\nObjetivo: Reforçar e praticar '{skill_name}'.")
            parts.append("- Use o conceito naturalmente na conversa.")
            parts.append("- Dê oportunidades para o aluno praticar.")
        else:
            parts.append(f"\nObjetivo: Revisar e consolidar '{skill_name}'.")
            parts.append("- Use o conceito em contextos mais complexos.")
            parts.append("- Introduza variações e exceções.")
        
        return "\n".join(parts)

