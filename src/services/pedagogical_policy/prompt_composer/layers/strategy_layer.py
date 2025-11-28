"""
Strategy Layer - Estratégia pedagógica e instruções
"""

from typing import Dict, Any
from ...models import Strategy, ScaffoldingType


class StrategyLayer:
    """Camada que fornece estratégia pedagógica e instruções"""
    
    def __init__(
        self,
        strategy: Strategy,
        mastery_probability: float,
        strategy_instructions: Dict[str, Any],
        scaffolding_type: ScaffoldingType,
        cefr_level: str = "A1"
    ):
        self.strategy = strategy
        self.mastery_probability = mastery_probability
        self.strategy_instructions = strategy_instructions
        self.scaffolding_type = scaffolding_type
        self.cefr_level = cefr_level
    
    def render(self) -> str:
        """Renderiza estratégia pedagógica"""
        strategy_names = {
            Strategy.TEACH: "ENSINO EXPLÍCITO",
            Strategy.REINFORCE: "REFORÇO COM PRÁTICA",
            Strategy.CHALLENGE: "DESAFIO AVANÇADO"
        }
        
        scaffolding_names = {
            ScaffoldingType.IMPLICIT: "Correção Sutil (Recast)",
            ScaffoldingType.EXPLICIT: "Correção Direta"
        }
        
        parts = [
            "[Pedagogical Strategy]",
            f"Estratégia: {strategy_names.get(self.strategy, self.strategy.value)}",
            f"Mastery: {self.mastery_probability:.0%}",
            f"Abordagem: {self.strategy_instructions.get('approach', 'N/A')}",
            f"\n[Instrução Específica]",
            self.strategy_instructions.get('instruction', ''),
            f"\n[Estilo de Correção]",
            scaffolding_names.get(self.scaffolding_type, self.scaffolding_type.value),
            f"Velocidade: {self.strategy_instructions.get('pace', 'normal')}",
            f"Complexidade: {self.strategy_instructions.get('complexity', 'moderate')}",
            f"Feedback: {self.strategy_instructions.get('feedback', 'normal')}"
        ]
        
        # Adicionar instruções específicas de correção
        if self.scaffolding_type == ScaffoldingType.IMPLICIT:
            parts.append("\n- Se o aluno errar, reformule a frase correta naturalmente (recast).")
            parts.append("- Não pare a conversa para corrigir explicitamente.")
            parts.append("- Mantenha o fluxo conversacional.")
        else:
            parts.append("\n- Se o aluno errar, corrija explicitamente de forma clara e educada.")
            parts.append("- Explique o erro brevemente.")
            parts.append("- Dê um exemplo correto.")
        
        # Adicionar restrições linguísticas baseadas no nível CEFR
        parts.append(f"\n[Restrições Linguísticas - Nível {self.cefr_level}]")
        parts.append("IMPORTANTE: Ao aplicar esta estratégia pedagógica, use APENAS estruturas gramaticais e vocabulário apropriados ao nível do aluno.")
        parts.append(f"O aluno está no nível {self.cefr_level}, então:")
        parts.append(f"  - Use apenas gramática permitida para {self.cefr_level}")
        parts.append(f"  - Use apenas vocabulário apropriado para {self.cefr_level}")
        parts.append(f"  - Adapte suas explicações e exemplos ao nível {self.cefr_level}")
        
        if self.scaffolding_type == ScaffoldingType.IMPLICIT:
            parts.append(f"\nAo fazer recast (reformulação), certifique-se de que a frase correta está no nível {self.cefr_level}.")
        else:
            parts.append(f"\nAo corrigir explicitamente, use explicações e exemplos no nível {self.cefr_level}.")
        
        return "\n".join(parts)

