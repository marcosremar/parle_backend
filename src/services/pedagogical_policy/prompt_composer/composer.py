"""
Prompt Composer - Orquestrador de camadas para composição de prompts
"""

from typing import Dict, Any
from ..models import PromptContext
from ..policy_engine import PolicyEngine
from .layers import (
    BaseLayer,
    StudentStateLayer,
    StrategyLayer,
    FocusLayer,
    AffectiveLayer,
    TurnAnalysisLayer
)


class PromptComposer:
    """
    Compositor modular de prompts pedagógicos
    
    Combina múltiplas camadas de informação para criar um prompt
    completo e adaptado ao estado do estudante.
    """
    
    def __init__(self):
        self.policy_engine = PolicyEngine()
    
    def compose(self, context: PromptContext) -> Dict[str, Any]:
        """
        Compõe um prompt pedagógico completo baseado no contexto
        
        Args:
            context: Contexto completo do prompt
            
        Returns:
            Dicionário com prompt composto e metadados
        """
        # Processar contexto através do Policy Engine
        decisions = self.policy_engine.process_context(context)
        
        # Criar camadas
        layers = [
            BaseLayer(context.scenario, cefr_level=context.cefr_level.value),
            StudentStateLayer(
                context.cefr_level,
                context.native_language,
                context.cefr_details,
                context.interpretable_knowledge_state if hasattr(context, 'interpretable_knowledge_state') else None
            ),
            StrategyLayer(
                decisions["strategy"],
                decisions["mastery_probability"],
                decisions["strategy_instructions"],
                decisions["scaffolding_type"],
                cefr_level=context.cefr_level.value
            ),
            FocusLayer(context.target_skill),
            TurnAnalysisLayer(
                context.current_turn_analysis if hasattr(context, 'current_turn_analysis') else None,
                context.session_analysis if hasattr(context, 'session_analysis') else None
            ),
            AffectiveLayer(decisions["emotional_modulation"])
        ]
        
        # Renderizar todas as camadas
        prompt_parts = [layer.render() for layer in layers]
        
        # Adicionar instrução global de validação CEFR ao final
        global_instruction = f"\n\n[INSTRUÇÃO FINAL - CRÍTICA]\n"
        global_instruction += f"Antes de responder, VERIFIQUE obrigatoriamente:\n"
        global_instruction += f"1. Suas frases estão no nível {context.cefr_level.value}?\n"
        global_instruction += f"2. Você usou apenas gramática permitida para {context.cefr_level.value}?\n"
        global_instruction += f"3. Seu vocabulário é apropriado para {context.cefr_level.value}?\n"
        global_instruction += f"4. Todas as suas explicações, exemplos e falas estão adaptadas ao nível {context.cefr_level.value}?\n"
        global_instruction += f"\nSe NÃO estiver tudo no nível {context.cefr_level.value}, você DEVE:\n"
        global_instruction += f"  - SIMPLIFIQUE se estiver muito complexo para {context.cefr_level.value}\n"
        global_instruction += f"  - COMPLEXIFIQUE se estiver muito simples para {context.cefr_level.value}\n"
        global_instruction += f"\nEsta validação se aplica a TODAS as suas respostas: tutor, cenário, contexto, explicações, exemplos e correções."
        
        full_prompt = "\n\n".join(prompt_parts) + global_instruction
        
        return {
            "prompt": full_prompt,
            "strategy": decisions["strategy"],
            "scaffolding_type": decisions["scaffolding_type"],
            "metadata": {
                "cefr_level": context.cefr_level.value,
                "mastery_probability": decisions["mastery_probability"],
                "target_skill": context.target_skill,
                "emotional_state": context.emotional_state.value
            }
        }

