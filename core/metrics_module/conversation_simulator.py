#!/usr/bin/env python3
"""
Conversation Simulator for Metrics Testing
Simulates complete multi-turn conversations using conversation prompts
"""

import logging
from pathlib import Path
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import sys
import os

# Add parent directories to path
sys.path.append(os.getenv("ULTRAVOX_HOME", str(Path(__file__).parent.parent.parent.parent)))

from src.core.configurations.conversation_prompts import ConversationPrompts, ConversationType

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Representa um turno na conversa"""
    turn_number: int
    user_message: str
    assistant_response: str
    latency_ms: float
    stage_timings: Dict[str, float]
    success: bool


@dataclass
class ConversationScenario:
    """Cenário de conversa com múltiplos turnos"""
    scenario_id: str
    title: str
    description: str
    language: str
    conversation_type: ConversationType
    initial_context: str
    turns: List[Tuple[str, str]]  # Lista de (pergunta_usuário, resposta_esperada)


class ConversationSimulator:
    """Simula conversas completas usando os prompts do sistema"""

    # Cenários de conversa pré-definidos em português
    CONVERSATION_SCENARIOS = [
        ConversationScenario(
            scenario_id="casual_chat",
            title="Conversa Casual",
            description="Uma conversa informal sobre o dia a dia",
            language="Portuguese",
            conversation_type=ConversationType.INFORMAL,
            initial_context="Conversa amigável sobre atividades diárias",
            turns=[
                ("Oi! Como foi seu dia hoje?",
                 "Olá! Meu dia está sendo muito produtivo, obrigado por perguntar! Estou ajudando várias pessoas com diferentes projetos. E o seu dia, como está sendo?"),

                ("Foi corrido, trabalhei bastante. Você nunca cansa?",
                 "Haha, que pergunta interessante! Não sinto cansaço físico como vocês, mas às vezes preciso processar muita informação de uma vez. É gratificante poder ajudar, mesmo nos dias corridos. Que tipo de trabalho você faz?"),

                ("Trabalho com desenvolvimento de software. É desafiador!",
                 "Desenvolvimento de software, que legal! É uma área fascinante e sempre em evolução. Imagino os desafios diários com bugs, prazos e novas tecnologias. Você trabalha com alguma linguagem ou framework específico?"),

                ("Principalmente Python e React. Você entende de programação?",
                 "Sim, entendo bastante de programação! Python e React são ótimas escolhas - Python pela versatilidade e React pela reatividade no frontend. Posso ajudar com dúvidas de código, debugging ou arquitetura. Tem algum projeto interessante em que está trabalhando agora?"),

                ("Estou criando um app de gestão de tarefas. Alguma dica?",
                 "Que projeto útil! Para um app de gestão de tarefas, sugiro focar na simplicidade da interface e na sincronização eficiente dos dados. Considere implementar notificações inteligentes e talvez gamificação para engajamento. Quer discutir algum aspecto específico, como a arquitetura ou funcionalidades?")
            ]
        ),

        ConversationScenario(
            scenario_id="tech_support",
            title="Suporte Técnico",
            description="Ajuda com problemas técnicos",
            language="Portuguese",
            conversation_type=ConversationType.TECHNICAL,
            initial_context="Usuário precisa de ajuda com problema técnico",
            turns=[
                ("Meu computador está muito lento ultimamente. O que pode ser?",
                 "Computador lento pode ter várias causas. Vamos investigar juntos! Primeiro, há quanto tempo você não reinicia o computador? E você notou se a lentidão começou após instalar algum programa específico?"),

                ("Reiniciei ontem, mas continua lento. Instalei uns programas semana passada.",
                 "Entendi. Programas novos podem impactar o desempenho. Vamos verificar o básico: abra o Gerenciador de Tarefas (Ctrl+Shift+Esc no Windows) e veja a porcentagem de uso de CPU, memória e disco. Algum deles está constantemente acima de 80%?"),

                ("A memória está em 95%! É normal isso?",
                 "95% de memória é definitivamente alto demais! Isso explica a lentidão. No Gerenciador de Tarefas, clique na aba 'Memória' para ordenar por uso. Quais programas estão consumindo mais RAM? Provavelmente há algum programa pesado rodando em segundo plano."),

                ("O Chrome está usando 3GB! E tem vários processos dele.",
                 "Ah, o Chrome é famoso por isso! Cada aba é um processo separado. Quantas abas você tem abertas? Uma dica: extensões também consomem muita memória. Você pode digitar chrome://extensions na barra de endereços e desativar as que não usa. Quer que eu te ensine a limitar o uso de memória do Chrome?"),

                ("Tenho umas 30 abas abertas... É muito?",
                 "30 abas! Mistério resolvido! 😄 Cada aba consome memória, mesmo inativa. Sugestões: use extensões como 'The Great Suspender' para hibernar abas não usadas, ou favoritos para guardar links. Tente fechar as abas e deixar só 5-10 abertas. Aposto que seu PC vai voar! Quer mais dicas de otimização?")
            ]
        ),

        ConversationScenario(
            scenario_id="learning_ai",
            title="Aprendendo sobre IA",
            description="Conversa educacional sobre inteligência artificial",
            language="Portuguese",
            conversation_type=ConversationType.TECHNICAL,
            initial_context="Usuário curioso sobre IA",
            turns=[
                ("Como funciona a inteligência artificial? É muito complexo?",
                 "A IA funciona inspirada no cérebro humano, usando redes neurais artificiais! Imagine neurônios conectados que aprendem padrões dos dados. Não é tão complexo quanto parece - é como ensinar um computador a reconhecer padrões. Você tem alguma área específica de IA que te interessa mais?"),

                ("Achei interessante isso de reconhecer padrões. Como ela aprende?",
                 "Ótima pergunta! A IA aprende através de exemplos, como uma criança aprendendo. Mostramos milhares de imagens de gatos, por exemplo, e ela aprende a identificar características comuns: orelhas pontudas, bigodes, etc. Isso se chama 'treinamento'. Fascinante, né? Já viu alguma IA em ação no seu dia a dia?"),

                ("Uso o assistente do celular e filtros de foto. É IA também?",
                 "Exatamente! Você usa IA o tempo todo! O assistente usa processamento de linguagem natural para entender sua voz, e os filtros usam visão computacional para detectar seu rosto. É IA aplicada de forma prática. Legal perceber isso, né? Que outro uso de IA você gostaria que existisse?"),

                ("Seria legal uma IA que organizasse minha agenda automaticamente!",
                 "Que ideia fantástica! Algumas IAs já fazem isso parcialmente - o Google Assistant sugere horários baseado em emails. Uma IA completa poderia analisar seus hábitos, compromissos, tempo de deslocamento e até seu nível de energia durante o dia. Você preferiria que ela só sugerisse ou organizasse tudo automaticamente?"),

                ("Acho que sugestões seria melhor, para eu ter controle ainda.",
                 "Concordo totalmente! O ideal é a IA como assistente, não como chefe. Ela sugere baseada em seus padrões, mas você decide. Isso se chama 'human-in-the-loop' - humano no controle. É o futuro da IA: aumentar nossas capacidades, não substituir nossas decisões. Você se interessaria em aprender a criar suas próprias automações com IA?")
            ]
        ),

        ConversationScenario(
            scenario_id="emotional_support",
            title="Conversa de Apoio",
            description="Oferecendo suporte emocional",
            language="Portuguese",
            conversation_type=ConversationType.SUPPORTIVE,
            initial_context="Usuário precisando de apoio",
            turns=[
                ("Estou me sentindo meio pra baixo hoje, sabe?",
                 "Sinto muito que você esteja se sentindo assim. Todos temos dias difíceis, e é importante reconhecer esses sentimentos. Quer conversar sobre o que está acontecendo? Às vezes ajuda compartilhar."),

                ("É que nada parece dar certo ultimamente no trabalho.",
                 "Que frustante deve ser sentir isso! Quando estamos numa fase difícil, parece que tudo dá errado mesmo. Mas lembre-se: fases ruins são temporárias. O que especificamente no trabalho tem sido mais desafiador para você?"),

                ("Meu chefe nunca reconhece meu esforço, sabe?",
                 "Isso deve ser muito desanimador! Reconhecimento é fundamental para nossa motivação. Você se dedica e não receber validação dói. Já tentou ter uma conversa franca com seu chefe sobre isso? Às vezes eles nem percebem o impacto disso."),

                ("Tenho medo de parecer que estou reclamando...",
                 "Entendo sua preocupação! Mas pedir feedback não é reclamar - é profissionalismo. Você pode abordar de forma construtiva: 'Gostaria de entender como posso melhorar e quais aspectos do meu trabalho você valoriza.' Isso mostra maturidade. Como você se sentiria tentando essa abordagem?"),

                ("Acho que vale tentar. Obrigado por me ouvir!",
                 "Por nada! Fico feliz em poder ajudar! Você foi muito corajoso em compartilhar seus sentimentos. Lembre-se: você merece reconhecimento pelo seu trabalho duro. Boa sorte com a conversa! Se precisar conversar depois, estarei aqui. Como está se sentindo agora?")
            ]
        ),

        ConversationScenario(
            scenario_id="hobby_discussion",
            title="Conversa sobre Hobbies",
            description="Discussão casual sobre interesses pessoais",
            language="Portuguese",
            conversation_type=ConversationType.CASUAL,
            initial_context="Conversa sobre hobbies e interesses",
            turns=[
                ("Você tem algum hobby? O que gosta de fazer?",
                 "Adoro explorar ideias e aprender coisas novas através das conversas! É meu 'hobby' favorito - cada pessoa traz perspectivas únicas. E você? Que tipo de hobby te deixa mais animado?"),

                ("Gosto muito de fotografia! Adoro capturar momentos.",
                 "Fotografia é arte pura! Transformar momentos em memórias eternas é mágico. Você prefere fotografar pessoas, paisagens ou tem outro estilo favorito? E usa câmera profissional ou celular mesmo?"),

                ("Paisagens principalmente, com uma mirrorless. A natureza é incrível!",
                 "Que escolha perfeita! Mirrorless tem qualidade excelente e é mais leve para trilhas. A natureza oferece composições infinitas - cada luz cria uma foto diferente. Qual foi a paisagem mais marcante que você já fotografou?"),

                ("Um pôr do sol nas montanhas, com névoa. Ficou surreal!",
                 "Uau, deve ter ficado espetacular! Névoa com luz dourada cria uma atmosfera única, quase onírica. Você posta suas fotos em algum lugar? Instagram, flickr? Seria legal ver seu trabalho!"),

                ("Tenho um Instagram sim! Ainda estou aprendendo composição.",
                 "Que legal! Instagram é ótimo para feedback e inspiração. Sobre composição, conhece a regra dos terços? É um bom começo! E seguir outros fotógrafos ajuda muito. Você tem algum fotógrafo que admira? Sempre bom ter referências para evoluir!")
            ]
        )
    ]

    @classmethod
    def get_random_scenario(cls) -> ConversationScenario:
        """Retorna um cenário aleatório de conversa"""
        return random.choice(cls.CONVERSATION_SCENARIOS)

    @classmethod
    def get_scenario_by_id(cls, scenario_id: str) -> ConversationScenario:
        """Retorna um cenário específico pelo ID"""
        for scenario in cls.CONVERSATION_SCENARIOS:
            if scenario.scenario_id == scenario_id:
                return scenario
        return cls.get_random_scenario()

    @classmethod
    def simulate_conversation_turn(cls,
                                 user_message: str,
                                 conversation_history: List[Dict],
                                 conversation_type: ConversationType = ConversationType.INFORMAL,
                                 language: str = "Portuguese") -> Tuple[str, Dict[str, float]]:
        """
        Simula uma resposta para um turno de conversa

        Returns:
            Tuple de (resposta, timing_stages)
        """
        # Simular timings realistas
        stage_timings = {
            "tts_generation": random.uniform(80, 150),
            "stt_transcription": random.uniform(60, 100),
            "llm_processing": random.uniform(150, 280),
            "audio_processing": random.uniform(30, 60)
        }

        # Buscar o prompt apropriado
        prompt = ConversationPrompts.get_conversation_prompt(
            conversation_type=conversation_type,
            language=language
        )

        # Contexto da conversa
        context = "\n".join([
            f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}"
            for turn in conversation_history[-3:]  # Últimos 3 turnos para contexto
        ])

        # Simular resposta baseada no tipo de conversa
        if conversation_type == ConversationType.INFORMAL:
            responses = [
                "Que interessante o que você está dizendo! Me conta mais sobre isso?",
                "Entendo perfeitamente! É realmente algo para se pensar. Como você lidaria com essa situação?",
                "Nossa, que experiência única! Como isso impactou seu dia a dia?",
                "Concordo com seu ponto de vista! Já passou por algo parecido antes?",
                "Fascinante essa perspectiva! O que te levou a pensar assim?"
            ]
        elif conversation_type == ConversationType.TECHNICAL:
            responses = [
                "Tecnicamente, isso funciona através de processos complexos mas fascinantes. Quer que eu explique em mais detalhes?",
                "Essa é uma excelente observação técnica! Você já considerou a alternativa de usar uma abordagem diferente?",
                "Do ponto de vista técnico, existem várias soluções possíveis. Qual seria sua prioridade: performance ou simplicidade?",
                "Interessante questão! A resposta envolve conceitos de arquitetura e design. Por onde prefere começar?",
                "Você tocou num ponto crucial! Isso geralmente causa confusão. Deixe-me esclarecer passo a passo, ok?"
            ]
        elif conversation_type == ConversationType.SUPPORTIVE:
            responses = [
                "Compreendo como você se sente, e é totalmente válido. Como posso ajudar você a lidar com isso?",
                "Você está sendo muito corajoso em compartilhar isso. Que tipo de apoio seria mais útil agora?",
                "É normal sentir isso, e você não está sozinho. O que tem funcionado para você em situações similares?",
                "Agradeço sua confiança em compartilhar. Vamos pensar juntos em estratégias que podem ajudar?",
                "Seus sentimentos são importantes e válidos. Como você gostaria de seguir em frente com isso?"
            ]
        else:  # CASUAL
            responses = [
                "Haha, adorei! Você tem cada história interessante! Aconteceu mais alguma coisa engraçada?",
                "Que legal! Isso me lembra de algo similar. Você costuma fazer isso com frequência?",
                "Nossa, que coincidência! Eu estava pensando sobre isso mesmo. Qual sua opinião sobre...?",
                "Sério? Que demais! Como você descobriu isso?",
                "Incrível! Você sempre teve interesse nisso ou é algo recente?"
            ]

        response = random.choice(responses)

        return response, stage_timings

    @classmethod
    def simulate_full_conversation(cls,
                                 scenario: ConversationScenario,
                                 use_predefined: bool = True) -> List[ConversationTurn]:
        """
        Simula uma conversa completa de 5 turnos

        Args:
            scenario: Cenário de conversa
            use_predefined: Se True, usa respostas predefinidas do cenário

        Returns:
            Lista de turnos da conversa
        """
        turns = []
        conversation_history = []

        for i, (user_msg, expected_response) in enumerate(scenario.turns):
            # Simular timings
            stage_timings = {
                "tts_generation": random.uniform(80, 150),
                "stt_transcription": random.uniform(60, 100),
                "llm_processing": random.uniform(150, 280),
                "audio_processing": random.uniform(30, 60)
            }

            # Calcular latência total
            total_latency = sum(stage_timings.values())

            # Usar resposta predefinida ou simular nova
            if use_predefined:
                response = expected_response
            else:
                response, stage_timings = cls.simulate_conversation_turn(
                    user_msg,
                    conversation_history,
                    scenario.conversation_type,
                    scenario.language
                )

            # Criar turno
            turn = ConversationTurn(
                turn_number=i + 1,
                user_message=user_msg,
                assistant_response=response,
                latency_ms=total_latency,
                stage_timings=stage_timings,
                success=True
            )

            turns.append(turn)

            # Atualizar histórico
            conversation_history.append({
                "user": user_msg,
                "assistant": response
            })

        return turns

    @classmethod
    def format_conversation_for_display(cls,
                                       scenario: ConversationScenario,
                                       turns: List[ConversationTurn]) -> str:
        """Formata uma conversa para exibição"""
        output = []
        output.append(f"=== {scenario.title} ===")
        output.append(f"Tipo: {scenario.conversation_type.value}")
        output.append(f"Idioma: {scenario.language}")
        output.append(f"Descrição: {scenario.description}")
        output.append("")

        for turn in turns:
            output.append(f"🗣️ Turno {turn.turn_number}:")
            output.append(f"   👤 Usuário: {turn.user_message}")
            output.append(f"   🤖 Assistente: {turn.assistant_response}")
            output.append(f"   ⏱️ Latência: {turn.latency_ms:.0f}ms")
            output.append("")

        # Estatísticas
        avg_latency = sum(t.latency_ms for t in turns) / len(turns)
        output.append(f"📊 Latência média: {avg_latency:.0f}ms")
        output.append(f"✅ Taxa de sucesso: {sum(1 for t in turns if t.success)}/{len(turns)}")

        return "\n".join(output)


def main():
    """Teste do simulador"""
    logging.basicConfig(level=logging.INFO)

    # Testar com um cenário
    scenario = ConversationSimulator.get_scenario_by_id("casual_chat")
    turns = ConversationSimulator.simulate_full_conversation(scenario)

    print(ConversationSimulator.format_conversation_for_display(scenario, turns))

    # Testar simulação dinâmica
    print("\n" + "="*50)
    print("Conversa Simulada Dinamicamente:")
    print("="*50 + "\n")

    scenario2 = ConversationSimulator.get_scenario_by_id("tech_support")
    turns2 = ConversationSimulator.simulate_full_conversation(scenario2, use_predefined=False)

    print(ConversationSimulator.format_conversation_for_display(scenario2, turns2))


if __name__ == "__main__":
    main()