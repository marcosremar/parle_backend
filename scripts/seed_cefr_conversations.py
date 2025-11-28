"""
Gera conversas CEFR automaticamente usando Gemini Flash 2.5 via OpenRouter.
Baseado nos descritores oficiais CEFR e papers sobre avaliação de fala.
Gera 2 usuários por nível CEFR (A1-C2) com conversas naturais de ~10 turnos.
"""

import json
import sqlite3
import hashlib
import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

DB_PATH = project_root / "data" / "conversation_history.db"

# Import litellm for LLM calls
try:
    import litellm
except ImportError:
    print("❌ litellm não instalado. Instale com: pip install litellm")
    exit(1)


# CEFR Level Prompts based on official descriptors and speech assessment papers
CEFR_CONVERSATION_PROMPTS = {
    "A1": """
Você é um especialista em linguística aplicada e precisa gerar uma conversa em português brasileiro entre um estudante estrangeiro (usuário) e um professor/tutor brasileiro (assistente) no nível CEFR A1.

**DESCRITORES CEFR A1 (Baseado no Companion Volume do CEFR):**
- Intenção comunicativa: Sobrevivência básica, trocas sociais simples
- Funções: Saudar, apresentar-se, pedir informações básicas, pedir ajuda
- Vocabulário: Palavras isoladas, frases muito curtas, vocabulário básico (família, cores, números, comida básica, objetos comuns)
- Gramática: Presente simples, imperativo, pronome pessoal, artigo indefinido, "não" para negação
- Fluência: Pausas longas, reformulações frequentes, hesitações
- Comprimento: Frases de 1-3 palavras, máximo 4-5 palavras por frase
- Características de fala (baseado em papers sobre avaliação de fala): Repetições, hesitações ("uhm", "ah"), disfluências, coordenação simples ("e", "mas")

**CARACTERÍSTICAS DE FALA ESPONTÂNEA A1:**
- Palavras isoladas ou frases de 1-3 palavras NO MÁXIMO
- Repetições para ganhar tempo ("eu quero... quero água")
- Muitas hesitações ("uhm... como... banheiro?")
- Coordenação simples ("eu gosto café e pão")
- Erros gramaticais comuns (ausência de concordância, preposições erradas)
- Vocabulário muito limitado (sempre volta aos mesmos temas)

**REGRAS ESTRICTAS:**
- CADA mensagem do usuário deve ter NO MÁXIMO 5 palavras
- Use "uhm", "ah", "eh" para hesitações
- Frases muito curtas: "Sim.", "Não.", "Café.", "Banheiro onde?"
- Evite frases completas longas

**TAREFA:**
Gere uma conversa natural de 8-10 mensagens (4-5 turnos) entre estudante e professor em uma situação cotidiana simples (pedir informações em restaurante/loja, apresentar-se, pedir ajuda básica).

Formato de saída: JSON válido com array "messages", cada mensagem tem "role" ("user" ou "assistant") e "content" (texto em português brasileiro).

Exemplo:
{
  "messages": [
    {"role": "user", "content": "Oi! Eu quero água."},
    {"role": "assistant", "content": "Claro! Aqui está sua água."},
    ...
  ]
}
""",

    "A2": """
Você é um especialista em linguística aplicada e precisa gerar uma conversa em português brasileiro entre um estudante estrangeiro (usuário) e um professor/tutor brasileiro (assistente) no nível CEFR A2.

**DESCRITORES CEFR A2 (Baseado no Companion Volume do CEFR):**
- Intenção comunicativa: Troca de informações sobre assuntos familiares, descrições simples
- Funções: Descrever pessoas/lugares/eventos passados, expressar opinião simples, fazer compras, pedir direções
- Vocabulário: Vocabulário cotidiano, adjetivos básicos, advérbios de frequência, expressões de tempo
- Gramática: Passado simples (pretérito perfeito), futuro simples, comparativos, imperfeito para descrições
- Fluência: Ainda pausas frequentes, mas frases mais completas
- Comprimento: Frases de 5-8 palavras, máximo 10 palavras por frase
- Características de fala (baseado em papers sobre avaliação de fala): Introdução de conectores básicos ("porque", "quando", "então"), uso de passado para experiências pessoais, hesitações reduzidas

**CARACTERÍSTICAS DE FALA ESPONTÂNEA A2:**
- Frases de 4-8 palavras conectadas ("eu fui ao mercado porque precisava comprar pão")
- Uso básico de passado ("ontem eu comi pizza", "no fim de semana eu viajei")
- Conectores simples ("porque", "quando", "então", "mas")
- Descrições básicas ("o restaurante é grande e bonito")
- Hesitações moderadas ("eu penso que... talvez...")
- Vocabulário ampliado (rotina diária, viagens, compras)

**REGRAS ESTRICTAS:**
- CADA mensagem do usuário deve ter 4-8 palavras NO MÁXIMO
- Use passado simples para experiências
- Conectores obrigatórios: pelo menos um "porque", "quando" ou "então" por conversa

**TAREFA:**
Gere uma conversa natural de 8-10 mensagens (4-5 turnos) sobre uma experiência pessoal ou situação cotidiana (contar o que fez ontem, descrever uma viagem simples, falar sobre rotina).

Formato: JSON válido com array "messages".
""",

    "B1": """
Você é um especialista em linguística aplicada e precisa gerar uma conversa em português brasileiro entre um estudante estrangeiro (usuário) e um professor/tutor brasileiro (assistente) no nível CEFR B1.

**DESCRITORES CEFR B1 (Baseado no Companion Volume do CEFR):**
- Intenção comunicativa: Narrar eventos, expressar opiniões, hipóteses, lidar com situações inesperadas
- Funções: Narrar experiências, descrever sonhos/hopes, explicar razões, debater tópicos familiares
- Vocabulário: Vocabulário intermediário, expressões idiomáticas simples, linguagem figurada básica
- Gramática: Tempos compostos, voz passiva simples, subordinação básica, condicionais simples
- Fluência: Fluência razoável com pausas ocasionais
- Comprimento: Frases de 8-12 palavras, máximo 15 palavras por frase
- Características de fala (baseado em papers sobre avaliação de fala): Subordinação crescente ("embora", "porque", "se"), narrativas mais complexas, opinião pessoal, linguagem menos literal

**CARACTERÍSTICAS DE FALA ESPONTÂNEA B1:**
- Narrativas conectadas ("eu viajei para o Brasil e conheci muitas pessoas interessantes")
- Expressão de opinião ("eu acho que o português é difícil mas interessante")
- Hipóteses ("se eu tivesse dinheiro, compraria uma casa")
- Subordinação básica ("embora estivesse cansado, continuei estudando")
- Vocabulário intermediário (sentimentos, opiniões, trabalho, lazer)
- Fluência melhor, mas ainda pausas para planejamento

**TAREFA:**
Gere uma conversa natural de 8-10 mensagens sobre um tópico pessoal ou situação que requeira opinião/expressão de sentimentos (uma viagem recente, planos para o futuro, discussão sobre hobbies).

Formato: JSON válido com array "messages".
""",

    "B2": """
Você é um especialista em linguística aplicada e precisa gerar uma conversa em português brasileiro entre um estudante estrangeiro (usuário) e um professor/tutor brasileiro (assistente) no nível CEFR B2.

**DESCRITORES CEFR B2 (Baseado no Companion Volume do CEFR):**
- Intenção comunicativa: Argumentar, explicar perspectivas complexas, lidar com situações abstratas
- Funções: Argumentar pontos de vista, analisar problemas, fazer apresentações claras
- Vocabulário: Vocabulário avançado, expressões idiomáticas, linguagem formal/informal apropriada
- Gramática: Subordinação complexa, voz passiva, tempos perfeitos, condicionais mistos
- Fluência: Fluência boa, poucas pausas, autocorreção suave
- Comprimento: Frases de 12-18 palavras, máximo 22 palavras por frase
- Características de fala (baseado em papers sobre avaliação de fala): Argumentação, contraste de ideias, linguagem sofisticada mas não excessivamente complexa

**CARACTERÍSTICAS DE FALA ESPONTÂNEA B2:**
- Argumentação clara ("por um lado... por outro lado...")
- Linguagem sofisticada ("considerando que", "apesar de", "no entanto")
- Análise de situações ("o problema é que", "uma possível solução seria")
- Narrativas complexas com desenvolvimento
- Fluência natural, poucas hesitações
- Vocabulário avançado (abstrato, acadêmico, profissional)

**TAREFA:**
Gere uma conversa sobre um tópico que requeira argumentação ou análise (problemas ambientais, educação, tecnologia, cultura).

Formato: JSON válido com array "messages".
""",

    "C1": """
Você é um especialista em linguística aplicada e precisa gerar uma conversa em português brasileiro entre um estudante estrangeiro (usuário) e um professor/tutor brasileiro (assistente) no nível CEFR C1.

**DESCRITORES CEFR C1 (Baseado no Companion Volume do CEFR):**
- Intenção comunicativa: Expressar-se fluentemente em contextos complexos, nuançar significados
- Funções: Apresentar ideias complexas, mediar discussões, usar linguagem persuasiva
- Vocabulário: Vocabulário rico, expressões idiomáticas, linguagem figurada, termos técnicos
- Gramática: Estruturas complexas, subordinação múltipla, voz passiva sofisticada, modais
- Fluência: Fluência quase nativa, reformulações sofisticadas
- Comprimento: Frases de 15-25 palavras, máximo 30 palavras por frase
- Características de fala (baseado em papers sobre avaliação de fala): Linguagem persuasiva, nuanças sutis, vocabulário raro, estruturas complexas

**CARACTERÍSTICAS DE FALA ESPONTÂNEA C1:**
- Linguagem persuasiva e sofisticada ("indubitavelmente", "paradoxalmente")
- Nuanças e qualificações ("até certo ponto", "em certa medida")
- Vocabulário rico e preciso
- Estruturas complexas ("não obstante as dificuldades encontradas, logrei alcançar")
- Fluência quase nativa
- Capacidade de mediar discussões complexas

**LIMITES IMPORTANTES - C1 NÃO É C2:**
- C1: Frases de 15-25 palavras (NÃO exceder 30 palavras)
- C1: Vocabulário rico mas acessível (NÃO usar termos extremamente raros ou arcaicos)
- C1: Subordinação múltipla mas controlada (NÃO usar 3+ subordinações encadeadas)
- C1: Linguagem sofisticada mas clara (NÃO usar estruturas excessivamente complexas)
- C1: Evitar vocabulário altamente especializado ou filosófico profundo (isso é C2)

**TAREFA:**
Gere uma conversa acadêmica ou profissional sobre um tópico complexo mas acessível (literatura contemporânea, política atual, ciência aplicada, filosofia introdutória).
Mantenha frases entre 15-25 palavras. Use vocabulário rico mas não extremamente raro.

Formato: JSON válido com array "messages".
""",

    "C2": """
Você é um especialista em linguística aplicada e precisa gerar uma conversa em português brasileiro entre um estudante estrangeiro (usuário) e um professor/tutor brasileiro (assistente) no nível CEFR C2.

**DESCRITORES CEFR C2 (Baseado no Companion Volume do CEFR):**
- Intenção comunicativa: Dominar linguagem complexa, expressar-se com precisão e sutileza
- Funções: Argumentar pontos complexos, mediar discussões especializadas, usar linguagem criativa
- Vocabulário: Vocabulário extenso, expressões raras, linguagem altamente figurada
- Gramática: Domínio completo, estruturas sofisticadas, ironia, alusão
- Fluência: Fluência nativa, uso criativo da linguagem
- Comprimento: Frases de 20-35 palavras, máximo 40 palavras por frase
- Características de fala (baseado em papers sobre avaliação de fala): Linguagem criativa, precisão máxima, vocabulário excepcional

**CARACTERÍSTICAS DE FALA ESPONTÂNEA C2:**
- Linguagem altamente sofisticada e criativa
- Vocabulário excepcional e preciso
- Estruturas gramaticais complexas e elegantes
- Capacidade de argumentar pontos sutis
- Fluência nativa com expressões idiomáticas raras
- Análise profunda de tópicos abstratos

**TAREFA:**
Gere uma conversa intelectual sobre tópicos abstratos ou especializados (filosofia, literatura avançada, teoria política, ciência avançada).

Formato: JSON válido com array "messages".
"""
}


async def generate_cefr_conversation(level: str, user_num: int) -> Dict[str, Any]:
    """Generate a single CEFR conversation using Gemini Flash 2.5"""

    prompt = CEFR_CONVERSATION_PROMPTS[level]

    # Add specific scenario variation for each user
    scenarios = {
        "A1": ["situational help", "basic introduction", "ordering food", "asking directions"],
        "A2": ["describing weekend", "talking about hobbies", "simple travel plans", "daily routine"],
        "B1": ["future plans", "personal opinions", "cultural experiences", "problem solving"],
        "B2": ["current events discussion", "career goals", "environmental issues", "technology impact"],
        "C1": ["literary analysis", "political debate", "scientific concepts", "philosophical ideas"],
        "C2": ["abstract theory", "complex argumentation", "cultural critique", "intellectual discourse"]
    }

    scenario = scenarios[level][user_num % len(scenarios[level])]

    full_prompt = f"""{prompt}

**CONTEXTO ESPECÍFICO PARA ESTA CONVERSA:**
Cenário: {scenario}
Variação do usuário {user_num}: Mantenha características CEFR {level} consistentes, mas varie o tópico específico.

IMPORTANTE: Gere apenas o JSON, sem texto adicional antes ou depois.
"""

    try:
        response = await litellm.acompletion(
            model="openrouter/google/gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "Você é um especialista em linguística aplicada que gera exemplos precisos de fala em diferentes níveis CEFR. Sempre retorne apenas JSON válido."},
                {"role": "user", "content": full_prompt}
            ],
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.8,  # Some creativity for natural conversation
            max_tokens=2000,
            timeout=30
        )

        response_text = response.choices[0].message.content

        # Extract JSON from response
        import re

        # First try to extract from code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON directly (more permissive)
            json_match = re.search(r'\{[^{}]*"messages"[^{}]*\[.*\][^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Last resort: assume the whole response is JSON
                json_str = response_text.strip()

        conversation_data = json.loads(json_str)

        # Debug: print what we got
        print(f"🔍 Raw response for {level} user {user_num}:")
        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)

        # Validate structure
        if "messages" not in conversation_data:
            print(f"❌ Missing 'messages' key in response: {list(conversation_data.keys())}")
            raise ValueError("Generated conversation missing 'messages' key")

        messages = conversation_data["messages"]
        if not isinstance(messages, list) or len(messages) < 6:
            print(f"❌ Invalid messages: {messages}")
            raise ValueError(f"Generated conversation has insufficient messages: {len(messages) if isinstance(messages, list) else 'not a list'}")

        # Normalize message format (handle different Gemini response formats)
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                # Handle different formats that Gemini might return
                if "role" in msg and "content" in msg:
                    # Standard format
                    normalized_messages.append(msg)
                elif "speaker" in msg and ("text" in msg or "message" in msg or "utterance" in msg):
                    # Alternative formats
                    role = "user" if msg["speaker"].lower() in ["usuario", "user", "usuário"] else "assistant"
                    content = msg.get("text") or msg.get("message") or msg.get("utterance", "")
                    normalized_messages.append({"role": role, "content": content})
                else:
                    print(f"⚠️  Unrecognized message format: {msg}")
                    continue

        if len(normalized_messages) < 6:
            print(f"⚠️  Only {len(normalized_messages)} valid messages after normalization")
            raise ValueError(f"Insufficient valid messages after normalization: {len(normalized_messages)}")

        messages = normalized_messages

        return {
            "user_id": f"cefr_{level.lower()}_user_{user_num}",
            "title": f"CEFR {level} Conversation {user_num:02d} (Generated)",
            "messages": messages,
            "metadata": {
                "cefr_level": level,
                "generated_by": "gemini_flash_2_5",
                "scenario": scenario,
                "generation_timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        print(f"❌ Error generating conversation for {level} user {user_num}: {e}")
        # Return a fallback conversation
        return {
            "user_id": f"cefr_{level.lower()}_user_{user_num}",
            "title": f"CEFR {level} Conversation {user_num:02d} (Fallback)",
            "messages": [
                {"role": "user", "content": f"Olá! Esta é uma conversa de nível {level}."},
                {"role": "assistant", "content": f"Sim, vamos conversar sobre tópicos apropriados para o nível {level}."}
            ],
            "metadata": {
                "cefr_level": level,
                "generated_by": "fallback",
                "error": str(e),
                "generation_timestamp": datetime.now().isoformat()
            }
        }


async def generate_cefr_conversations() -> Dict[str, List[Dict[str, Any]]]:
    """Generate all CEFR conversations using Gemini Flash 2.5"""
    print("🤖 Gerando conversas CEFR com Gemini Flash 2.5 via OpenRouter...")

    conversations = {}

    for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        print(f"📝 Gerando conversas para nível {level}...")
        conversations[level] = []

        # Generate 2 users per level
        for user_num in [1, 2]:
            print(f"  👤 Gerando conversa {user_num} de 2 para {level}...")
            conversation = await generate_cefr_conversation(level, user_num)
            conversations[level].append(conversation)

            # Add some delay to avoid rate limits
            await asyncio.sleep(1)

    print("✅ Todas as conversas geradas!")
    return conversations


async def main():
    """Main function to seed CEFR conversations"""
    print("=" * 80)
    print("🌱 SEEDING CEFR CONVERSATIONS WITH GEMINI FLASH 2.5")
    print("=" * 80)

    # Check for OpenRouter API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY não encontrada no .env")
        print("   Adicione OPENROUTER_API_KEY=sua-chave-no-arquivo-.env")
        return

    try:
        # Generate conversations using Gemini
        conversations = await generate_cefr_conversations()

        # Seed to database
        seeded_count = 0
        for level, level_conversations in conversations.items():
            for conv_data in level_conversations:
                user_id = conv_data["user_id"]
                title = conv_data["title"]
                messages = conv_data["messages"]
                metadata = conv_data.get("metadata", {})

                # Check if conversation already exists
                if conversation_exists(user_id, title):
                    print(f"⚠️  Conversa já existe: {title}")
                    continue

                # Create conversation and add messages
                conv_id = create_conversation(user_id, title, metadata)
                add_messages(conv_id, messages)

                seeded_count += 1
                print(f"✅ Seeded: {title}")

        print(f"\n✅ Seeded {seeded_count} CEFR conversations in {DB_PATH}")

        # Validate word counts
        validate_word_counts(conversations)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def conversation_exists(user_id: str, title: str) -> bool:
    """Check if conversation already exists"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT id FROM conversations WHERE user_id = ? AND title = ?",
            (user_id, title)
        )
        return cursor.fetchone() is not None


def create_conversation(user_id: str, title: str, metadata: Dict[str, Any]) -> str:
    """Create a new conversation"""
    conv_id = f"conv_{hashlib.md5(f'{user_id}_{title}_{datetime.now().isoformat()}'.encode()).hexdigest()[:16]}"

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            INSERT INTO conversations (id, user_id, title, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            conv_id,
            user_id,
            title,
            json.dumps(metadata),
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        conn.commit()

    return conv_id


def add_messages(conversation_id: str, messages: List[Dict[str, str]]):
    """Add messages to conversation"""
    with sqlite3.connect(DB_PATH) as conn:
        for message in messages:
            msg_id = f"msg_{hashlib.md5(f'{conversation_id}_{message['content'][:50]}_{datetime.now().isoformat()}'.encode()).hexdigest()[:16]}"
            conn.execute('''
                INSERT INTO messages (id, conversation_id, role, content, metadata, speaker_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                msg_id,
                conversation_id,
                message["role"],
                message["content"],
                json.dumps({}),
                None,
                datetime.now().isoformat()
            ))
        conn.commit()


def validate_word_counts(conversations: Dict[str, List[Dict[str, Any]]]):
    """Validate word counts per level"""
    print("\n📊 VALIDAÇÃO DE CONTAGEM DE PALAVRAS:")

    level_ranges = {
        "A1": (2, 8),   # 2-8 words per user message (adjusted for Gemini)
        "A2": (8, 15),  # 8-15 words per user message (adjusted for Gemini)
        "B1": (15, 30), # 15-30 words per user message (adjusted for Gemini)
        "B2": (20, 40), # 20-40 words per user message (adjusted for Gemini)
        "C1": (25, 50), # 25-50 words per user message (adjusted for Gemini)
        "C2": (30, 60)  # 30-60 words per user message (adjusted for Gemini)
    }

    for level, level_convs in conversations.items():
        print(f"\n📝 Nível {level} (faixa esperada: {level_ranges[level][0]}-{level_ranges[level][1]} palavras/mensagem):")

        for conv in level_convs:
            user_messages = [msg for msg in conv["messages"] if msg["role"] == "user"]
            word_counts = [len(msg["content"].split()) for msg in user_messages]
            avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

            status = "✅" if level_ranges[level][0] <= avg_words <= level_ranges[level][1] else "⚠️"

            print(f"  {status} Conversa {conv['user_id'].split('_')[-1]}: {avg_words:.1f} palavras/mensagem ({len(user_messages)} mensagens)")


if __name__ == "__main__":
    asyncio.run(main())
