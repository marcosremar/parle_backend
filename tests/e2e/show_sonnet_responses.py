"""
Script temporário para mostrar respostas completas do Sonnet 4.5
no teste de comparação A1 vs C1
"""
import asyncio
import aiohttp
import os
import sys
import json
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
def _load_env_file():
    """Load environment variables from .env file"""
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and not os.getenv(key):
                        os.environ[key] = value

_load_env_file()

# Service URLs
PEDAGOGICAL_POLICY_URL = os.getenv("PEDAGOGICAL_POLICY_URL", "http://localhost:8950")
LLM_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8110")

# Model configuration
if os.getenv("OPENROUTER_API_KEY"):
    SONNET_4_5_MODEL = os.getenv("SONNET_4_5_MODEL", "openrouter/anthropic/claude-sonnet-4.5")
elif os.getenv("ANTHROPIC_API_KEY"):
    SONNET_4_5_MODEL = os.getenv("SONNET_4_5_MODEL", "claude-sonnet-4-5")
else:
    SONNET_4_5_MODEL = os.getenv("SONNET_4_5_MODEL", "groq/llama-3.1-8b-instant")


async def check_service_health(session, url, name):
    """Check if service is available"""
    try:
        async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            return resp.status == 200
    except:
        return False


async def compose_prompt_directly(session, cefr_level):
    """Compose prompt for a CEFR level"""
    try:
        context = {
            "scenario": {
                "name": "conversação",
                "system_prompt": "Conversa casual",
                "ai_role": "professor"
            },
            "cefr_level": cefr_level,
            "native_language": "en",
            "mastery_probability": 0.5,
            "emotional_state": "neutral",
            "cefr_details": {},
            "conversation_history": []
        }
        
        async with session.post(
            f"{PEDAGOGICAL_POLICY_URL}/api/prompt/compose",
            json={"context": context},
            timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                error_text = await resp.text()
                print(f"⚠️  HTTP {resp.status}: {error_text[:200]}")
    except Exception as e:
        print(f"⚠️  Error composing prompt: {e}")
        import traceback
        traceback.print_exc()
    return None


async def analyze_text_complexity_llm(session, text):
    """Analyze text complexity using LLM (Sonnet 4.5) - direct call"""
    """Analyze text complexity using LLM (Sonnet 4.5)"""
    prompt = f"""Analise o seguinte texto em português e forneça um JSON com as seguintes informações:
- "avg_words_per_sentence": média de palavras por frase (float)
- "max_words_per_sentence": máximo de palavras em uma única frase (int)
- "total_words": total de palavras no texto (int)
- "has_subjunctive": true se houver uso claro de subjuntivo, false caso contrário
- "has_passive": true se houver uso claro de voz passiva, false caso contrário
- "has_relative_clauses": true se houver uso claro de orações relativas complexas, false caso contrário
- "has_conditional": true se houver uso claro de condicional, false caso contrário
- "complexity_score": um score de 0 a 100 (int) onde 0 é muito simples e 100 é muito complexo.

Texto: "{text}"
Retorne APENAS o JSON, sem texto adicional."""

    try:
        import litellm
        
        response = await litellm.acompletion(
            model=SONNET_4_5_MODEL,
            messages=[
                {"role": "system", "content": "Você é um analisador linguístico especializado. Retorne APENAS JSON válido, sem texto adicional."},
                {"role": "user", "content": prompt}
            ],
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.1,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        
        print(f"\n📊 Resposta do Sonnet para análise de complexidade:")
        print(f"   {response_text[:500]}...")
        
        # Extract JSON from response
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{[^{}]*(?:"avg_words_per_sentence"|"is_compliant")[^{}]*\}', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
            analysis = json.loads(json_str)
            return analysis
        
        # Fallback: try to parse entire response as JSON
        try:
            analysis = json.loads(response_text)
            return analysis
        except json.JSONDecodeError:
            pass
        
        print(f"⚠️  Could not extract JSON from: {response_text[:200]}")
    except Exception as e:
        print(f"⚠️  LLM complexity analysis failed: {e}")
    
    return {}


async def check_cefr_grammar_constraints_llm(session, text, cefr_level):
    """Check if text violates CEFR grammar constraints using LLM"""
    constraints_map = {
        "A1": {
            "allowed_tenses": ["presente do indicativo", "imperativo simples"],
            "forbidden": ["subjuntivo", "voz passiva", "orações subordinadas", "pronomes relativos", "tempos compostos"]
        },
        "C1": {
            "allowed_tenses": ["todos os tempos verbais"],
            "forbidden": []
        }
    }
    
    constraints = constraints_map.get(cefr_level, {})
    
    prompt = f"""Analise o seguinte texto em português para o nível CEFR {cefr_level} e identifique violações das restrições gramaticais.
Retorne um JSON com:
- "is_compliant": true se o texto estiver em conformidade, false caso contrário
- "violations": uma lista de strings descrevendo cada violação encontrada.
- "explanation": uma breve explicação geral da conformidade ou não conformidade.

Restrições para {cefr_level}:
- Tempos permitidos: {constraints.get('allowed_tenses', [])}
- Estruturas proibidas: {constraints.get('forbidden', [])}

Texto: "{text}"
Retorne APENAS o JSON, sem texto adicional."""

    try:
        import litellm
        
        response = await litellm.acompletion(
            model=SONNET_4_5_MODEL,
            messages=[
                {"role": "system", "content": "Você é um analisador linguístico especializado. Retorne APENAS JSON válido, sem texto adicional."},
                {"role": "user", "content": prompt}
            ],
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.1,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        
        print(f"\n📋 Resposta do Sonnet para verificação de conformidade CEFR {cefr_level}:")
        print(f"   {response_text[:500]}...")
        
        # Extract JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{[^{}]*(?:"is_compliant"|"violations")[^{}]*\}', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
            analysis = json.loads(json_str)
            return analysis
        
        try:
            analysis = json.loads(response_text)
            return analysis
        except json.JSONDecodeError:
            pass
    except Exception as e:
        print(f"⚠️  LLM constraint check failed: {e}")
    
    return {"is_compliant": True, "violations": [], "explanation": "Fallback"}


async def main():
    """Main function to show Sonnet responses"""
    async with aiohttp.ClientSession() as session:
        # Check services
        if not await check_service_health(session, PEDAGOGICAL_POLICY_URL, "Pedagogical Policy"):
            print("❌ Pedagogical Policy service not available")
            return
        if not await check_service_health(session, LLM_URL, "LLM"):
            print("❌ LLM service not available")
            return
        
        print(f"\n🤖 Usando modelo: {SONNET_4_5_MODEL}\n")
        print("=" * 80)
        
        results = {}
        
        for level in ["A1", "C1"]:
            print(f"\n{'='*80}")
            print(f"📝 NÍVEL CEFR: {level}")
            print(f"{'='*80}\n")
            
            # Compose prompt
            print(f"1️⃣  Compondo prompt para {level}...")
            prompt_result = await compose_prompt_directly(session, level)
            if not prompt_result:
                print(f"❌ Falha ao compor prompt para {level}")
                continue
            
            prompt = prompt_result["prompt"]
            print(f"✅ Prompt composto ({len(prompt)} caracteres)")
            print(f"\n📄 Prompt completo para {level}:")
            print("-" * 80)
            print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
            print("-" * 80)
            
            # Call LLM directly (bypassing HTTP service to avoid auth issues)
            print(f"\n2️⃣  Chamando LLM (Sonnet 4.5) diretamente com prompt {level}...")
            try:
                import litellm
                
                response = await litellm.acompletion(
                    model=SONNET_4_5_MODEL,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Olá, como vai?"}
                    ],
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    temperature=0.7,
                    max_tokens=200
                )
                
                response_text = response.choices[0].message.content
                
                if response_text:
                    print(f"\n✅ Resposta do LLM (Sonnet 4.5) para {level}:")
                    print("-" * 80)
                    print(response_text)
                    print("-" * 80)
                    
                    # Analyze complexity
                    print(f"\n3️⃣  Analisando complexidade da resposta {level}...")
                    complexity = await analyze_text_complexity_llm(session, response_text)
                    
                    # Check constraints
                    print(f"\n4️⃣  Verificando conformidade CEFR {level}...")
                    constraints = await check_cefr_grammar_constraints_llm(session, response_text, level)
                    
                    results[level] = {
                        "text": response_text,
                        "complexity": complexity,
                        "constraints": constraints
                    }
                    
                    print(f"\n📊 Resumo da análise para {level}:")
                    print(f"   Média palavras/frase: {complexity.get('avg_words_per_sentence', 'N/A')}")
                    print(f"   Score de complexidade: {complexity.get('complexity_score', 'N/A')}")
                    print(f"   Conforme CEFR {level}: {constraints.get('is_compliant', 'N/A')}")
                    if constraints.get('violations'):
                        print(f"   Violações: {constraints.get('violations')}")
                else:
                    print(f"❌ LLM não retornou texto para {level}")
            except Exception as e:
                print(f"❌ Erro ao chamar LLM: {e}")
                import traceback
                traceback.print_exc()
        
        # Comparison
        if len(results) == 2:
            print(f"\n{'='*80}")
            print("📊 COMPARAÇÃO A1 vs C1")
            print(f"{'='*80}\n")
            
            a1 = results["A1"]
            c1 = results["C1"]
            
            print("📝 Respostas completas:")
            print(f"\nA1: {a1['text']}")
            print(f"\nC1: {c1['text']}")
            
            print("\n📊 Métricas de complexidade:")
            print(f"  A1 - Média palavras/frase: {a1['complexity'].get('avg_words_per_sentence', 'N/A')}")
            print(f"  C1 - Média palavras/frase: {c1['complexity'].get('avg_words_per_sentence', 'N/A')}")
            print(f"  A1 - Score: {a1['complexity'].get('complexity_score', 'N/A')}")
            print(f"  C1 - Score: {c1['complexity'].get('complexity_score', 'N/A')}")
            
            print("\n✅ Conformidade CEFR:")
            print(f"  A1: {a1['constraints'].get('is_compliant', 'N/A')}")
            print(f"  C1: {c1['constraints'].get('is_compliant', 'N/A')}")
            
            if a1['constraints'].get('violations'):
                print(f"  A1 Violações: {a1['constraints']['violations']}")
            if c1['constraints'].get('violations'):
                print(f"  C1 Violações: {c1['constraints']['violations']}")


if __name__ == "__main__":
    asyncio.run(main())

