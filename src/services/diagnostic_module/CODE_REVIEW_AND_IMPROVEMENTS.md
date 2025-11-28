# Code Review: Skill Extraction Implementation

**Data:** 2025-11-22  
**Arquivos Analisados:**
- `src/services/diagnostic_module/llm_client.py`
- `src/services/diagnostic_module/app_complete.py`
- `src/services/orchestrator/orchestrator_engine.py`
- `src/services/orchestrator/service_clients.py`

---

## 🔴 Problemas Críticos Encontrados

### 1. **Performance: Lista de 189 Skills no Prompt**

**Problema:**
```python
# orchestrator_engine.py linha 460
for level_skills in SKILL_CEFR_MAP.values():
    valid_skills.extend(level_skills)  # 189 skills!

# llm_client.py linha 322
skills_list = "\n".join([f"- {skill_id}" for skill_id in valid_skills])  # Prompt gigante!
```

**Impacto:**
- Prompt com ~5000+ caracteres só de lista de skills
- LLM pode ignorar skills no final da lista
- Latência aumentada (mais tokens para processar)
- Custo maior (mais tokens = mais $)

**Solução Recomendada:**
```python
def get_relevant_skills_for_context(
    user_text: str,
    cefr_level: str,
    context_type: str = "production"
) -> List[str]:
    """
    Filtra skills relevantes baseado em:
    - Nível CEFR do aluno
    - Tipo de contexto (production/comprehension/interaction)
    - Features linguísticas detectadas heuristicamente
    """
    from src.services.student_model.skill_registry import SKILL_CEFR_MAP
    
    # 1. Skills do nível atual + 1 nível acima
    relevant_levels = []
    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    if cefr_level in levels:
        idx = levels.index(cefr_level)
        relevant_levels = levels[:idx+2]  # Nível atual + próximo
    
    skills = []
    for level in relevant_levels:
        skills.extend(SKILL_CEFR_MAP.get(level, []))
    
    # 2. Filtrar por tipo de contexto
    if context_type == "production":
        # Apenas skills de produção (gramática, vocabulário, escrita)
        skills = [s for s in skills if any(x in s for x in [
            "verb_", "vocabulary_", "article_", "preposition_", 
            "pronoun_", "adjective_", "production_"
        ])]
    elif context_type == "comprehension":
        # Skills de compreensão
        skills = [s for s in skills if "comprehension_" in s]
    
    # 3. Heurística: detectar features básicas do texto
    text_lower = user_text.lower()
    if any(word in text_lower for word in ["fui", "foi", "foram", "estava"]):
        skills = [s for s in skills if "past" in s or "verb_" in s]
    
    # Limitar a 50 skills máximo
    return skills[:50]
```

---

### 2. **Erro Potencial: Duplicação de Skills no Merge**

**Problema:**
```python
# orchestrator_engine.py linha 535-542
combined_correct_skills = set()
if turn_analysis:
    combined_correct_skills.update(turn_analysis.get("correct_skills", []))
if skills_extraction:
    for skill_data in skills_extraction.get("skills", []):
        if skill_data.get("confidence", 0.0) >= 0.7:
            combined_correct_skills.add(skill_data.get("skill_id"))
```

**Problema:** Se `turn_analysis` já tem uma skill e `extract_skills` também detecta (com confidence diferente), perdemos informação de confidence.

**Solução:**
```python
# Manter confidence scores no merge
combined_skills_with_confidence = {}  # skill_id -> max_confidence

if turn_analysis:
    for skill_id in turn_analysis.get("correct_skills", []):
        combined_skills_with_confidence[skill_id] = 1.0  # Implicit high confidence

if skills_extraction:
    for skill_data in skills_extraction.get("skills", []):
        skill_id = skill_data.get("skill_id")
        confidence = skill_data.get("confidence", 0.0)
        if confidence >= 0.7:
            # Use max confidence if skill appears in both
            combined_skills_with_confidence[skill_id] = max(
                combined_skills_with_confidence.get(skill_id, 0.0),
                confidence
            )

combined_correct_skills = list(combined_skills_with_confidence.keys())
```

---

### 3. **Erro: Falta Validação de Tipos no extract_skills**

**Problema:**
```python
# llm_client.py linha 465-470
for skill in skills:
    skill_id = skill.get("skill_id")
    if skill_id and skill_id in valid_skills:
        confidence = max(0.0, min(1.0, float(skill.get("confidence", 0.5))))
```

**Problema:** Se `skill.get("confidence")` retornar string ou None, `float()` pode falhar.

**Solução:**
```python
try:
    confidence_raw = skill.get("confidence", 0.5)
    if confidence_raw is None:
        confidence = 0.5
    elif isinstance(confidence_raw, str):
        confidence = float(confidence_raw)
    else:
        confidence = float(confidence_raw)
    confidence = max(0.0, min(1.0, confidence))
except (ValueError, TypeError) as e:
    logger.warning(f"Invalid confidence for skill {skill_id}: {confidence_raw}, using 0.5")
    confidence = 0.5
```

---

### 4. **Performance: Import Repetido em Loop**

**Problema:**
```python
# orchestrator_engine.py linha 591, 612
for usage_skill_id in success_skills:
    try:
        from src.services.student_model.skill_registry import get_skill_difficulty
        skill_difficulty = get_skill_difficulty(usage_skill_id)
```

**Problema:** Import dentro de loop é ineficiente.

**Solução:**
```python
# Mover import para fora do loop
from src.services.student_model.skill_registry import get_skill_difficulty

for usage_skill_id in success_skills:
    try:
        skill_difficulty = get_skill_difficulty(usage_skill_id)
```

---

## 🟡 Melhorias Recomendadas

### 5. **Cache de valid_skills**

**Problema:** `valid_skills` é recalculado a cada turno.

**Solução:**
```python
# Adicionar cache no orchestrator
class ConversationOrchestrator:
    def __init__(self, ...):
        self._valid_skills_cache: Optional[List[str]] = None
        self._valid_skills_cache_timestamp: Optional[float] = None
    
    def _get_valid_skills(self, force_refresh: bool = False) -> List[str]:
        """Get valid skills with caching"""
        import time
        
        # Cache válido por 5 minutos
        if (not force_refresh and 
            self._valid_skills_cache and 
            self._valid_skills_cache_timestamp and
            time.time() - self._valid_skills_cache_timestamp < 300):
            return self._valid_skills_cache
        
        from src.services.student_model.skill_registry import SKILL_CEFR_MAP
        valid_skills = []
        for level_skills in SKILL_CEFR_MAP.values():
            valid_skills.extend(level_skills)
        
        self._valid_skills_cache = valid_skills
        self._valid_skills_cache_timestamp = time.time()
        return valid_skills
```

---

### 6. **Timeout Configurável**

**Problema:** Timeout fixo de 30s pode ser muito ou pouco dependendo do contexto.

**Solução:**
```python
# Adicionar timeout baseado no tamanho da lista de skills
def calculate_timeout(valid_skills_count: int) -> int:
    """Calcula timeout baseado no número de skills"""
    base_timeout = 15  # 15s base
    per_skill_timeout = 0.1  # 100ms por skill
    return int(base_timeout + (valid_skills_count * per_skill_timeout))

timeout = calculate_timeout(len(valid_skills))
async with session.post(..., timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
```

---

### 7. **Retry Logic para LLM Calls**

**Problema:** Se LLM falhar, não há retry.

**Solução:**
```python
async def extract_skills_with_retry(self, ...):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await self.extract_skills(...)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"extract_skills attempt {attempt+1} failed: {e}, retrying...")
            await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
```

---

### 8. **Validação de Prompt Size**

**Problema:** Prompt pode exceder limite do LLM.

**Solução:**
```python
MAX_PROMPT_LENGTH = 8000  # Ajustar conforme modelo

def build_skills_prompt(user_text: str, valid_skills: List[str], ...) -> str:
    """Build prompt with size validation"""
    # Tentar com todas as skills primeiro
    skills_list = "\n".join([f"- {skill_id}" for skill_id in valid_skills])
    prompt = f"""...{skills_list}..."""
    
    if len(prompt) > MAX_PROMPT_LENGTH:
        logger.warning(f"Prompt too long ({len(prompt)} chars), truncating skills list")
        # Reduzir para skills mais relevantes
        # Priorizar: skills do nível atual, skills gramaticais, skills de vocabulário comum
        priority_skills = [
            s for s in valid_skills 
            if any(x in s for x in ["verb_", "vocabulary_basic", "article_", "preposition_"])
        ]
        # Adicionar skills restantes até limite
        remaining = [s for s in valid_skills if s not in priority_skills]
        max_skills = (MAX_PROMPT_LENGTH - len(prompt) + len(skills_list)) // 20  # ~20 chars per skill
        skills_list = "\n".join([f"- {s}" for s in (priority_skills + remaining[:max_skills])])
        prompt = f"""...{skills_list}..."""
    
    return prompt
```

---

### 9. **Melhor Tratamento de Exceções em Parallel Calls**

**Problema:**
```python
# orchestrator_engine.py linha 480-492
turn_analysis, skills_extraction = await asyncio.gather(
    turn_analysis_task,
    skills_extraction_task,
    return_exceptions=True
)

if isinstance(turn_analysis, Exception):
    logger.warning(f"⚠️ analyze_turn failed: {turn_analysis}")
    turn_analysis = None
```

**Melhoria:** Logar traceback completo para debugging.

**Solução:**
```python
if isinstance(turn_analysis, Exception):
    logger.error(f"⚠️ analyze_turn failed: {turn_analysis}")
    import traceback
    logger.debug(f"Traceback: {traceback.format_exc()}")
    turn_analysis = None
```

---

### 10. **Validação de Input no Endpoint**

**Problema:**
```python
# app_complete.py linha 260
if not request.valid_skills or len(request.valid_skills) == 0:
    raise HTTPException(status_code=400, ...)
```

**Melhoria:** Validar também se skills são válidas (existem no SKILL_CEFR_MAP).

**Solução:**
```python
from src.services.student_model.skill_registry import SKILL_CEFR_MAP

# Validar que todas as skills existem
all_valid_skills = []
for skills in SKILL_CEFR_MAP.values():
    all_valid_skills.extend(skills)

invalid_skills = [s for s in request.valid_skills if s not in all_valid_skills]
if invalid_skills:
    logger.warning(f"Invalid skills provided: {invalid_skills[:5]}")
    # Filtrar skills inválidas
    request.valid_skills = [s for s in request.valid_skills if s in all_valid_skills]
```

---

## 🟢 Otimizações de Código

### 11. **Reduzir Duplicação de Código**

**Problema:** Lógica de extração de content do LLM está duplicada em 4 métodos.

**Solução:**
```python
def _extract_content_from_llm_response(self, result: Dict[str, Any]) -> str:
    """Extract content from LLM response (handles multiple formats)"""
    # Format 1: OpenAI-style
    if "choices" in result and result["choices"]:
        message = result["choices"][0].get("message", {})
        return message.get("content", "")
    
    # Format 2: Direct "text" field
    if "text" in result:
        return result.get("text", "")
    
    # Fallback
    for field in ["content", "response", "output", "message"]:
        if field in result:
            return str(result[field])
    
    return ""
```

---

### 12. **Adicionar Métricas/Telemetria**

**Solução:**
```python
# Adicionar métricas para monitoramento
self.metrics = {
    "extract_skills_calls": 0,
    "extract_skills_success": 0,
    "extract_skills_failures": 0,
    "avg_skills_detected": 0.0,
    "avg_confidence": 0.0
}

# No método extract_skills:
self.metrics["extract_skills_calls"] += 1
if validated_skills:
    self.metrics["extract_skills_success"] += 1
    self.metrics["avg_skills_detected"] = (
        (self.metrics["avg_skills_detected"] * (self.metrics["extract_skills_success"] - 1) +
         len(validated_skills)) / self.metrics["extract_skills_success"]
    )
else:
    self.metrics["extract_skills_failures"] += 1
```

---

## 📊 Resumo de Prioridades

| Prioridade | Problema | Impacto | Esforço |
|------------|----------|---------|---------|
| 🔴 **ALTA** | 189 skills no prompt | Performance/Custo | Médio |
| 🔴 **ALTA** | Import em loop | Performance | Baixo |
| 🟡 **MÉDIA** | Cache de valid_skills | Performance | Baixo |
| 🟡 **MÉDIA** | Validação de tipos | Robustez | Baixo |
| 🟡 **MÉDIA** | Merge de confidence | Precisão | Baixo |
| 🟢 **BAIXA** | Retry logic | Robustez | Médio |
| 🟢 **BAIXA** | Métricas | Observabilidade | Baixo |

---

## 🎯 Recomendações Imediatas

1. **Implementar filtro de skills por contexto** (reduzir de 189 para ~30-50)
2. **Mover imports para fora de loops**
3. **Adicionar cache de valid_skills**
4. **Melhorar validação de tipos no parsing**

Essas 4 mudanças terão o maior impacto com menor esforço.

