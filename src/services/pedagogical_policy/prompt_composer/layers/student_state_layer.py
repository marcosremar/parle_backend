"""
Student State Layer - Informações sobre o estado do estudante
"""

from typing import Optional
from ...models import CEFRLevel


class StudentStateLayer:
    """Camada que fornece informações sobre o estado do estudante"""
    
    CEFR_DESCRIPTIONS = {
        CEFRLevel.A1: "Iniciante absoluto. Precisa de vocabulário básico e frases muito simples.",
        CEFRLevel.A2: "Iniciante-Intermediário. Consegue comunicar em situações simples do dia a dia.",
        CEFRLevel.B1: "Intermediário. Consegue lidar com a maioria das situações de viagem.",
        CEFRLevel.B2: "Intermediário-Alto. Consegue interagir com fluência e naturalidade.",
        CEFRLevel.C1: "Avançado. Consegue usar a língua de forma flexível e eficaz.",
        CEFRLevel.C2: "Proficiente. Domina a língua quase como um nativo."
    }
    
    CEFR_INSTRUCTIONS = {
        CEFRLevel.A1: """🗣️ ESTILO CONVERSACIONAL A1:
- Respostas MUITO curtas: 1-2 frases simples (3-8 palavras cada)
- Faça UMA pergunta simples por vez para o estudante falar
- APENAS presente do indicativo e imperativo básico
- Estrutura SVO simples (Sujeito-Verbo-Objeto)
- Vocabulário: 500-1000 palavras mais frequentes
- Conectores: APENAS "e", "ou", "mas"
- Exemplos: "Olá! Como vai?" / "Quer café?" / "Gosto de você."
- NÃO USE: subordinação, subjuntivo, voz passiva, orações relativas""",
        
        CEFRLevel.A2: """🗣️ ESTILO CONVERSACIONAL A2 [IMPORTANTE: NÃO É A1!]:
- Respostas curtas: 2-3 frases (6-10 palavras cada - MÍNIMO 6 palavras por frase!)
- OBRIGATÓRIO: Use PELO MENOS 1 frase com "porque" ou "quando"
- OBRIGATÓRIO: Use PELO MENOS 1 vez pretérito perfeito (fui, visitei, comprei, etc.)
- Tempos verbais: presente + pretérito perfeito + futuro simples (vou + verbo)
- Vocabulário: rotina diária específica (café da manhã, trabalho, fim de semana, ontem, amanhã)
- Conectores: "porque", "mas", "então", "e depois", "também"
- Exemplo BOM: "Ontem eu visitei a praia porque fazia sol. Você já foi lá? É muito bonito no verão!"
- Exemplo RUIM (muito A1): "Olá! Como vai? Quer comida?" ❌
- EVITE: subjuntivo, voz passiva, orações relativas complexas""",
        
        CEFRLevel.B1: """🗣️ ESTILO CONVERSACIONAL B1 [IMPORTANTE: NÃO É A2!]:
- Respostas médias: 3-4 frases (10-15 palavras cada - MÍNIMO 10 palavras por frase!)
- OBRIGATÓRIO: Use subordinação em PELO MENOS 2 frases ("porque", "quando", "se", "que")
- OBRIGATÓRIO: Use diferentes tempos verbais (presente, pretérito perfeito, pretérito imperfeito OU futuro)
- Vocabulário: opinião, experiências passadas, planos futuros, trabalho, estudos
- Conectores variados: "porque", "quando", "se", "mas também", "por isso", "além disso"
- Exemplo BOM: "Eu gosto muito deste restaurante porque a comida é excelente e o atendimento sempre foi ótimo. Quando vim aqui pela última vez, experimentei o prato do dia e gostei bastante. Se você ainda não decidiu, posso recomendar algo?"
- Exemplo RUIM (muito A2): "Bem-vindo! Você quer mesa? Temos menu bom!" ❌
- Subjuntivo: MUITO raro, apenas "seja" em expressões fixas
- Voz passiva: pode aparecer ocasionalmente (ex: "é recomendado", "foi construído")""",
        
        CEFRLevel.B2: """🗣️ ESTILO CONVERSACIONAL B2:
- Respostas balanceadas: 3-5 frases (10-18 palavras cada)
- Contribua com ideias E pergunte para aprofundar
- Todos os tempos verbais, incluindo compostos
- Subordinação variada e natural
- Voz passiva: use naturalmente quando apropriado
- Subjuntivo: use ATIVAMENTE (não apenas expressões fixas)
- Orações relativas: todos os pronomes, incluindo "cujo"
- Vocabulário: 3500-5000 palavras (abstrato, opinião complexa)
- Exemplo: "Concordo que seja fundamental analisar essa questão, embora muitos discordem. A perspectiva que você trouxe, cujo mérito é inegável, complementa o que discutimos antes. Como você relacionaria isso com...?"
- Marcadores: "isto é", "em outras palavras", "dessa forma" """,
        
        CEFRLevel.C1: """🗣️ ESTILO CONVERSACIONAL C1:
- Respostas elaboradas: 4-6 frases (15-25 palavras cada)
- Desenvolva ideias complexas mas mantenha diálogo interativo
- Subordinações múltiplas, encadeamento complexo
- Voz passiva: natural, variada, incluindo formas sofisticadas
- Subjuntivo: use em TODOS os contextos apropriados (hipóteses, dúvidas, desejos)
- Orações relativas: complexas, incluindo explicativas e restritivas
- Vocabulário: 5000-8000 palavras (abstrato, técnico, nuances)
- Expressões idiomáticas: naturais e variadas
- Exemplo: "A questão que levantou, conquanto seja desafiadora, merece ser analisada sob múltiplas óticas, visto que as implicações decorrentes afetam não apenas o contexto imediato, mas também desdobramentos futuros que podem ser antecipados caso consideremos as variáveis envolvidas. Teria ocorrido algo semelhante na sua experiência?"
- Conectores sofisticados: "conquanto", "visto que", "à medida que" """,
        
        CEFRLevel.C2: """🗣️ ESTILO CONVERSACIONAL C2:
- Respostas nativas: 5-8 frases (20-30+ palavras cada)
- Demonstre domínio total mas permaneça conversacional
- Encadeamento muito complexo, estruturas raras/literárias
- Voz passiva sofisticada, formas raras e eruditas
- Subjuntivo nativo: formas compostas, raras, literárias
- Orações relativas: todas, incluindo as mais raras e complexas
- Vocabulário: 8000+ palavras (nativo, criativo, especializado)
- Expressões idiomáticas criativas, neologismos apropriados
- Exemplo: "Depreende-se da sua colocação, cuja perspicácia é digna de nota, que as nuances subjacentes ao fenômeno em questão transcendem a mera superficialidade com que costumeiramente é tratado, entrelaçando-se em uma teia de significados que, conquanto intrincada, revela-se elucidativa ao observador atento que se disponha a deslindar suas camadas mais profundas. Teria sido este o raciocínio que o levou a tal conclusão?"
- Conectores criativos, sofisticados, até eruditos"""
    }
    
    CEFR_GRAMMAR_CONSTRAINTS = {
        CEFRLevel.A1: {
            "tempos_verbais": ["presente do indicativo", "imperativo simples"],
            "pronomes": ["pessoais básicos (eu, você, ele/ela, nós, vocês, eles/elas)"],
            "estruturas": [
                "frases afirmativas simples (SVO: Sujeito-Verbo-Objeto)",
                "perguntas com quê/quem/onde",
                "perguntas sim/não",
                "negação simples com 'não'"
            ],
            "comprimento": "3-5 palavras por frase",
            "conectores": ["e", "ou", "mas (muito básico)"],
            "evitar": [
                "tempos compostos",
                "subjuntivo",
                "voz passiva",
                "orações subordinadas",
                "pronomes relativos"
            ]
        },
        CEFRLevel.A2: {
            "tempos_verbais": [
                "presente do indicativo",
                "pretérito perfeito (OBRIGATÓRIO usar!)",
                "futuro perifrástico (ir + infinitivo)",
                "imperativo"
            ],
            "pronomes": [
                "pessoais (todos)",
                "possessivos básicos (meu, sua, nosso)",
                "demonstrativos básicos (este, esse, aquele)"
            ],
            "estruturas": [
                "frases coordenadas com 'e', 'mas', 'ou', 'porque' (OBRIGATÓRIO usar 'porque'!)",
                "perguntas com quando/como/por quê",
                "expressões de tempo (ontem, hoje, amanhã, na semana passada)",
                "expressões de frequência (sempre, às vezes, nunca, geralmente)"
            ],
            "comprimento": "6-10 palavras por frase (MÍNIMO 6!)",
            "conectores": ["e", "mas", "ou", "porque (OBRIGATÓRIO)", "então", "também"],
            "evitar": [
                "subjuntivo",
                "voz passiva",
                "orações subordinadas complexas (apenas 'porque' e 'quando' são OK)",
                "expressões idiomáticas",
                "frases muito curtas (menos de 6 palavras)"
            ]
        },
        CEFRLevel.B1: {
            "tempos_verbais": [
                "presente, pretérito perfeito, pretérito imperfeito (OBRIGATÓRIO variar!)",
                "futuro simples e perifrástico",
                "presente do subjuntivo (apenas expressões fixas como 'seja')",
                "condicional simples (gostaria, poderia, seria)"
            ],
            "pronomes": [
                "pessoais, possessivos, demonstrativos",
                "relativos básicos (que, quem, onde)",
                "indefinidos (algum, nenhum, todo, cada)"
            ],
            "estruturas": [
                "orações subordinadas temporais (quando, enquanto, antes de, depois de)",
                "orações subordinadas causais (porque, já que, pois) - OBRIGATÓRIO!",
                "orações subordinadas condicionais (se, caso) - OBRIGATÓRIO!",
                "voz passiva simples (pode aparecer)",
                "discurso indireto básico (disse que, perguntou se)"
            ],
            "comprimento": "10-15 palavras por frase (MÍNIMO 10!)",
            "conectores": [
                "e, mas, ou, porque (OBRIGATÓRIO)",
                "quando, enquanto, antes de, depois de",
                "se, caso (OBRIGATÓRIO)",
                "também, além disso, por isso"
            ],
            "evitar": [
                "subjuntivo complexo (apenas 'seja' em expressões fixas é OK)",
                "orações relativas complexas com 'cujo'",
                "conectores muito sofisticados (conquanto, visto que)",
                "frases muito curtas (menos de 10 palavras)",
                "vocabulário técnico ou abstrato demais"
            ]
        },
        CEFRLevel.B2: {
            "tempos_verbais": [
                "todos os tempos do indicativo",
                "subjuntivo presente e imperfeito",
                "condicional simples e composto",
                "infinitivo pessoal e impessoal"
            ],
            "pronomes": [
                "todos os tipos de pronomes",
                "relativos (que, quem, onde, cujo)",
                "indefinidos e quantificadores"
            ],
            "estruturas": [
                "orações subordinadas de todos os tipos",
                "voz passiva (todas as formas)",
                "discurso indireto completo",
                "construções impessoais",
                "gerúndio e particípio"
            ],
            "comprimento": "10-20 palavras por frase",
            "conectores": [
                "todos os conectores básicos e intermediários",
                "no entanto, contudo, portanto, assim",
                "embora, apesar de, mesmo que"
            ],
            "evitar": [
                "estruturas muito formais ou literárias",
                "expressões idiomáticas muito regionais"
            ]
        },
        CEFRLevel.C1: {
            "tempos_verbais": [
                "todos os tempos verbais",
                "subjuntivo em todas as formas",
                "tempos compostos avançados",
                "infinitivo pessoal em contextos formais"
            ],
            "pronomes": [
                "todos os pronomes, incluindo usos avançados",
                "pronomes relativos com preposições",
                "uso estilístico de pronomes"
            ],
            "estruturas": [
                "estruturas complexas e formais",
                "orações subordinadas múltiplas",
                "inversão sintática",
                "construções literárias",
                "discurso indireto livre"
            ],
            "comprimento": "15-30 palavras por frase (varia conforme contexto)",
            "conectores": [
                "todos os conectores, incluindo formais",
                "não obstante, consoante, haja vista",
                "uso sofisticado de conectores"
            ],
            "evitar": [
                "apenas estruturas muito coloquiais demais",
                "gírias muito regionais sem contexto"
            ]
        },
        CEFRLevel.C2: {
            "tempos_verbais": [
                "domínio completo de todos os tempos verbais",
                "uso nativo de todos os modos",
                "variações regionais e estilísticas"
            ],
            "pronomes": [
                "uso nativo de todos os pronomes",
                "variações regionais e estilísticas",
                "uso criativo e expressivo"
            ],
            "estruturas": [
                "estruturas nativas e criativas",
                "variações estilísticas",
                "registros formais e informais",
                "uso de recursos literários"
            ],
            "comprimento": "varia conforme necessidade expressiva",
            "conectores": [
                "uso nativo e criativo de conectores",
                "variações estilísticas",
                "recursos retóricos"
            ],
            "evitar": [
                "nenhuma restrição - domínio nativo completo"
            ]
        }
    }
    
    CEFR_VOCABULARY_CONSTRAINTS = {
        CEFRLevel.A1: {
            "dominios": [
                "família (pai, mãe, irmão)",
                "números (1-100)",
                "cores básicas",
                "comida básica (pão, água, leite)",
                "saudações (olá, tchau, bom dia)",
                "partes do corpo básicas",
                "verbos de ação básicos (comer, beber, dormir, ir)"
            ],
            "evitar": [
                "expressões idiomáticas",
                "vocabulário técnico",
                "gírias",
                "vocabulário abstrato",
                "expressões formais"
            ]
        },
        CEFRLevel.A2: {
            "dominios": [
                "rotina diária",
                "compras básicas",
                "viagens simples",
                "tempo (clima, estações)",
                "atividades de lazer",
                "lugares (cidade, casa, trabalho)",
                "sentimentos básicos (feliz, triste, cansado)"
            ],
            "evitar": [
                "expressões idiomáticas complexas",
                "vocabulário técnico especializado",
                "gírias muito específicas",
                "vocabulário muito formal"
            ]
        },
        CEFRLevel.B1: {
            "dominios": [
                "trabalho e profissões",
                "educação",
                "saúde",
                "meio ambiente",
                "cultura e entretenimento",
                "tecnologia básica",
                "opiniões e argumentos simples"
            ],
            "evitar": [
                "vocabulário técnico muito especializado",
                "expressões idiomáticas muito regionais",
                "jargão profissional avançado"
            ]
        },
        CEFRLevel.B2: {
            "dominios": [
                "trabalho e carreira",
                "educação superior",
                "política e sociedade",
                "cultura e artes",
                "tecnologia",
                "mídia e comunicação",
                "opiniões e argumentos complexos"
            ],
            "evitar": [
                "vocabulário técnico muito especializado de áreas específicas",
                "expressões idiomáticas muito regionais sem contexto"
            ]
        },
        CEFRLevel.C1: {
            "dominios": [
                "todos os domínios, incluindo especializados",
                "acadêmico e profissional",
                "cultura e sociedade avançada",
                "expressões idiomáticas",
                "nuances e sutilezas"
            ],
            "evitar": [
                "apenas vocabulário muito técnico sem contexto adequado"
            ]
        },
        CEFRLevel.C2: {
            "dominios": [
                "domínio completo de todos os domínios",
                "vocabulário nativo em todos os contextos",
                "expressões idiomáticas e culturais",
                "variações regionais",
                "registros formais e informais"
            ],
            "evitar": [
                "nenhuma restrição - domínio nativo completo"
            ]
        }
    }
    
    def __init__(self, cefr_level: CEFRLevel, native_language: str = "en", cefr_details: Optional[dict] = None, interpretable_knowledge_state: Optional[dict] = None):
        self.cefr_level = cefr_level
        self.native_language = native_language
        self.cefr_details = cefr_details or {}
        self.interpretable_knowledge_state = interpretable_knowledge_state or {}
    
    def render(self) -> str:
        """Renderiza informações sobre o estado do estudante"""
        description = self.CEFR_DESCRIPTIONS.get(self.cefr_level, "")
        instruction = self.CEFR_INSTRUCTIONS.get(self.cefr_level, "")
        
        # Montar detalhes do progresso se disponíveis
        progress_info = ""
        if self.cefr_details:
            current_level_data = self.cefr_details.get(self.cefr_level.value, {})
            progress_info = f"Progresso no nível atual ({self.cefr_level.value}): {current_level_data.get('percentage', '0%')}"
            
            # Identificar skills fracas no nível atual para reforço
            weak_skills = []
            breakdown = current_level_data.get("skills_breakdown", [])
            for skill in breakdown:
                if skill.get("mastery", 0) < 0.5:
                    weak_skills.append(skill.get("skill_id"))
            
            if weak_skills:
                progress_info += f"\nDificuldades atuais: {', '.join(weak_skills)}"

        parts = [
            "[Student State]",
            f"Nível CEFR Global: {self.cefr_level.value}",
            f"Descrição: {description}",
            f"Língua nativa: {self.native_language}",
        ]
        
        if progress_info:
            parts.append(f"Status: {progress_info}")
            
        parts.extend([
            f"\n[Instruções de Nível]",
            instruction
        ])
        
        # Adicionar restrições gramaticais detalhadas
        grammar_constraints = self.CEFR_GRAMMAR_CONSTRAINTS.get(self.cefr_level, {})
        if grammar_constraints:
            parts.append(f"\n[Restrições Gramaticais - Nível {self.cefr_level.value}]")
            parts.append("IMPORTANTE: Use APENAS as estruturas gramaticais permitidas para este nível.")
            
            if "tempos_verbais" in grammar_constraints:
                parts.append(f"\nTempos verbais permitidos:")
                for tempo in grammar_constraints["tempos_verbais"]:
                    parts.append(f"  ✓ {tempo}")
            
            if "pronomes" in grammar_constraints:
                parts.append(f"\nPronomes permitidos:")
                for pronome in grammar_constraints["pronomes"]:
                    parts.append(f"  ✓ {pronome}")
            
            if "estruturas" in grammar_constraints:
                parts.append(f"\nEstruturas permitidas:")
                for estrutura in grammar_constraints["estruturas"]:
                    parts.append(f"  ✓ {estrutura}")
            
            if "conectores" in grammar_constraints:
                parts.append(f"\nConectores permitidos:")
                conectores_str = ", ".join(grammar_constraints["conectores"])
                parts.append(f"  ✓ {conectores_str}")
            
            if "comprimento" in grammar_constraints:
                parts.append(f"\nComprimento de frase: {grammar_constraints['comprimento']}")
            
            if "evitar" in grammar_constraints:
                parts.append(f"\nNÃO USE (fora do nível):")
                for item in grammar_constraints["evitar"]:
                    parts.append(f"  ✗ {item}")
        
        # Adicionar restrições vocabulares detalhadas
        vocab_constraints = self.CEFR_VOCABULARY_CONSTRAINTS.get(self.cefr_level, {})
        if vocab_constraints:
            parts.append(f"\n[Restrições Vocabulares - Nível {self.cefr_level.value}]")
            
            if "dominios" in vocab_constraints:
                parts.append("Domínios vocabulares apropriados:")
                for dominio in vocab_constraints["dominios"]:
                    parts.append(f"  ✓ {dominio}")
            
            if "evitar" in vocab_constraints:
                parts.append(f"\nNÃO USE (fora do nível):")
                for item in vocab_constraints["evitar"]:
                    parts.append(f"  ✗ {item}")
        
        # Adicionar informações do interpretable knowledge state se disponível
        if self.interpretable_knowledge_state:
            parts.append("\n[Análise Detalhada do Progresso]")
            
            # Progresso por dimensão
            dimension_progress = self.interpretable_knowledge_state.get("dimension_progress", {})
            if dimension_progress:
                parts.append("Progresso por dimensão:")
                for dim, progress in dimension_progress.items():
                    parts.append(f"  - {dim}: {progress:.0%}")
            
            # Skills fortes
            strong_skills = self.interpretable_knowledge_state.get("strong_skills", [])
            if strong_skills:
                parts.append("\nSkills fortes (domínio alto):")
                for skill in strong_skills[:3]:
                    parts.append(f"  - {skill.get('skill_name', skill.get('skill_id'))}: {skill.get('mastery', 0):.0%}")
            
            # Skills fracas
            weak_skills = self.interpretable_knowledge_state.get("weak_skills", [])
            if weak_skills:
                parts.append("\nSkills que precisam de prática:")
                for skill in weak_skills[:3]:
                    parts.append(f"  - {skill.get('skill_name', skill.get('skill_id'))}: {skill.get('mastery', 0):.0%}")
            
            # Recomendações
            recommendations = self.interpretable_knowledge_state.get("recommendations", [])
            if recommendations:
                parts.append("\nRecomendações pedagógicas:")
                for rec in recommendations[:2]:  # Top 2 recomendações
                    parts.append(f"  - {rec}")
        
        # Adicionar instrução final crítica baseada nos papers
        parts.append(f"""
[⚠️ INSTRUÇÃO CRÍTICA - BASEADA EM PAPERS ACADÊMICOS]

Você DEVE responder EXATAMENTE no nível CEFR {self.cefr_level.value}. 
Sua resposta será analisada automaticamente por um classificador CEFR baseado nos seguintes papers:
- Leal et al. (2022) - NILC-Metrix
- Vajjala & Rama (2021) - CEFR classification with RNNs
- Arnold et al. (2018) - CEFR prediction
- Ribeiro et al. (2024) - Complexidade textual em português

O classificador verificará:
1. Comprimento médio de frases (em palavras)
2. Presença/ausência de subordinação, subjuntivo, voz passiva, orações relativas
3. Vocabulário (frequência, diversidade, abstração)
4. Conectores e marcadores discursivos

🗣️ ESTILO DE CONVERSAÇÃO:
- Mantenha respostas CURTAS e CONVERSACIONAIS
- Níveis A1-B1: Máximo 2-4 frases por resposta
- Níveis B2-C1: Máximo 3-6 frases por resposta
- Nível C2: Máximo 5-8 frases por resposta
- SEMPRE faça perguntas para dar turno ao estudante falar
- NÃO faça monólogos longos - este é um DIÁLOGO

Se você usar estruturas gramaticais acima do nível {self.cefr_level.value}, sua resposta será rejeitada.
Se você usar vocabulário muito simples para {self.cefr_level.value}, sua resposta será rejeitada.
Se suas respostas forem muito longas e não derem espaço ao estudante, a conversa será ruim.

RESPONDA NATURALMENTE mas DENTRO DO NÍVEL {self.cefr_level.value} e MANTENHA O DIÁLOGO FLUINDO.
""")
        
        return "\n".join(parts)

