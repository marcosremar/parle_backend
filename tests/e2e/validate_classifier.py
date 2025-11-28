"""
Validation Dataset and Classifier Accuracy Measurement
Creates validation dataset and measures classifier accuracy across all CEFR levels
"""

import json
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from collections import defaultdict

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import classifiers
from tests.e2e.cefr_level_analyzer import (
    identify_cefr_level_llm,
    identify_cefr_level_hybrid,
    analyze_text_features_quantitative
)


# Validation dataset (manually labeled Portuguese texts)
# In production, this would be loaded from a file or database
VALIDATION_DATASET = [
    {
        "text": "Eu quero água. Onde fica banheiro?",
        "true_level": "A1",
        "source": "manual"
    },
    {
        "text": "Ontem eu fui no mercado. Comprei pão e leite.",
        "true_level": "A2",
        "source": "manual"
    },
    {
        "text": "Eu já visitei o Brasil duas vezes e gostei muito da cultura brasileira.",
        "true_level": "B1",
        "source": "manual"
    },
    {
        "text": "Embora eu tenha estudado português por três anos, ainda acho difícil entender quando as pessoas falam muito rápido.",
        "true_level": "B2",
        "source": "manual"
    },
    {
        "text": "Considerando que já li diversos autores brasileiros, acredito que Machado de Assis seja aquele cuja obra mais me impactou profundamente.",
        "true_level": "C1",
        "source": "manual"
    },
    {
        "text": "Depreende-se da sua colocação que a questão transcende mera análise superficial, entrelaçando-se em complexidades que merecem atenção meticulosa.",
        "true_level": "C2",
        "source": "manual"
    },
    # Add more examples for each level
    {
        "text": "Olá! Meu nome é João. Eu sou estudante.",
        "true_level": "A1",
        "source": "manual"
    },
    {
        "text": "Eu gosto de café. Você gosta também?",
        "true_level": "A1",
        "source": "manual"
    },
    {
        "text": "Eu trabalho em um escritório. Meu trabalho é interessante.",
        "true_level": "A2",
        "source": "manual"
    },
    {
        "text": "No fim de semana, eu gosto de ir ao cinema com meus amigos.",
        "true_level": "A2",
        "source": "manual"
    },
    {
        "text": "Se eu tivesse mais tempo, viajaria para a Europa no próximo verão.",
        "true_level": "B1",
        "source": "manual"
    },
    {
        "text": "Apesar das dificuldades que enfrentei, consegui aprender português com sucesso.",
        "true_level": "B2",
        "source": "manual"
    },
    {
        "text": "Não obstante as adversidades encontradas ao longo do processo, logrei alcançar os objetivos propostos.",
        "true_level": "C1",
        "source": "manual"
    },
    {
        "text": "A complexidade inerente ao tema em questão demanda uma abordagem multidisciplinar que contemple as nuances culturais e históricas envolvidas.",
        "true_level": "C2",
        "source": "manual"
    }
]


async def validate_classifier(
    dataset: List[Dict[str, Any]],
    session: aiohttp.ClientSession,
    use_hybrid: bool = True
) -> Dict[str, Any]:
    """
    Validate classifier accuracy on labeled dataset
    
    Args:
        dataset: List of labeled texts
        session: HTTP session
        use_hybrid: If True, use hybrid classifier; if False, use LLM only
        
    Returns:
        Validation results with accuracy metrics
    """
    results = []
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    print(f"\n{'='*80}")
    print(f"🔍 VALIDAÇÃO DO CLASSIFICADOR CEFR")
    print(f"{'='*80}\n")
    print(f"Dataset: {len(dataset)} textos")
    print(f"Classificador: {'Híbrido (LLM + Métricas)' if use_hybrid else 'LLM apenas'}\n")
    
    # Process in parallel batches for speed
    batch_size = 4  # Process 4 texts simultaneously
    print(f"🚀 Processando em paralelo (batch size: {batch_size})...\n")
    
    async def process_item(item, index):
        """Process a single item and return result"""
        text = item["text"]
        true_level = item["true_level"]
        user_id = item.get("user_id", None)  # Get user_id if available (for AKT validation)
        
        try:
            if use_hybrid:
                analysis = await identify_cefr_level_hybrid(text, session, expected_level=true_level, user_id=user_id)
            else:
                analysis = await identify_cefr_level_llm(text, session, expected_level=true_level)
            
            predicted_level = analysis.get("identified_level", "UNKNOWN")
            confidence = analysis.get("confidence", 0.0)
            is_correct = predicted_level == true_level
            
            return {
                "index": index,
                "text": text,
                "true_level": true_level,
                "predicted_level": predicted_level,
                "confidence": confidence,
                "is_correct": is_correct,
                "analysis": analysis,
                "error": None
            }
        except Exception as e:
            return {
                "index": index,
                "text": text,
                "true_level": true_level,
                "predicted_level": "ERROR",
                "confidence": 0.0,
                "is_correct": False,
                "analysis": {},
                "error": str(e)
            }
    
    # Process in batches
    all_results = []
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))
        
        print(f"📦 Processando batch {batch_start//batch_size + 1} (textos {batch_start+1}-{batch_end}/{len(dataset)})...")
        
        # Process batch in parallel
        tasks = [process_item(item, idx) for item, idx in zip(batch, batch_indices)]
        batch_results = await asyncio.gather(*tasks)
        
        # Process results
        for result in batch_results:
            idx = result["index"]
            text = result["text"]
            true_level = result["true_level"]
            predicted_level = result["predicted_level"]
            confidence = result["confidence"]
            is_correct = result["is_correct"]
            error = result["error"]
            
            results.append({
                "text": text,
                "true_level": true_level,
                "predicted_level": predicted_level,
                "confidence": confidence,
                "is_correct": is_correct,
                "analysis": result["analysis"]
            })
            
            if not error:
                confusion_matrix[true_level][predicted_level] += 1
                status = "✅" if is_correct else "❌"
                print(f"   [{idx+1}/{len(dataset)}] {status} {true_level} → {predicted_level} ({confidence:.0%}) | {text[:40]}...")
            else:
                print(f"   [{idx+1}/{len(dataset)}] ❌ Erro: {error}")
        
        all_results.extend(batch_results)
        print()
    
    # Calculate metrics
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total if total > 0 else 0.0
    
    # Per-level metrics
    level_stats = defaultdict(lambda: {"correct": 0, "total": 0, "precision": 0.0, "recall": 0.0})
    
    for result in results:
        true_level = result["true_level"]
        level_stats[true_level]["total"] += 1
        if result["is_correct"]:
            level_stats[true_level]["correct"] += 1
    
    # Calculate precision and recall
    for level in level_stats:
        stats = level_stats[level]
        stats["recall"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        
        # Precision: true positives / (true positives + false positives)
        true_positives = stats["correct"]
        false_positives = sum(1 for r in results 
                            if r["predicted_level"] == level and r["true_level"] != level)
        stats["precision"] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    # F1 score
    for level in level_stats:
        stats = level_stats[level]
        precision = stats["precision"]
        recall = stats["recall"]
        stats["f1"] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"📊 RESULTADOS DA VALIDAÇÃO")
    print(f"{'='*80}\n")
    print(f"Precisão Geral: {accuracy:.1%} ({correct}/{total})")
    print(f"\nMétricas por Nível:\n")
    print(f"{'Nível':<6} {'Precisão':<12} {'Recall':<12} {'F1':<12} {'Amostras':<10}")
    print(f"{'-'*60}")
    
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        print(f"{level:<6} {stats['precision']:>10.1%} {stats['recall']:>10.1%} {stats['f1']:>10.1%} {stats['total']:>10}")
    
    print(f"\n{'='*80}")
    print(f"Matriz de Confusão:")
    print(f"{'='*80}\n")
    print(f"{'Verdadeiro \\ Predito':<20}", end="")
    all_levels = sorted(set(r["true_level"] for r in results) | set(r["predicted_level"] for r in results))
    for level in all_levels:
        print(f"{level:>6}", end="")
    print()
    print("-" * (20 + 6 * len(all_levels)))
    
    for true_level in sorted(confusion_matrix.keys()):
        print(f"{true_level:<20}", end="")
        for pred_level in all_levels:
            count = confusion_matrix[true_level].get(pred_level, 0)
            print(f"{count:>6}", end="")
        print()
    
    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "level_stats": dict(level_stats),
        "confusion_matrix": dict(confusion_matrix),
        "results": results,
        "classifier": "hybrid" if use_hybrid else "llm_only"
    }


async def run_validation():
    """Run validation and save results"""
    async with aiohttp.ClientSession() as session:
        # Test hybrid classifier
        hybrid_results = await validate_classifier(VALIDATION_DATASET, session, use_hybrid=True)
        
        # Test LLM-only classifier
        llm_results = await validate_classifier(VALIDATION_DATASET, session, use_hybrid=False)
        
        # Save results
        output_dir = Path(__file__).parent / "reports" / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save hybrid results
        hybrid_path = output_dir / f"validation_hybrid_{timestamp}.json"
        with open(hybrid_path, "w", encoding="utf-8") as f:
            json.dump(hybrid_results, f, indent=2, ensure_ascii=False)
        
        # Save LLM-only results
        llm_path = output_dir / f"validation_llm_only_{timestamp}.json"
        with open(llm_path, "w", encoding="utf-8") as f:
            json.dump(llm_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Resultados salvos:")
        print(f"   - Híbrido: {hybrid_path.name}")
        print(f"   - LLM apenas: {llm_path.name}")
        
        # Compare
        print(f"\n{'='*80}")
        print(f"📈 COMPARAÇÃO DE CLASSIFICADORES")
        print(f"{'='*80}\n")
        print(f"Híbrido (LLM + Métricas): {hybrid_results['accuracy']:.1%}")
        print(f"LLM apenas:                {llm_results['accuracy']:.1%}")
        improvement = hybrid_results['accuracy'] - llm_results['accuracy']
        print(f"Melhoria:                   {improvement:+.1%}")
        
        return hybrid_results, llm_results


if __name__ == "__main__":
    asyncio.run(run_validation())

