"""
Processador de Prompt Formato 4 com Streaming e Pydantic
Implementa corte inteligente por pontuação e validação
"""

from pydantic import ValidationError, BaseModel, field_validator, Field
from typing import List, Optional, Tuple, Dict, Any
import re
import asyncio
import time
from dataclasses import dataclass
from src.core.exceptions import UltravoxError, wrap_exception


class StreamingPrompt:
    """Formato 4 - Prompt otimizado para streaming com corte por pontuação"""

    @staticmethod
    def create_prompt(user_question: str) -> str:
        return f"""INSTRUÇÕES IMPORTANTES:
1. Termine cada frase com ponto final.
2. Use frases curtas e diretas.
3. Após a resposta, adicione marcador "###SUGESTOES###"
4. Liste 3 sugestões, uma por linha.

Exemplo de formato:
A capital é Brasília. Foi fundada em 1960. Tem 3 milhões de habitantes.
###SUGESTOES###
Qual a população exata?
Quem foi o arquiteto?
Como é o clima?

Pergunta: {user_question}
Resposta:"""


@dataclass
class ProcessedSentence:
    """Sentença processada pronta para TTS"""
    text: str
    index: int
    timestamp: float


class LLMStreamingResponse(BaseModel):
    """Parser e validador da resposta do LLM com suporte a streaming"""

    raw_text: str = Field(..., min_length=1)
    sentences: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    response_text: str = ""
    has_suggestions_marker: bool = False

    @field_validator('raw_text')
    @classmethod
    def parse_streaming_format(cls, v: str) -> str:
        """Valida e processa o formato streaming"""
        if not v or not v.strip():
            raise ValueError("Resposta vazia do LLM")
        return v.strip()

    def process(self) -> Dict[str, Any]:
        """Processa texto completo separando resposta e sugestões"""
        parts = self.raw_text.split("###SUGESTOES###")

        # Processa resposta principal
        response_part = parts[0].strip()
        self.response_text = response_part

        # Divide em sentenças para streaming
        # Regex melhorado para detectar fim de sentença
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        self.sentences = [s.strip() for s in re.split(sentence_pattern, response_part) if s.strip()]

        # Processa sugestões se existirem
        if len(parts) > 1:
            self.has_suggestions_marker = True
            suggestions_text = parts[1].strip()

            # Extrai sugestões (remove numeração se houver)
            lines = suggestions_text.split('\n')
            for line in lines[:3]:  # Máximo 3 sugestões
                clean_line = re.sub(r'^\d+[\.\-\)]\s*', '', line.strip())
                clean_line = clean_line.strip('- •*')
                if clean_line:
                    self.suggestions.append(clean_line)

        return {
            'sentences': self.sentences,
            'suggestions': self.suggestions,
            'full_response': self.response_text,
            'total_sentences': len(self.sentences),
            'total_suggestions': len(self.suggestions)
        }


class IncrementalStreamProcessor(BaseModel):
    """
    Processador incremental que detecta sentenças completas
    e dispara síntese assim que detecta pontuação
    """

    buffer: str = Field(default="")
    processed_sentences: List[ProcessedSentence] = Field(default_factory=list)
    pending_suggestions: List[str] = Field(default_factory=list)
    is_processing_suggestions: bool = Field(default=False)

    class Config:
        arbitrary_types_allowed = True

    def add_chunk(self, chunk: str) -> List[ProcessedSentence]:
        """
        Adiciona chunk ao buffer e retorna sentenças completas encontradas
        """
        new_sentences = []

        # Se encontrou marcador de sugestões
        if "###SUGESTOES###" in chunk:
            self.is_processing_suggestions = True
            parts = chunk.split("###SUGESTOES###")

            # Processa parte antes do marcador
            if parts[0]:
                self.buffer += parts[0]
                new_sentences.extend(self._extract_sentences())

            # Começa a processar sugestões
            if len(parts) > 1:
                self._process_suggestions_chunk(parts[1])

            return new_sentences

        # Processamento normal de texto
        if not self.is_processing_suggestions:
            self.buffer += chunk
            new_sentences = self._extract_sentences()
        else:
            # Processando sugestões
            self._process_suggestions_chunk(chunk)

        return new_sentences

    def _extract_sentences(self) -> List[ProcessedSentence]:
        """Extrai sentenças completas do buffer"""
        sentences = []

        # Padrão para detectar fim de sentença
        # Melhorado para lidar com abreviações comuns
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'

        # Divide o buffer
        parts = re.split(pattern, self.buffer)

        # Processa todas exceto a última (pode estar incompleta)
        for i in range(len(parts) - 1):
            if parts[i].strip():
                sentence = ProcessedSentence(
                    text=parts[i].strip(),
                    index=len(self.processed_sentences),
                    timestamp=time.time()
                )
                self.processed_sentences.append(sentence)
                sentences.append(sentence)

        # Mantém apenas a última parte no buffer (potencialmente incompleta)
        self.buffer = parts[-1] if parts else ""

        # Se o buffer termina com pontuação final, é uma sentença completa
        if self.buffer and re.search(r'[.!?]\s*$', self.buffer):
            sentence = ProcessedSentence(
                text=self.buffer.strip(),
                index=len(self.processed_sentences),
                timestamp=time.time()
            )
            self.processed_sentences.append(sentence)
            sentences.append(sentence)
            self.buffer = ""

        return sentences

    def _process_suggestions_chunk(self, chunk: str):
        """Processa chunk de sugestões"""
        lines = chunk.strip().split('\n')
        for line in lines:
            clean_line = re.sub(r'^\d+[\.\-\)]\s*', '', line.strip())
            clean_line = clean_line.strip('- •*')
            if clean_line and len(self.pending_suggestions) < 3:
                self.pending_suggestions.append(clean_line)

    def finalize(self) -> Tuple[Optional[ProcessedSentence], List[str]]:
        """Finaliza processamento retornando texto pendente e sugestões"""
        final_sentence = None

        # Se ainda há texto no buffer, processa como sentença final
        if self.buffer.strip():
            final_sentence = ProcessedSentence(
                text=self.buffer.strip(),
                index=len(self.processed_sentences),
                timestamp=time.time()
            )
            self.processed_sentences.append(final_sentence)
            self.buffer = ""

        return final_sentence, self.pending_suggestions


# Funções auxiliares para demonstração

async def simulate_tts(sentence: ProcessedSentence):
    """Simula síntese de áudio"""
    print(f"  🔊 Sintetizando [{sentence.index}]: {sentence.text}")
    await asyncio.sleep(0.2)  # Simula tempo de síntese
    return f"audio_{sentence.index}.wav"


async def simulate_llm_streaming(prompt: str) -> List[str]:
    """Simula LLM retornando resposta em chunks"""
    # Simula resposta completa
    full_response = """A capital do Brasil é Brasília. Foi inaugurada em 21 de abril de 1960.
A cidade foi planejada por Lúcio Costa e Oscar Niemeyer.
###SUGESTOES###
Qual é a população atual de Brasília?
Por que mudaram a capital do Rio de Janeiro?
Quais são os principais monumentos da cidade?"""

    # Divide em chunks para simular streaming
    chunks = []
    words = full_response.split()
    chunk_size = 3  # Palavras por chunk

    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        if i + chunk_size < len(words):
            chunk += ' '
        chunks.append(chunk)

    return chunks


async def process_with_streaming_demo():
    """Demonstração do processamento com streaming"""
    print("=" * 70)
    print("🚀 DEMONSTRAÇÃO: Processamento Streaming com Formato 4")
    print("=" * 70)

    # 1. Cria prompt
    user_question = "Qual é a capital do Brasil?"
    prompt = StreamingPrompt.create_prompt(user_question)

    print(f"\n📝 Pergunta: {user_question}")
    print("-" * 70)

    # 2. Simula resposta em streaming do LLM
    chunks = await simulate_llm_streaming(prompt)

    # 3. Processa chunks incrementalmente
    processor = IncrementalStreamProcessor()
    all_audio_tasks = []

    print("\n⚡ Processamento em Streaming:")
    print("-" * 70)

    for i, chunk in enumerate(chunks):
        print(f"\n📨 Chunk {i+1}: '{chunk}'")

        # Adiciona chunk e obtém sentenças completas
        sentences = processor.add_chunk(chunk)

        # Dispara síntese para cada sentença detectada
        for sentence in sentences:
            print(f"  ✅ Sentença completa detectada!")
            # Inicia síntese sem bloquear
            audio_task = asyncio.create_task(simulate_tts(sentence))
            all_audio_tasks.append(audio_task)

        # Simula delay entre chunks
        await asyncio.sleep(0.1)

    # 4. Finaliza processamento
    print("\n🏁 Finalizando processamento...")
    final_sentence, suggestions = processor.finalize()

    if final_sentence:
        print(f"  ✅ Sentença final: {final_sentence.text}")
        audio_task = asyncio.create_task(simulate_tts(final_sentence))
        all_audio_tasks.append(audio_task)

    # 5. Exibe sugestões
    if suggestions:
        print(f"\n💡 Sugestões detectadas:")
        for i, sugg in enumerate(suggestions, 1):
            print(f"  {i}. {sugg}")

    # 6. Aguarda todas as sínteses
    print("\n⏳ Aguardando sínteses de áudio...")
    audio_files = await asyncio.gather(*all_audio_tasks)

    print(f"\n✅ Total de áudios gerados: {len(audio_files)}")

    # 7. Resumo
    print("\n" + "=" * 70)
    print("📊 RESUMO DO PROCESSAMENTO:")
    print("=" * 70)
    print(f"  • Sentenças processadas: {len(processor.processed_sentences)}")
    print(f"  • Sugestões extraídas: {len(suggestions)}")
    print(f"  • Áudios sintetizados: {len(audio_files)}")

    # Mostra timeline
    if processor.processed_sentences:
        first_time = processor.processed_sentences[0].timestamp
        print(f"\n⏱️ Timeline de processamento:")
        for sentence in processor.processed_sentences:
            delay = (sentence.timestamp - first_time) * 1000
            print(f"  {delay:6.0f}ms: {sentence.text[:40]}...")


def test_parser():
    """Testa o parser com diferentes formatos"""
    print("\n" + "=" * 70)
    print("🧪 TESTE: Parser de Formato 4")
    print("=" * 70)

    test_cases = [
        # Caso 1: Formato correto
        """A capital é Brasília. Foi fundada em 1960. Tem 3 milhões de habitantes.
###SUGESTOES###
Qual a população exata?
Quem foi o arquiteto?
Como é o clima?""",

        # Caso 2: Sem sugestões
        "A capital do Brasil é Brasília. Foi inaugurada em 1960.",

        # Caso 3: Sugestões numeradas
        """Brasília é a capital. É uma cidade planejada.
###SUGESTOES###
1. Quando foi construída?
2. Quem foi o presidente?
3. Qual o clima?""",
    ]

    for i, test_text in enumerate(test_cases, 1):
        print(f"\n📝 Teste {i}:")
        print("-" * 40)

        try:
            response = LLMStreamingResponse(raw_text=test_text)
            result = response.process()

            print(f"✅ Sentenças: {result['total_sentences']}")
            for j, sentence in enumerate(result['sentences'], 1):
                print(f"   {j}. {sentence}")

            print(f"✅ Sugestões: {result['total_suggestions']}")
            for j, sugg in enumerate(result['suggestions'], 1):
                print(f"   {j}. {sugg}")

        except Exception as e:
            print(f"❌ Erro: {e}")


if __name__ == "__main__":
    # Testa parser
    test_parser()

    # Testa streaming
    print("\n")
    asyncio.run(process_with_streaming_demo())