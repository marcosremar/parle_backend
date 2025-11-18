"""
Processador de Streaming com Pydantic
Corta texto em chunks por pontuação e dispara síntese incremental
"""

from pydantic import BaseModel, validator, root_validator
from typing import List, Optional, Callable, Any
import asyncio
import re
from enum import Enum


class ChunkType(str, Enum):
    """chunk type"""
    TEXT_PARTIAL = "text_partial"
    TEXT_SENTENCE = "text_sentence"  # Sentença completa
    AUDIO_CHUNK = "audio_chunk"
    SUGGESTION = "suggestion"
    COMPLETE = "complete"


class TextAccumulator(BaseModel):
    """
    Acumula texto e dispara ações quando detecta sentenças completas
    """
    buffer: str = ""
    sentences: List[str] = []
    on_sentence_callback: Optional[Callable] = None
    on_partial_callback: Optional[Callable] = None

    class Config:
        """Configuration settings for """
        arbitrary_types_allowed = True
        validate_assignment = True

    @validator('buffer', always=True)
    def process_text_chunks(cls, v, values) -> Any:
        """
        Detecta pontuação e dispara síntese incremental
        """
        if not v:
            return v

        # Callback para texto parcial (opcional)
        if values.get('on_partial_callback'):
            values['on_partial_callback'](v)

        # Detecta sentenças completas
        # Pontos de corte: . ! ? ; \n
        sentence_endings = r'[.!?;]\s|[\n]'

        parts = re.split(f'({sentence_endings})', v)
        complete_sentences = []
        remaining = ""

        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and re.match(sentence_endings, parts[i + 1]):
                # Sentença completa encontrada
                sentence = parts[i] + parts[i + 1].strip()
                complete_sentences.append(sentence)
                i += 2
            else:
                # Texto incompleto
                remaining += parts[i]
                i += 1

        # Dispara callback para cada sentença completa
        callback = values.get('on_sentence_callback')
        if callback and complete_sentences:
            for sentence in complete_sentences:
                # DISPARA SÍNTESE IMEDIATAMENTE
                asyncio.create_task(callback(sentence))

        # Atualiza lista de sentenças
        if 'sentences' in values:
            values['sentences'].extend(complete_sentences)

        return remaining  # Retorna só o que não foi processado


class SmartStreamingChunk(BaseModel):
    """
    Chunk inteligente que identifica tipo e dispara ação apropriada
    """
    type: ChunkType
    content: str
    metadata: dict = {}

    @root_validator
    def smart_dispatch(cls, values) -> Any:
        """
        Dispara ação baseada no tipo e conteúdo
        """
        chunk_type = values.get('type')
        content = values.get('content', '')

        if chunk_type == ChunkType.TEXT_PARTIAL:
            # Texto parcial - acumula
            return values

        elif chunk_type == ChunkType.TEXT_SENTENCE:
            # Sentença completa - dispara TTS
            asyncio.create_task(synthesize_audio(content))

        elif chunk_type == ChunkType.SUGGESTION:
            # Sugestão - adiciona à UI
            asyncio.create_task(update_suggestions(content))

        return values


class IncrementalProcessor(BaseModel):
    """
    Processador principal que corta e processa incrementalmente
    """
    text_accumulator: TextAccumulator
    audio_queue: List[bytes] = []
    suggestions: List[str] = []

    class Config:
        """Configuration settings for """
        arbitrary_types_allowed = True

    def process_llm_stream(self, text_chunk: str) -> Any:
        """
        Processa chunk de texto do LLM
        Corta em sentenças e dispara síntese
        """
        # Adiciona ao buffer
        self.text_accumulator.buffer += text_chunk

        # O validator do TextAccumulator vai:
        # 1. Detectar sentenças completas
        # 2. Disparar síntese para cada uma
        # 3. Manter apenas texto incompleto no buffer

        # Força revalidação
        self.text_accumulator = self.text_accumulator

    def add_audio_chunk(self, audio: bytes) -> Any:
        """
        Adiciona áudio sintetizado à fila
        """
        self.audio_queue.append(audio)
        # Dispara streaming de áudio
        asyncio.create_task(stream_audio(audio))

    def add_suggestion(self, suggestion: str) -> Any:
        """
        Adiciona e valida sugestão
        """
        # Remove marcadores
        clean = re.sub(r'^[\d\-\.\*\•]+\s*', '', suggestion.strip())
        if clean and len(clean) < 100:
            self.suggestions.append(clean)
            # Dispara atualização UI
            asyncio.create_task(update_ui_suggestion(clean))


# Funções auxiliares (seriam implementadas no sistema real)
async def synthesize_audio(text: str) -> Any:
    """Sintetiza áudio para sentença"""
    print(f"🔊 Sintetizando: {text[:50]}...")

async def stream_audio(audio: bytes) -> Any:
    """Faz streaming de áudio"""
    print(f"📡 Streaming {len(audio)} bytes de áudio")

async def update_suggestions(suggestion: str) -> Any:
    """Atualiza sugestões na UI"""
    print(f"💡 Nova sugestão: {suggestion}")

async def update_ui_suggestion(suggestion: str) -> Any:
    """Atualiza UI com sugestão"""
    print(f"🔄 UI atualizada com: {suggestion}")


# Exemplo de uso prático
def create_processor() -> IncrementalProcessor:
    """
    Cria processador com callbacks configurados
    """

    async def on_sentence_ready(sentence: str) -> Any:
        """Callback quando sentença completa é detectada"""
        print(f"✅ Sentença pronta: {sentence}")
        # Inicia síntese de áudio imediatamente
        await synthesize_audio(sentence)

    async def on_partial_text(partial: str) -> Any:
        """Callback para texto parcial (opcional)"""
        print(f"📝 Parcial: {partial[-20:]}...")

    accumulator = TextAccumulator(
        on_sentence_callback=on_sentence_ready,
        on_partial_callback=on_partial_text
    )

    return IncrementalProcessor(text_accumulator=accumulator)


# Exemplo de fluxo completo
async def example_streaming_flow() -> Any:
    """
    Simula fluxo de streaming com corte inteligente
    """
    processor = create_processor()

    # Simula texto chegando em chunks do LLM
    text_chunks = [
        "A capital do Brasil é ",
        "Brasília. ",  # <- Dispara síntese aqui
        "Foi fundada em 1960 por ",
        "Juscelino Kubitschek. ",  # <- Dispara síntese aqui
        "A cidade foi plane",
        "jada por Lúcio Costa e ",
        "Oscar Niemeyer."  # <- Dispara síntese aqui
    ]

    for chunk in text_chunks:
        processor.process_llm_stream(chunk)
        await asyncio.sleep(0.1)  # Simula delay

    # Finaliza qualquer texto pendente
    if processor.text_accumulator.buffer:
        print(f"📌 Texto final: {processor.text_accumulator.buffer}")
        await synthesize_audio(processor.text_accumulator.buffer)


if __name__ == "__main__":
    # Testa o fluxo
    asyncio.run(example_streaming_flow())