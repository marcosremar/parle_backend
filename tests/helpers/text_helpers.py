"""
Helper functions for text testing
"""

from typing import List, Dict, Optional


def create_test_phrases(language: str = "pt") -> List[str]:
    """
    Create test phrases for different languages
    
    Args:
        language: Language code ("pt" or "en")
        
    Returns:
        List of test phrases
    """
    if language == "pt":
        return [
            "Olá, como você está?",
            "Eu gosto de programação.",
            "O tempo está bom hoje.",
            "Vou ao supermercado comprar comida.",
            "Este é um teste de transcrição."
        ]
    else:  # en
        return [
            "Hello, how are you?",
            "I like programming.",
            "The weather is nice today.",
            "I'm going to the supermarket to buy food.",
            "This is a transcription test."
        ]


def create_complex_phrases(language: str = "pt") -> List[str]:
    """
    Create complex test phrases
    
    Args:
        language: Language code ("pt" or "en")
        
    Returns:
        List of complex phrases
    """
    if language == "pt":
        return [
            "A programação de computadores é uma arte que combina lógica, criatividade e conhecimento técnico.",
            "O sistema de inteligência artificial utiliza algoritmos avançados para processar informações complexas.",
            "A análise de dados requer conhecimento estatístico e habilidades de programação em múltiplas linguagens."
        ]
    else:  # en
        return [
            "Computer programming is an art that combines logic, creativity, and technical knowledge.",
            "The artificial intelligence system uses advanced algorithms to process complex information.",
            "Data analysis requires statistical knowledge and programming skills in multiple languages."
        ]


def create_text_with_numbers(language: str = "pt") -> str:
    """
    Create text with numbers
    
    Args:
        language: Language code ("pt" or "en")
        
    Returns:
        Text with numbers
    """
    if language == "pt":
        return "Eu tenho 25 anos e moro na rua número 123. Meu telefone é 98765-4321."
    else:  # en
        return "I am 25 years old and live at number 123. My phone is 98765-4321."


def create_text_with_proper_nouns(language: str = "pt") -> str:
    """
    Create text with proper nouns
    
    Args:
        language: Language code ("pt" or "en")
        
    Returns:
        Text with proper nouns
    """
    if language == "pt":
        return "João Silva mora em São Paulo e trabalha na empresa Microsoft."
    else:  # en
        return "John Smith lives in New York and works at Microsoft company."


def calculate_word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text (transcription)
        
    Returns:
        WER as a float (0.0 = perfect, 1.0 = all words wrong)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # Simple WER calculation (Levenshtein distance on words)
    # For a more accurate implementation, use python-Levenshtein library
    from difflib import SequenceMatcher
    
    matcher = SequenceMatcher(None, ref_words, hyp_words)
    similarity = matcher.ratio()
    
    return 1.0 - similarity


def validate_text_length(text: str, min_length: int = 1, max_length: Optional[int] = None) -> bool:
    """
    Validate text length
    
    Args:
        text: Text to validate
        min_length: Minimum length
        max_length: Maximum length (None = no limit)
        
    Returns:
        True if valid
    """
    length = len(text)
    if length < min_length:
        return False
    if max_length is not None and length > max_length:
        return False
    return True


def extract_keywords(text: str, language: str = "pt") -> List[str]:
    """
    Extract keywords from text (simple implementation)
    
    Args:
        text: Text to extract keywords from
        language: Language code
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction (remove stop words)
    stop_words_pt = {"o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das", 
                     "em", "no", "na", "nos", "nas", "para", "com", "por", "é", "são"}
    stop_words_en = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
                     "of", "with", "by", "is", "are", "was", "were"}
    
    stop_words = stop_words_pt if language == "pt" else stop_words_en
    
    words = text.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    return keywords
