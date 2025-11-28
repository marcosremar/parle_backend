"""
Dependency Parser using SpaCy
Provides Portuguese language parsing with dependency relations, POS tags, and syntactic trees
"""

import spacy
from typing import List, Dict, Any, Optional
from loguru import logger
import asyncio
from functools import lru_cache


class DependencyParser:
    """SpaCy-based dependency parser for Portuguese"""
    
    def __init__(self, model_name: str = "pt_core_news_lg"):
        """
        Initialize the parser with Portuguese model
        
        Args:
            model_name: SpaCy model name (default: pt_core_news_lg)
        """
        self.model_name = model_name
        self.nlp: Optional[spacy.Language] = None
        self._load_lock = asyncio.Lock()
        self._loaded = False
    
    async def _ensure_loaded(self):
        """Ensure the SpaCy model is loaded (thread-safe)"""
        if self._loaded and self.nlp is not None:
            return
        
        async with self._load_lock:
            if self._loaded and self.nlp is not None:
                return
            
            try:
                logger.info(f"Loading SpaCy model: {self.model_name}")
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                self.nlp = await loop.run_in_executor(
                    None,
                    lambda: spacy.load(self.model_name)
                )
                self._loaded = True
                logger.info(f"✅ SpaCy model {self.model_name} loaded successfully")
            except OSError as e:
                logger.error(f"❌ Failed to load SpaCy model {self.model_name}: {e}")
                logger.info("💡 Install with: python -m spacy download pt_core_news_lg")
                raise
    
    async def parse_text(self, text: str) -> Dict[str, Any]:
        """
        Parse text and extract linguistic features
        
        Args:
            text: Input text to parse
            
        Returns:
            Dictionary with parsed features:
            - sentences: List of sentence objects
            - tokens: List of token objects with POS, dependency info
            - dependency_tree: Dependency relations
            - pos_tags: Part-of-speech tags
        """
        await self._ensure_loaded()
        
        if not self.nlp:
            raise RuntimeError("SpaCy model not loaded")
        
        # Run parsing in executor to avoid blocking
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(None, self.nlp, text)
        
        # Extract sentences
        sentences = []
        for sent in doc.sents:
            sentence_data = {
                "text": sent.text,
                "start": sent.start_char,
                "end": sent.end_char,
                "tokens": []
            }
            
            # Extract tokens with dependency info
            for token in sent:
                token_data = {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                    "head": token.head.text if token.head else None,
                    "head_pos": token.head.pos_ if token.head else None,
                    "children": [child.text for child in token.children],
                    "is_punct": token.is_punct,
                    "is_space": token.is_space,
                    "is_stop": token.is_stop,
                    "is_alpha": token.is_alpha,
                }
                sentence_data["tokens"].append(token_data)
            
            sentences.append(sentence_data)
        
        # Extract dependency relations
        dependency_relations = []
        for token in doc:
            if not token.is_punct and not token.is_space:
                dependency_relations.append({
                    "token": token.text,
                    "head": token.head.text if token.head else "ROOT",
                    "relation": token.dep_,
                    "pos": token.pos_,
                    "head_pos": token.head.pos_ if token.head else None
                })
        
        # Extract POS tags
        pos_tags = [{"token": token.text, "pos": token.pos_, "tag": token.tag_} 
                   for token in doc if not token.is_space]
        
        return {
            "text": text,
            "sentences": sentences,
            "dependency_relations": dependency_relations,
            "pos_tags": pos_tags,
            "num_sentences": len(list(doc.sents)),
            "num_tokens": len([t for t in doc if not t.is_space]),
            "num_words": len([t for t in doc if t.is_alpha])
        }
    
    async def extract_dependency_tree(self, text: str) -> Dict[str, Any]:
        """
        Extract dependency tree structure for syntactic analysis
        
        Args:
            text: Input text
            
        Returns:
            Tree structure with depth information
        """
        await self._ensure_loaded()
        
        if not self.nlp:
            raise RuntimeError("SpaCy model not loaded")
        
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(None, self.nlp, text)
        
        trees = []
        for sent in doc.sents:
            # Find root token
            root = None
            for token in sent:
                if token.dep_ == "ROOT":
                    root = token
                    break
            
            if root:
                tree = self._build_tree(root)
                trees.append(tree)
        
        return {
            "trees": trees,
            "num_sentences": len(trees)
        }
    
    def _build_tree(self, token) -> Dict[str, Any]:
        """Build tree structure from token"""
        node = {
            "text": token.text,
            "pos": token.pos_,
            "dep": token.dep_,
            "children": []
        }
        
        for child in token.children:
            child_tree = self._build_tree(child)
            node["children"].append(child_tree)
        
        return node
    
    def get_sentence_boundaries(self, text: str) -> List[Dict[str, int]]:
        """
        Get sentence boundaries (synchronous, for quick operations)
        
        Args:
            text: Input text
            
        Returns:
            List of {start, end} character positions
        """
        if not self.nlp:
            # Try to load synchronously if not loaded
            try:
                self.nlp = spacy.load(self.model_name)
                self._loaded = True
            except:
                return []
        
        doc = self.nlp(text)
        boundaries = []
        for sent in doc.sents:
            boundaries.append({
                "start": sent.start_char,
                "end": sent.end_char
            })
        return boundaries

