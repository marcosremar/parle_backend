"""
Syntactic Complexity Metrics
Implements Yngve depth, Frazier depth, T-units, and other syntactic complexity measures
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from .parser import DependencyParser


class SyntacticMetricsCalculator:
    """Calculate syntactic complexity metrics"""
    
    def __init__(self, parser: DependencyParser):
        """
        Initialize with a dependency parser
        
        Args:
            parser: DependencyParser instance
        """
        self.parser = parser
    
    async def calculate_yngve_depth(self, text: str) -> Dict[str, Any]:
        """
        Calculate Yngve depth (left-branching complexity)
        
        Yngve depth measures syntactic complexity by counting left-branching
        nodes in the dependency tree. Higher values indicate more complex structures.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with Yngve depth metrics
        """
        parsed = await self.parser.parse_text(text)
        
        yngve_scores = []
        total_depth = 0
        
        for sentence in parsed["sentences"]:
            # Reconstruct tree from dependency relations
            tree = self._build_dependency_tree(sentence["tokens"])
            if tree:
                depth = self._calculate_yngve_for_tree(tree)
                yngve_scores.append(depth)
                total_depth += depth
        
        avg_yngve = total_depth / len(yngve_scores) if yngve_scores else 0.0
        max_yngve = max(yngve_scores) if yngve_scores else 0.0
        
        return {
            "mean_yngve_depth": avg_yngve,
            "max_yngve_depth": max_yngve,
            "sentence_yngve_depths": yngve_scores,
            "total_sentences": len(yngve_scores)
        }
    
    async def calculate_frazier_depth(self, text: str) -> Dict[str, Any]:
        """
        Calculate Frazier depth (right-branching complexity)
        
        Frazier depth measures complexity by counting right-branching nodes.
        Alternative to Yngve depth, better for languages with different branching patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with Frazier depth metrics
        """
        parsed = await self.parser.parse_text(text)
        
        frazier_scores = []
        total_depth = 0
        
        for sentence in parsed["sentences"]:
            tree = self._build_dependency_tree(sentence["tokens"])
            if tree:
                depth = self._calculate_frazier_for_tree(tree)
                frazier_scores.append(depth)
                total_depth += depth
        
        avg_frazier = total_depth / len(frazier_scores) if frazier_scores else 0.0
        max_frazier = max(frazier_scores) if frazier_scores else 0.0
        
        return {
            "mean_frazier_depth": avg_frazier,
            "max_frazier_depth": max_frazier,
            "sentence_frazier_depths": frazier_scores,
            "total_sentences": len(frazier_scores)
        }
    
    async def extract_t_units(self, text: str) -> Dict[str, Any]:
        """
        Extract T-units (minimal terminable units)
        
        A T-unit is one main clause plus all subordinate clauses attached to it.
        This is a key metric for measuring syntactic complexity.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with T-unit information
        """
        parsed = await self.parser.parse_text(text)
        
        t_units = []
        total_words = 0
        total_clauses = 0
        
        for sentence in parsed["sentences"]:
            # Identify main clause and subordinate clauses
            main_clause, subordinate_clauses = self._identify_clauses(sentence["tokens"])
            
            if main_clause:
                t_unit = {
                    "text": sentence["text"],
                    "main_clause": main_clause,
                    "subordinate_clauses": subordinate_clauses,
                    "num_clauses": 1 + len(subordinate_clauses),
                    "num_words": len([t for t in sentence["tokens"] if t["is_alpha"]])
                }
                t_units.append(t_unit)
                total_words += t_unit["num_words"]
                total_clauses += t_unit["num_clauses"]
        
        avg_words_per_tunit = total_words / len(t_units) if t_units else 0.0
        avg_clauses_per_tunit = total_clauses / len(t_units) if t_units else 0.0
        
        return {
            "t_units": t_units,
            "num_t_units": len(t_units),
            "avg_words_per_tunit": avg_words_per_tunit,
            "avg_clauses_per_tunit": avg_clauses_per_tunit,
            "total_words": total_words,
            "total_clauses": total_clauses
        }
    
    async def calculate_subordination_index(self, text: str) -> Dict[str, Any]:
        """
        Calculate subordination index (subordinate clauses per T-unit)
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with subordination metrics
        """
        t_units_data = await self.extract_t_units(text)
        
        if t_units_data["num_t_units"] == 0:
            return {
                "subordination_index": 0.0,
                "subordinate_clauses_per_tunit": 0.0,
                "num_t_units": 0
            }
        
        total_subordinate = sum(
            len(tu["subordinate_clauses"]) for tu in t_units_data["t_units"]
        )
        
        subordination_index = total_subordinate / t_units_data["num_t_units"]
        
        return {
            "subordination_index": subordination_index,
            "subordinate_clauses_per_tunit": subordination_index,
            "num_t_units": t_units_data["num_t_units"],
            "total_subordinate_clauses": total_subordinate
        }
    
    def _build_dependency_tree(self, tokens: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build tree structure from tokens"""
        if not tokens:
            return None
        
        # Find root (token with dep == "ROOT" or head is None)
        root_token = None
        token_map = {}
        
        for token in tokens:
            if token["dep"] == "ROOT" or token["head"] is None:
                root_token = token
            token_map[token["text"]] = token
        
        if not root_token:
            # Fallback: use first token
            root_token = tokens[0]
        
        return self._build_tree_node(root_token, tokens, token_map)
    
    def _build_tree_node(self, token: Dict[str, Any], all_tokens: List[Dict[str, Any]], 
                        token_map: Dict[str, Any], visited: Optional[set] = None, max_depth: int = 50) -> Dict[str, Any]:
        """Recursively build tree node with cycle protection"""
        if visited is None:
            visited = set()
        
        # Protection against infinite recursion
        if len(visited) > max_depth:
            return {
                "text": token["text"],
                "pos": token.get("pos", "UNK"),
                "dep": token.get("dep", "UNK"),
                "children": []
            }
        
        token_id = id(token)  # Use object id to track visited nodes
        if token_id in visited:
            return {
                "text": token["text"],
                "pos": token.get("pos", "UNK"),
                "dep": token.get("dep", "UNK"),
                "children": []
            }
        
        visited.add(token_id)
        
        node = {
            "text": token["text"],
            "pos": token.get("pos", "UNK"),
            "dep": token.get("dep", "UNK"),
            "children": []
        }
        
        # Find children (tokens that have this token as head)
        for t in all_tokens:
            if t.get("head") == token["text"] and id(t) != token_id:
                child_node = self._build_tree_node(t, all_tokens, token_map, visited.copy(), max_depth)
                node["children"].append(child_node)
        
        return node
    
    def _calculate_yngve_for_tree(self, tree: Dict[str, Any], depth: int = 0, max_depth: int = 100) -> float:
        """Calculate Yngve depth recursively with depth limit"""
        if depth > max_depth:
            return 0.0
        
        if not tree or "children" not in tree:
            return float(depth)
        
        # Yngve: count left-branching nodes (children before current position)
        yngve_score = float(depth)
        
        for i, child in enumerate(tree["children"]):
            # Left-branching: earlier children add to complexity
            child_score = self._calculate_yngve_for_tree(child, depth + i + 1, max_depth)
            yngve_score += child_score
        
        return yngve_score
    
    def _calculate_frazier_for_tree(self, tree: Dict[str, Any], depth: int = 0, max_depth: int = 100) -> float:
        """Calculate Frazier depth recursively with depth limit"""
        if depth > max_depth:
            return 0.0
        
        if not tree or "children" not in tree:
            return float(depth)
        
        # Frazier: count right-branching nodes
        frazier_score = float(depth)
        
        for child in tree["children"]:
            child_score = self._calculate_frazier_for_tree(child, depth + 1, max_depth)
            frazier_score = max(frazier_score, child_score)
        
        return frazier_score
    
    def _identify_clauses(self, tokens: List[Dict[str, Any]]) -> tuple:
        """
        Identify main clause and subordinate clauses
        
        Returns:
            (main_clause_tokens, [subordinate_clause_tokens])
        """
        main_clause = []
        subordinate_clauses = []
        current_clause = []
        in_subordinate = False
        
        # Subordinating conjunctions and markers
        subord_markers = ["que", "quando", "onde", "como", "porque", "se", 
                         "embora", "enquanto", "até", "conforme"]
        
        for token in tokens:
            # Check if this starts a subordinate clause
            if token["text"].lower() in subord_markers or token["dep"] in ["mark", "advcl", "acl"]:
                if current_clause and not in_subordinate:
                    # Save previous main clause
                    main_clause = current_clause.copy()
                    current_clause = []
                in_subordinate = True
                current_clause.append(token)
            elif in_subordinate:
                current_clause.append(token)
                # Check if clause ends (punctuation or new sentence)
                if token.get("is_punct") and token["text"] in [".", "!", "?"]:
                    subordinate_clauses.append(current_clause.copy())
                    current_clause = []
                    in_subordinate = False
            else:
                current_clause.append(token)
        
        # Handle remaining clause
        if current_clause:
            if in_subordinate:
                subordinate_clauses.append(current_clause)
            else:
                main_clause = current_clause
        
        return main_clause, subordinate_clauses

