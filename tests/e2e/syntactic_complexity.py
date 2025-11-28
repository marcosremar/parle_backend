"""
Syntactic Complexity Metrics
Implements subordination index, dependency depth, T-unit analysis, clause density
Requires dependency parser from linguistic_analysis service
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from loguru import logger


class SyntacticComplexityCalculator:
    """Calculate syntactic complexity metrics using dependency parser"""
    
    def __init__(self, linguistic_analysis_url: str = "http://localhost:8901"):
        """
        Initialize calculator
        
        Args:
            linguistic_analysis_url: URL of linguistic analysis service
        """
        self.linguistic_analysis_url = linguistic_analysis_url
    
    async def calculate_all_syntactic_metrics(self, text: str) -> Dict[str, Any]:
        """
        Calculate all syntactic complexity metrics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with all syntactic metrics
        """
        try:
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Get all syntactic metrics from service
                url = f"{self.linguistic_analysis_url}/api/syntactic-metrics"
                async with session.post(url, json={"text": text}) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract key metrics
                        yngve = data.get("yngve_depth", {})
                        frazier = data.get("frazier_depth", {})
                        t_units = data.get("t_units", {})
                        subordination = data.get("subordination_index", {})
                        
                        return {
                            "yngve_mean_depth": yngve.get("mean_yngve_depth", 0.0),
                            "yngve_max_depth": yngve.get("max_yngve_depth", 0.0),
                            "frazier_mean_depth": frazier.get("mean_frazier_depth", 0.0),
                            "frazier_max_depth": frazier.get("max_frazier_depth", 0.0),
                            "num_t_units": t_units.get("num_t_units", 0),
                            "avg_words_per_tunit": t_units.get("avg_words_per_tunit", 0.0),
                            "avg_clauses_per_tunit": t_units.get("avg_clauses_per_tunit", 0.0),
                            "subordination_index": subordination.get("subordination_index", 0.0),
                            "clause_density": self._calculate_clause_density(t_units),
                            "coordination_ratio": self._calculate_coordination_ratio(t_units, subordination)
                        }
                    else:
                        error_text = await response.text()
                        logger.warning(f"Linguistic analysis service returned {response.status}: {error_text[:200]}")
                        return self._get_default_metrics()
        except aiohttp.ClientError as e:
            logger.warning(f"Client error calling linguistic analysis service: {e}")
            return self._get_default_metrics()
        except asyncio.TimeoutError:
            logger.warning(f"Timeout calling linguistic analysis service")
            return self._get_default_metrics()
        except Exception as e:
            logger.error(f"Error calculating syntactic metrics: {e}")
            return self._get_default_metrics()
    
    async def calculate_subordination_index(self, text: str) -> float:
        """
        Calculate subordination index (subordinate clauses per T-unit)
        
        Args:
            text: Input text
            
        Returns:
            Subordination index
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.linguistic_analysis_url}/api/subordination-index"
                async with session.post(url, json={"text": text}) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("subordination_index", 0.0)
                    return 0.0
        except Exception as e:
            logger.error(f"Error calculating subordination index: {e}")
            return 0.0
    
    async def calculate_dependency_depth(self, text: str) -> Dict[str, float]:
        """
        Calculate dependency depth (Yngve and Frazier)
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with mean and max depths
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Get Yngve depth
                url_yngve = f"{self.linguistic_analysis_url}/api/yngve-depth"
                async with session.post(url_yngve, json={"text": text}) as response:
                    yngve_mean = 0.0
                    yngve_max = 0.0
                    if response.status == 200:
                        data = await response.json()
                        yngve_mean = data.get("mean_yngve_depth", 0.0)
                        yngve_max = data.get("max_yngve_depth", 0.0)
                
                # Get Frazier depth
                url_frazier = f"{self.linguistic_analysis_url}/api/frazier-depth"
                async with session.post(url_frazier, json={"text": text}) as response:
                    frazier_mean = 0.0
                    frazier_max = 0.0
                    if response.status == 200:
                        data = await response.json()
                        frazier_mean = data.get("mean_frazier_depth", 0.0)
                        frazier_max = data.get("max_frazier_depth", 0.0)
                
                return {
                    "yngve_mean": yngve_mean,
                    "yngve_max": yngve_max,
                    "frazier_mean": frazier_mean,
                    "frazier_max": frazier_max
                }
        except Exception as e:
            logger.error(f"Error calculating dependency depth: {e}")
            return {
                "yngve_mean": 0.0,
                "yngve_max": 0.0,
                "frazier_mean": 0.0,
                "frazier_max": 0.0
            }
    
    async def calculate_t_unit_metrics(self, text: str) -> Dict[str, Any]:
        """
        Calculate T-unit metrics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with T-unit metrics
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.linguistic_analysis_url}/api/t-units"
                async with session.post(url, json={"text": text}) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "num_t_units": data.get("num_t_units", 0),
                            "avg_words_per_tunit": data.get("avg_words_per_tunit", 0.0),
                            "avg_clauses_per_tunit": data.get("avg_clauses_per_tunit", 0.0),
                            "total_words": data.get("total_words", 0),
                            "total_clauses": data.get("total_clauses", 0)
                        }
                    return self._get_default_tunit_metrics()
        except Exception as e:
            logger.error(f"Error calculating T-unit metrics: {e}")
            return self._get_default_tunit_metrics()
    
    def _calculate_clause_density(self, t_units_data: Dict[str, Any]) -> float:
        """
        Calculate clause density (clauses per sentence)
        
        Args:
            t_units_data: T-units data from service
            
        Returns:
            Clause density
        """
        num_t_units = t_units_data.get("num_t_units", 0)
        total_clauses = t_units_data.get("total_clauses", 0)
        
        if num_t_units == 0:
            return 0.0
        
        return total_clauses / num_t_units
    
    def _calculate_coordination_ratio(self, t_units_data: Dict[str, Any], 
                                       subordination_data: Dict[str, Any]) -> float:
        """
        Calculate coordination vs subordination ratio
        
        Args:
            t_units_data: T-units data
            subordination_data: Subordination data
            
        Returns:
            Ratio of coordination to subordination
        """
        num_t_units = t_units_data.get("num_t_units", 0)
        subordination_index = subordination_data.get("subordination_index", 0.0)
        
        if num_t_units == 0:
            return 0.0
        
        # Total subordinate clauses
        total_subordinate = subordination_index * num_t_units
        
        # Total clauses (main + subordinate)
        total_clauses = t_units_data.get("total_clauses", 0)
        
        # Coordination = main clauses = total - subordinate
        total_main = total_clauses - total_subordinate if total_clauses >= total_subordinate else 0
        
        # Ratio: coordination / subordination
        if subordination_index > 0:
            return total_main / total_subordinate
        else:
            # If no subordination, return high ratio
            return 10.0 if total_main > 0 else 0.0
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when service is unavailable"""
        return {
            "yngve_mean_depth": 0.0,
            "yngve_max_depth": 0.0,
            "frazier_mean_depth": 0.0,
            "frazier_max_depth": 0.0,
            "num_t_units": 0,
            "avg_words_per_tunit": 0.0,
            "avg_clauses_per_tunit": 0.0,
            "subordination_index": 0.0,
            "clause_density": 0.0,
            "coordination_ratio": 0.0
        }
    
    def _get_default_tunit_metrics(self) -> Dict[str, Any]:
        """Return default T-unit metrics"""
        return {
            "num_t_units": 0,
            "avg_words_per_tunit": 0.0,
            "avg_clauses_per_tunit": 0.0,
            "total_words": 0,
            "total_clauses": 0
        }

