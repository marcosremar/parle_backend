"""
Attentive Knowledge Tracing (AKT) Implementation
Modelo avançado de knowledge tracing usando attention mechanisms

AKT considera:
- Sequência de interações (não apenas última)
- Contexto e dificuldade
- Relações entre habilidades
- Adaptação individual por estudante
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone
from .base_tracer import KnowledgeTracer
import math


class AttentiveKnowledgeTracer(KnowledgeTracer):
    """
    Implementação do Attentive Knowledge Tracing (AKT)
    
    AKT usa attention mechanisms para considerar toda a sequência de interações,
    não apenas a última, permitindo adaptação mais precisa ao progresso individual.
    
    Características:
    - Attention sobre histórico de interações
    - Adaptação de parâmetros por estudante
    - Considera dificuldade e contexto
    - Modela relações temporais
    """
    
    def __init__(self, skill_params: Optional[Dict[str, Any]] = None):
        """
        Inicializa o AKT tracer
        
        Args:
            skill_params: Parâmetros do modelo AKT
        """
        if skill_params is None:
            skill_params = self._get_default_params()
        
        # Parâmetros base (similar ao BKT, mas adaptativos)
        self.p_L0 = skill_params.get('p_L0', 0.2)
        self.p_T = skill_params.get('p_T', 0.15)
        self.p_F = skill_params.get('p_F', 0.05)
        self.p_G = skill_params.get('p_G', 0.85)
        self.p_S = skill_params.get('p_S', 0.3)
        
        # Estado atual
        self.mastery_probability = self.p_L0
        
        # Histórico de interações (para attention)
        self.interaction_history: List[Dict[str, Any]] = []
        
        # Parâmetros adaptativos (ajustam-se com o tempo)
        self.adaptive_p_T = self.p_T
        self.adaptive_p_F = self.p_F
        
        # Estatísticas
        self.total_interactions = 0
        self.correct_count = 0
        self.recent_correct_streak = 0
        self.recent_error_streak = 0
        
        # Decay factor para interações antigas
        self.temporal_decay = skill_params.get('temporal_decay', 0.95)
        
        # Rastreamento de padrões por linguistic features (para adaptação)
        # Ex: se sempre erra na 3ª pessoa, ajustar p_T para essa feature
        self.feature_patterns: Dict[str, Dict[str, Any]] = {}  # feature_key -> {correct_count, error_count, last_seen}
        
        # FoLiBi parameters
        self.folibi_enabled = skill_params.get('folibi_enabled', True)
        self.linear_decay_factor = skill_params.get('linear_decay_factor', 0.3)
        
        # Attention window (aumentado para melhor context-awareness)
        self.attention_window = skill_params.get('attention_window', 30)  # Padrão: 30 (aumentado de 10)
    
    @staticmethod
    def _get_default_params() -> Dict[str, Any]:
        """Retorna parâmetros padrão do AKT"""
        return {
            'p_L0': 0.2,
            'p_T': 0.15,
            'p_F': 0.05,
            'p_G': 0.85,
            'p_S': 0.3,
            'temporal_decay': 0.95,  # Decay para interações antigas
            'attention_window': 30,   # Número de interações para considerar (aumentado)
            'adaptation_rate': 0.1,   # Taxa de adaptação de parâmetros
            'folibi_enabled': True,   # FoLiBi forgetting-aware linear bias
            'linear_decay_factor': 0.3  # Fator de decay linear para FoLiBi
        }
    
    def _compute_folibi_bias(self, interaction_index: int, total_interactions: int) -> float:
        """
        Calcula bias linear FoLiBi (Forgetting-aware Linear Bias)
        
        Desacopla esquecimento da correlação entre questões usando bias linear
        baseado na posição na sequência.
        
        Args:
            interaction_index: Índice da interação (0 = mais antiga)
            total_interactions: Total de interações no histórico
            
        Returns:
            Bias linear (0.0 a 1.0)
        """
        if not self.folibi_enabled or total_interactions == 0:
            return 1.0
        
        # Bias linear: interações mais antigas têm menos peso
        # bias = 1.0 - (position / total) * linear_decay_factor
        position_ratio = interaction_index / total_interactions
        bias = 1.0 - (position_ratio * self.linear_decay_factor)
        
        return max(0.0, min(1.0, bias))
    
    def _compute_attention_weights(self, current_time: datetime) -> List[float]:
        """
        Calcula pesos de atenção para interações históricas
        
        Interações mais recentes recebem mais peso.
        Usa decay temporal exponencial combinado com FoLiBi linear bias.
        
        Args:
            current_time: Tempo atual
            
        Returns:
            Lista de pesos de atenção (normalizados)
        """
        if not self.interaction_history:
            return []
        
        weights = []
        total_interactions = len(self.interaction_history)
        
        for i, interaction in enumerate(self.interaction_history):
            # Calcular tempo desde a interação
            interaction_time = interaction['timestamp']
            # Ensure both are timezone-aware
            if interaction_time.tzinfo is None:
                interaction_time = interaction_time.replace(tzinfo=timezone.utc)
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            time_diff = (current_time - interaction_time).total_seconds()
            days_diff = time_diff / 86400  # Converter para dias
            
            # Decay exponencial: interações mais recentes têm mais peso
            temporal_weight = math.exp(-days_diff * (1 - self.temporal_decay))
            
            # FoLiBi bias linear (desacopla esquecimento da correlação)
            folibi_bias = self._compute_folibi_bias(i, total_interactions)
            
            # Combinar: weight = temporal_decay * folibi_bias
            weight = temporal_weight * folibi_bias
            weights.append(weight)
        
        # Normalizar pesos
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return weights
    
    def _compute_contextual_features(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extrai features contextuais da interação
        
        Inclui dificuldade IRT (Rasch Model-based) do skill.
        
        Args:
            context: Contexto da interação (pode conter 'difficulty', 'skill_id', etc.)
            
        Returns:
            Dicionário com features contextuais (valores podem ser float ou string para linguistic features)
        """
        features: Dict[str, Any] = {
            'difficulty': 0.5,  # Dificuldade média por padrão
            'complexity': 0.5,  # Complexidade média
            'time_since_last': 1.0,  # 1 dia por padrão
            'error_type_severity': 0.5  # Severidade média
        }
        
        if context:
            # Se difficulty não está no context, tentar obter do skill_id via IRT
            if 'difficulty' in context:
                features['difficulty'] = context.get('difficulty', 0.5)
            elif 'skill_id' in context:
                # Importar função para obter dificuldade IRT
                from ..skill_registry import get_skill_difficulty
                features['difficulty'] = get_skill_difficulty(context['skill_id'])
            else:
                features['difficulty'] = 0.5
            
            features['complexity'] = context.get('complexity', 0.5)
            features['error_type_severity'] = context.get('error_severity', 0.5)
            
            # Extrair linguistic_features do context e incluí-las nas features contextuais
            # para uso em similaridade semântica e adaptação de parâmetros
            if 'linguistic_features' in context:
                linguistic_features = context['linguistic_features']
                # Adicionar features linguísticas relevantes às features contextuais
                if isinstance(linguistic_features, dict):
                    # Incluir tense, person, number para similaridade
                    if 'tense' in linguistic_features:
                        features['tense'] = linguistic_features['tense']
                    if 'person' in linguistic_features:
                        features['person'] = linguistic_features['person']
                    if 'number' in linguistic_features:
                        features['number'] = linguistic_features['number']
                    if 'register' in linguistic_features:
                        features['register'] = linguistic_features['register']
                    if 'domain' in linguistic_features:
                        features['domain'] = linguistic_features['domain']
            
            # Calcular tempo desde última interação
            if self.interaction_history:
                last_time = self.interaction_history[-1]['timestamp']
                # Ensure both datetimes are timezone-aware
                if last_time.tzinfo is None:
                    # If naive, assume UTC
                    last_time = last_time.replace(tzinfo=timezone.utc)
                current_time = datetime.now(timezone.utc)
                time_diff = (current_time - last_time).total_seconds()
                features['time_since_last'] = time_diff / 86400  # Em dias
        
        return features
    
    def _adapt_parameters(self, correct: bool, features: Dict[str, Any]):
        """
        Adapta parâmetros do modelo baseado no histórico e linguistic features
        
        Usa padrões de linguistic features (tense, person, number, register, domain)
        para ajustar parâmetros adaptativos. Ex: se sempre erra na 3ª pessoa,
        aumenta p_T para essa feature específica.
        
        Args:
            correct: Se a resposta foi correta
            features: Features contextuais (pode conter linguistic features como strings)
        """
        adaptation_rate = 0.1
        
        # Adaptar p_T (probabilidade de aprender) baseado em streaks
        if correct:
            self.recent_correct_streak += 1
            self.recent_error_streak = 0
            
            # Se está em streak de acertos, aumenta p_T (aprende mais rápido)
            if self.recent_correct_streak >= 3:
                self.adaptive_p_T = min(0.3, self.adaptive_p_T + adaptation_rate * 0.05)
        else:
            self.recent_error_streak += 1
            self.recent_correct_streak = 0
            
            # Se está em streak de erros, diminui p_T (aprende mais devagar)
            if self.recent_error_streak >= 3:
                self.adaptive_p_T = max(0.05, self.adaptive_p_T - adaptation_rate * 0.05)
        
        # Adaptar p_F (probabilidade de esquecer) baseado em tempo
        time_since_last = features.get('time_since_last', 1.0)
        if time_since_last > 7:  # Mais de uma semana
            self.adaptive_p_F = min(0.15, self.adaptive_p_F + adaptation_rate * 0.02)
        elif time_since_last < 1:  # Menos de um dia
            self.adaptive_p_F = max(0.01, self.adaptive_p_F - adaptation_rate * 0.01)
        
        # NOVO: Adaptação baseada em linguistic features (padrões de erro)
        # Rastrear padrões por feature linguística
        linguistic_feature_keys = ['tense', 'person', 'number', 'register', 'domain']
        for key in linguistic_feature_keys:
            if key in features:
                feature_value = features[key]
                # Criar chave composta para rastreamento
                pattern_key = f"{key}:{feature_value}"
                
                # Inicializar padrão se não existir
                if pattern_key not in self.feature_patterns:
                    self.feature_patterns[pattern_key] = {
                        'correct_count': 0,
                        'error_count': 0,
                        'last_seen': None
                    }
                
                # Atualizar contadores
                if correct:
                    self.feature_patterns[pattern_key]['correct_count'] += 1
                else:
                    self.feature_patterns[pattern_key]['error_count'] += 1
                
                self.feature_patterns[pattern_key]['last_seen'] = datetime.now(timezone.utc)
                
                # Se há padrão claro de erro para esta feature (>= 3 erros, taxa de erro > 70%)
                pattern = self.feature_patterns[pattern_key]
                total_attempts = pattern['correct_count'] + pattern['error_count']
                if total_attempts >= 3:
                    error_rate = pattern['error_count'] / total_attempts
                    if error_rate > 0.7:  # Mais de 70% de erros para esta feature
                        # Aumentar p_T ligeiramente para esta feature específica
                        # (indica que precisa de mais prática, não que é impossível)
                        self.adaptive_p_T = min(0.3, self.adaptive_p_T + adaptation_rate * 0.02)
                    elif error_rate < 0.3 and total_attempts >= 5:  # Menos de 30% de erros, bem praticado
                        # Diminuir p_T ligeiramente (já domina esta feature)
                        self.adaptive_p_T = max(0.05, self.adaptive_p_T - adaptation_rate * 0.01)
    
    def _compute_semantic_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Calcula similaridade semântica entre duas interações baseada em features contextuais
        
        Compara:
        - error_type, category (se disponíveis)
        - difficulty
        - complexity
        - error_type_severity
        - linguistic features (tense, person, number, register, domain)
        
        Args:
            features1: Features da primeira interação
            features2: Features da segunda interação
            
        Returns:
            Similaridade (0.0 a 1.0)
        """
        similarity_score = 0.0
        total_weights = 0.0
        
        # Comparar difficulty (peso: 0.25)
        if 'difficulty' in features1 and 'difficulty' in features2:
            if isinstance(features1['difficulty'], (int, float)) and isinstance(features2['difficulty'], (int, float)):
                diff_diff = abs(features1['difficulty'] - features2['difficulty'])
                similarity_score += (1.0 - diff_diff) * 0.25
                total_weights += 0.25
        
        # Comparar complexity (peso: 0.15)
        if 'complexity' in features1 and 'complexity' in features2:
            if isinstance(features1['complexity'], (int, float)) and isinstance(features2['complexity'], (int, float)):
                comp_diff = abs(features1['complexity'] - features2['complexity'])
                similarity_score += (1.0 - comp_diff) * 0.15
                total_weights += 0.15
        
        # Comparar error_type_severity (peso: 0.15)
        if 'error_type_severity' in features1 and 'error_type_severity' in features2:
            if isinstance(features1['error_type_severity'], (int, float)) and isinstance(features2['error_type_severity'], (int, float)):
                sev_diff = abs(features1['error_type_severity'] - features2['error_type_severity'])
                similarity_score += (1.0 - sev_diff) * 0.15
                total_weights += 0.15
        
        # Comparar error_type e category se disponíveis (peso: 0.15)
        if 'error_type' in features1 and 'error_type' in features2:
            if features1['error_type'] == features2['error_type']:
                similarity_score += 0.075
            total_weights += 0.075
        
        if 'category' in features1 and 'category' in features2:
            if features1['category'] == features2['category']:
                similarity_score += 0.075
            total_weights += 0.075
        
        # Comparar linguistic features (tense, person, number, register, domain) - peso: 0.30
        linguistic_feature_keys = ['tense', 'person', 'number', 'register', 'domain']
        for key in linguistic_feature_keys:
            if key in features1 and key in features2:
                # Comparar strings (valores de linguistic features)
                if features1[key] == features2[key]:
                    similarity_score += 0.06  # 0.06 * 5 = 0.30 total
                total_weights += 0.06
        
        # Normalizar
        if total_weights > 0:
            return similarity_score / total_weights
        
        return 0.5  # Similaridade média se não houver features para comparar
    
    def _compute_attention_based_update(
        self,
        correct: bool,
        features: Dict[str, Any]
    ) -> float:
        """
        Calcula atualização de mastery usando attention sobre histórico
        
        Considera:
        - Similaridade semântica entre interações
        - Padrões temporais (streaks, intervalos)
        - Decay temporal + FoLiBi bias
        
        Args:
            correct: Se a resposta foi correta
            features: Features contextuais
            
        Returns:
            Nova probabilidade de mastery
        """
        if not self.interaction_history:
            # Primeira interação: usar BKT básico
            return self._bkt_update(correct, features)
        
        # Calcular pesos de atenção (temporal + FoLiBi)
        attention_weights = self._compute_attention_weights(datetime.now(timezone.utc))
        
        # Calcular contribuição de cada interação histórica
        weighted_contributions = []
        for i, interaction in enumerate(self.interaction_history):
            weight = attention_weights[i] if i < len(attention_weights) else 0.0
            was_correct = interaction['correct']
            interaction_features = interaction.get('features', {})
            
            # Similaridade semântica com interação atual
            semantic_similarity = self._compute_semantic_similarity(features, interaction_features)
            
            # Contribuição baseada em similaridade semântica e correção
            # Interações similares e com mesmo resultado (correto/incorreto) têm mais peso
            if was_correct == correct:
                similarity_factor = 1.0 + (semantic_similarity * 0.5)  # Até 50% de bônus por similaridade
            else:
                similarity_factor = 0.5 - (semantic_similarity * 0.2)  # Penalidade menor se similar
            
            # Bônus adicional se linguistic features são idênticas (mesmo tense, person, etc.)
            linguistic_match_bonus = 0.0
            linguistic_feature_keys = ['tense', 'person', 'number', 'register', 'domain']
            matches = 0
            for key in linguistic_feature_keys:
                if key in features and key in interaction_features:
                    if features[key] == interaction_features[key]:
                        matches += 1
            
            if matches > 0:
                # Bônus proporcional ao número de matches (até 20% adicional)
                linguistic_match_bonus = (matches / len(linguistic_feature_keys)) * 0.2
                similarity_factor += linguistic_match_bonus
            
            # Contribuição ponderada
            contribution = weight * similarity_factor * (1.0 if was_correct else -0.5)
            weighted_contributions.append(contribution)
        
        # Soma das contribuições
        total_contribution = sum(weighted_contributions)
        
        # Atualização base (BKT com IRT)
        base_update = self._bkt_update(correct, features)
        
        # Combinar atualização base com atenção
        # Attention modula a atualização base
        attention_modulation = 1.0 + (total_contribution * 0.3)  # Até 30% de modulação
        attention_modulation = max(0.7, min(1.3, attention_modulation))  # Limitar entre 0.7 e 1.3
        
        new_mastery = base_update * attention_modulation
        
        # Aplicar limites
        return max(0.0, min(1.0, new_mastery))
    
    def _bkt_update(self, correct: bool, features: Dict[str, Any]) -> float:
        """
        Atualização base usando BKT com ajuste IRT (Rasch Model-based)
        
        Usa dificuldade do skill para ajustar probabilidade de acerto:
        P(correct) = mastery / (mastery + difficulty * (1 - mastery))
        
        Args:
            correct: Se a resposta foi correta
            features: Features contextuais (inclui 'difficulty')
            
        Returns:
            Nova probabilidade de mastery
        """
        # Obter dificuldade IRT
        difficulty = features.get('difficulty', 0.5)
        
        # Ajustar p_G e p_S baseado na dificuldade IRT
        # Skills mais difíceis têm menor p_G (menor chance de acertar mesmo sabendo)
        # e menor p_S (menor chance de acertar por acaso)
        adjusted_p_G = self.p_G * (1.0 - difficulty * 0.3)  # Reduz até 30% para skills difíceis
        adjusted_p_S = self.p_S * (1.0 - difficulty * 0.2)  # Reduz até 20% para skills difíceis
        
        # Garantir limites mínimos
        adjusted_p_G = max(0.5, min(0.95, adjusted_p_G))
        adjusted_p_S = max(0.1, min(0.4, adjusted_p_S))
        
        # Calcular probabilidades condicionais
        if correct:
            p_correct_given_L1 = adjusted_p_G
            p_correct_given_L0 = adjusted_p_S
        else:
            p_correct_given_L1 = 1 - adjusted_p_G
            p_correct_given_L0 = 1 - adjusted_p_S
        
        # Teorema de Bayes
        p_observation = (
            p_correct_given_L1 * self.mastery_probability +
            p_correct_given_L0 * (1 - self.mastery_probability)
        )
        
        if p_observation == 0:
            p_observation = 0.0001
        
        p_L1_given_obs = (p_correct_given_L1 * self.mastery_probability) / p_observation
        
        # Transição de estado usando parâmetros adaptativos
        new_p_L = (
            p_L1_given_obs * (1 - self.adaptive_p_F) +
            (1 - p_L1_given_obs) * self.adaptive_p_T
        )
        
        return max(0.0, min(1.0, new_p_L))
    
    def update_belief(self, correct: bool, context: Dict[str, Any] = None) -> float:
        """
        Atualiza crença usando modelo AKT com attention
        
        Args:
            correct: Se o estudante acertou (True) ou errou (False)
            context: Contexto adicional (dificuldade, complexidade, etc.)
            
        Returns:
            Nova probabilidade de domínio (0.0 a 1.0)
        """
        # Extrair features contextuais
        features = self._compute_contextual_features(context)
        
        # Adaptar parâmetros baseado no histórico
        self._adapt_parameters(correct, features)
        
        # Calcular atualização usando attention
        new_mastery = self._compute_attention_based_update(correct, features)
        
        # Atualizar estado
        self.mastery_probability = new_mastery
        
        # Adicionar à história (manter apenas últimas 50 interações)
        # Incluir linguistic_features completas no histórico para análise de padrões
        interaction_record = {
            'correct': correct,
            'timestamp': datetime.now(timezone.utc),
            'features': features,
            'mastery_after': new_mastery
        }
        
        # Adicionar linguistic_features completas se disponíveis no context
        if context and 'linguistic_features' in context:
            interaction_record['linguistic_features'] = context['linguistic_features']
        
        self.interaction_history.append(interaction_record)
        
        # Limitar histórico ao attention_window
        if len(self.interaction_history) > self.attention_window:
            self.interaction_history = self.interaction_history[-self.attention_window:]
        
        # Atualizar estatísticas
        self.total_interactions += 1
        if correct:
            self.correct_count += 1
        
        return self.mastery_probability
    
    def predict_performance(self, skill_id: str = None, difficulty: Optional[float] = None) -> float:
        """
        Prediz a probabilidade de o estudante acertar uma questão desta habilidade
        
        Usa attention sobre histórico e ajuste IRT (Rasch Model-based) para predição mais precisa.
        
        Args:
            skill_id: ID da habilidade (usado para obter dificuldade IRT se não fornecida)
            difficulty: Dificuldade IRT do skill (opcional, será obtida do skill_id se não fornecida)
            
        Returns:
            Probabilidade de acerto: P(correto) = P(correto|sabe) * P(sabe) + P(correto|não sabe) * P(não sabe)
            Ajustada pela dificuldade IRT
        """
        # Obter dificuldade IRT
        if difficulty is None and skill_id:
            from ..skill_registry import get_skill_difficulty
            difficulty = get_skill_difficulty(skill_id)
        elif difficulty is None:
            difficulty = 0.5
        
        # Ajustar p_G e p_S baseado na dificuldade IRT
        adjusted_p_G = self.p_G * (1.0 - difficulty * 0.3)
        adjusted_p_S = self.p_S * (1.0 - difficulty * 0.2)
        adjusted_p_G = max(0.5, min(0.95, adjusted_p_G))
        adjusted_p_S = max(0.1, min(0.4, adjusted_p_S))
        
        # Predição base (BKT com ajuste IRT)
        base_prediction = adjusted_p_G * self.mastery_probability + adjusted_p_S * (1 - self.mastery_probability)
        
        # Ajustar baseado em histórico recente (attention)
        if self.interaction_history:
            # Considerar últimas 5 interações
            recent_interactions = self.interaction_history[-5:]
            recent_correct_rate = sum(1 for i in recent_interactions if i['correct']) / len(recent_interactions)
            
            # Modular predição baseada em performance recente
            # Se performance recente é melhor que esperado, aumentar predição
            # Se é pior, diminuir
            adjustment = (recent_correct_rate - base_prediction) * 0.2
            base_prediction = base_prediction + adjustment
        
        return max(0.0, min(1.0, base_prediction))
    
    def get_mastery_probability(self) -> float:
        """
        Retorna a probabilidade atual de domínio
        
        Returns:
            Probabilidade de domínio (0.0 a 1.0)
        """
        return self.mastery_probability
    
    def reset(self):
        """Reseta o estado do tracer para valores iniciais"""
        self.mastery_probability = self.p_L0
        self.interaction_history = []
        self.adaptive_p_T = self.p_T
        self.adaptive_p_F = self.p_F
        self.total_interactions = 0
        self.correct_count = 0
        self.recent_correct_streak = 0
        self.recent_error_streak = 0
        self.feature_patterns = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do tracer"""
        return {
            'mastery_probability': self.mastery_probability,
            'initial_probability': self.p_L0,
            'total_interactions': self.total_interactions,
            'correct_count': self.correct_count,
            'correct_rate': self.correct_count / max(self.total_interactions, 1),
            'recent_correct_streak': self.recent_correct_streak,
            'recent_error_streak': self.recent_error_streak,
            'adaptive_parameters': {
                'p_T': self.adaptive_p_T,
                'p_F': self.adaptive_p_F
            },
            'history_length': len(self.interaction_history),
            'attention_window': self.attention_window,
            'folibi_enabled': self.folibi_enabled,
            'feature_patterns_count': len(self.feature_patterns),
            'parameters': {
                'p_L0': self.p_L0,
                'p_T': self.p_T,
                'p_F': self.p_F,
                'p_G': self.p_G,
                'p_S': self.p_S
            }
        }

