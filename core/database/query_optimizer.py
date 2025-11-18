"""
Query Optimizer and Performance Monitoring
Tracks query performance, detects N+1 patterns, and provides optimization suggestions
"""
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
import time
import logging
import statistics

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query execution"""
    query_name: str
    duration_ms: float
    timestamp: datetime
    params: Dict[str, Any] = field(default_factory=dict)
    result_count: Optional[int] = None
    error: Optional[str] = None


@dataclass
class QueryStats:
    """Aggregated statistics for a query"""
    query_name: str
    call_count: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    error_count: int = 0
    durations: List[float] = field(default_factory=list)


class QueryOptimizer:
    """
    Query performance optimizer and monitoring system

    Features:
    - Query timing and profiling
    - N+1 query detection
    - Slow query logging
    - Query statistics and percentiles
    - Optimization suggestions
    """

    def __init__(
        self,
        slow_query_threshold_ms: float = 100.0,
        n_plus_one_threshold: int = 10,
        max_history: int = 10000,
    ):
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.n_plus_one_threshold = n_plus_one_threshold
        self.max_history = max_history

        # Query history
        self.query_history: List[QueryMetrics] = []

        # Aggregated statistics
        self.query_stats: Dict[str, QueryStats] = defaultdict(QueryStats)

        # N+1 detection
        self.query_patterns: Dict[str, List[float]] = defaultdict(list)

        logger.info(
            f"Query optimizer initialized: "
            f"slow_threshold={slow_query_threshold_ms}ms, "
            f"n+1_threshold={n_plus_one_threshold}"
        )

    def track_query(
        self,
        query_name: str,
        duration_ms: float,
        params: Optional[Dict] = None,
        result_count: Optional[int] = None,
        error: Optional[str] = None,
    ):
        """Track a query execution"""
        metric = QueryMetrics(
            query_name=query_name,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow(),
            params=params or {},
            result_count=result_count,
            error=error,
        )

        # Add to history
        self.query_history.append(metric)
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)

        # Update statistics
        stats = self.query_stats[query_name]
        stats.query_name = query_name
        stats.call_count += 1
        stats.total_duration_ms += duration_ms
        stats.durations.append(duration_ms)

        if error:
            stats.error_count += 1

        # Update min/max
        stats.min_duration_ms = min(stats.min_duration_ms, duration_ms)
        stats.max_duration_ms = max(stats.max_duration_ms, duration_ms)

        # Update percentiles (keep last 1000 for efficiency)
        if len(stats.durations) > 1000:
            stats.durations = stats.durations[-1000:]

        stats.avg_duration_ms = stats.total_duration_ms / stats.call_count
        if len(stats.durations) >= 2:
            sorted_durations = sorted(stats.durations)
            stats.p50_duration_ms = statistics.median(sorted_durations)
            stats.p95_duration_ms = sorted_durations[int(len(sorted_durations) * 0.95)]
            stats.p99_duration_ms = sorted_durations[int(len(sorted_durations) * 0.99)]

        # Check for slow query
        if duration_ms > self.slow_query_threshold_ms:
            logger.warning(
                f"🐢 SLOW QUERY: {query_name} took {duration_ms:.2f}ms "
                f"(threshold: {self.slow_query_threshold_ms}ms) "
                f"params={params}"
            )

        # Check for N+1 pattern
        self._check_n_plus_one(query_name, duration_ms)

    def _check_n_plus_one(self, query_name: str, duration_ms: float):
        """Detect N+1 query patterns"""
        # Track recent queries (last 5 seconds)
        now = time.time()
        self.query_patterns[query_name].append(now)

        # Remove old entries
        self.query_patterns[query_name] = [
            t for t in self.query_patterns[query_name]
            if now - t < 5.0
        ]

        # Detect N+1 pattern
        count = len(self.query_patterns[query_name])
        if count >= self.n_plus_one_threshold:
            logger.error(
                f"🔴 N+1 QUERY DETECTED: {query_name} called {count} times "
                f"in 5 seconds - consider using batch loading or JOIN!"
            )

    def get_query_stats(self, query_name: Optional[str] = None) -> Dict[str, QueryStats]:
        """Get statistics for a specific query or all queries"""
        if query_name:
            return {query_name: self.query_stats.get(query_name, QueryStats(query_name))}
        return dict(self.query_stats)

    def get_slow_queries(self, limit: int = 10) -> List[QueryMetrics]:
        """Get slowest queries"""
        sorted_queries = sorted(
            self.query_history,
            key=lambda q: q.duration_ms,
            reverse=True
        )
        return sorted_queries[:limit]

    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on query patterns"""
        suggestions = []

        for query_name, stats in self.query_stats.items():
            # Suggest indices for slow queries
            if stats.avg_duration_ms > self.slow_query_threshold_ms:
                suggestions.append(
                    f"⚠️  {query_name}: Average {stats.avg_duration_ms:.2f}ms "
                    f"(called {stats.call_count}x) - Consider adding an index"
                )

            # Suggest batch loading for N+1 patterns
            if stats.call_count > self.n_plus_one_threshold:
                suggestions.append(
                    f"🔴 {query_name}: Called {stats.call_count}x "
                    f"- Possible N+1 pattern, use batch loading or JOIN"
                )

            # Suggest caching for frequently called queries
            if stats.call_count > 100 and stats.avg_duration_ms > 50:
                suggestions.append(
                    f"💾 {query_name}: Frequently called ({stats.call_count}x) "
                    f"with avg {stats.avg_duration_ms:.2f}ms - Consider caching"
                )

        return suggestions

    def print_report(self):
        """Print a formatted performance report"""
        print("\n" + "="*80)
        print("📊 QUERY PERFORMANCE REPORT")
        print("="*80 + "\n")

        # Top 10 slowest queries
        print("🐢 TOP 10 SLOWEST QUERIES:")
        for i, query in enumerate(self.get_slow_queries(10), 1):
            print(f"  {i}. {query.query_name}: {query.duration_ms:.2f}ms")
        print()

        # Query statistics
        print("📈 QUERY STATISTICS:")
        for query_name, stats in sorted(
            self.query_stats.items(),
            key=lambda x: x[1].total_duration_ms,
            reverse=True
        )[:10]:
            print(f"\n  {query_name}:")
            print(f"    Calls: {stats.call_count}")
            print(f"    Total: {stats.total_duration_ms:.2f}ms")
            print(f"    Avg: {stats.avg_duration_ms:.2f}ms")
            print(f"    Min/Max: {stats.min_duration_ms:.2f}ms / {stats.max_duration_ms:.2f}ms")
            print(f"    P50/P95/P99: {stats.p50_duration_ms:.2f}ms / {stats.p95_duration_ms:.2f}ms / {stats.p99_duration_ms:.2f}ms")
            if stats.error_count > 0:
                print(f"    Errors: {stats.error_count}")
        print()

        # Optimization suggestions
        suggestions = self.get_optimization_suggestions()
        if suggestions:
            print("💡 OPTIMIZATION SUGGESTIONS:")
            for suggestion in suggestions[:10]:
                print(f"  {suggestion}")
        else:
            print("✅ No optimization suggestions - queries look good!")

        print("\n" + "="*80 + "\n")

    def clear_stats(self):
        """Clear all statistics"""
        self.query_history.clear()
        self.query_stats.clear()
        self.query_patterns.clear()
        logger.info("Query statistics cleared")


# Singleton instance
_optimizer: Optional[QueryOptimizer] = None


def get_query_optimizer() -> QueryOptimizer:
    """Get or create the query optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = QueryOptimizer(
            slow_query_threshold_ms=100.0,
            n_plus_one_threshold=10,
        )
    return _optimizer


def track_query(query_name: str):
    """
    Decorator to automatically track query performance

    Usage:
        @track_query("get_user_documents")
        def get_user_documents(user_id: str):
            # Query implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_query_optimizer()
            start_time = time.time()
            error = None
            result = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000

                # Extract params (if first arg is self, skip it)
                params = {}
                if args:
                    if hasattr(args[0], '__class__'):
                        params = {'args': args[1:], 'kwargs': kwargs}
                    else:
                        params = {'args': args, 'kwargs': kwargs}

                # Get result count if iterable
                result_count = None
                if result is not None:
                    if isinstance(result, (list, tuple)):
                        result_count = len(result)
                    elif isinstance(result, dict):
                        result_count = len(result)

                optimizer.track_query(
                    query_name=query_name,
                    duration_ms=duration_ms,
                    params=params,
                    result_count=result_count,
                    error=error,
                )

        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    optimizer = QueryOptimizer(slow_query_threshold_ms=50.0)

    # Simulate queries
    for i in range(100):
        optimizer.track_query("get_user_documents", duration_ms=45.0 + i % 20)

    for i in range(20):
        optimizer.track_query("add_message", duration_ms=120.0)  # Slow

    # Print report
    optimizer.print_report()
