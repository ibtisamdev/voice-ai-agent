"""
Performance monitoring and optimization utilities for the AI system.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
import functools
import psutil
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationStats:
    """Statistics for a specific operation."""
    operation_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_call: Optional[datetime] = None
    error_count: int = 0
    
    @property
    def avg_time(self) -> float:
        """Calculate average execution time."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    def add_measurement(self, execution_time: float, success: bool = True):
        """Add a new measurement to the statistics."""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.last_call = datetime.utcnow()
        
        if not success:
            self.error_count += 1


class PerformanceMonitor:
    """Performance monitoring system for AI operations."""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.operation_stats: Dict[str, OperationStats] = {}
        self.system_metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval: float = 30.0):
        """Start background system monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_system_metrics,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background system monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_system_metrics(self, interval: float):
        """Background thread for system metrics collection."""
        while self._monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                with self._lock:
                    self.system_metrics.update({
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_available_gb': memory.available / (1024**3),
                        'disk_percent': disk.percent,
                        'disk_free_gb': disk.free / (1024**3),
                        'timestamp': datetime.utcnow()
                    })
                
                # Add metrics to history
                timestamp = datetime.utcnow()
                self.add_metric("system.cpu_percent", cpu_percent, "percent", timestamp)
                self.add_metric("system.memory_percent", memory.percent, "percent", timestamp)
                self.add_metric("system.memory_available", memory.available / (1024**3), "GB", timestamp)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(interval)
    
    def add_metric(self, name: str, value: float, unit: str, 
                  timestamp: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None):
        """Add a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=timestamp or datetime.utcnow(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.metrics_history.append(metric)
    
    def record_operation(self, operation_name: str, execution_time: float, success: bool = True):
        """Record operation execution statistics."""
        with self._lock:
            if operation_name not in self.operation_stats:
                self.operation_stats[operation_name] = OperationStats(operation_name)
            
            self.operation_stats[operation_name].add_measurement(execution_time, success)
        
        # Add to metrics history
        self.add_metric(
            f"operation.{operation_name}.execution_time",
            execution_time,
            "seconds",
            metadata={"success": success}
        )
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, OperationStats]:
        """Get operation statistics."""
        with self._lock:
            if operation_name:
                return {operation_name: self.operation_stats.get(operation_name)}
            return dict(self.operation_stats)
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of metrics for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_name[metric.name].append(metric.value)
        
        # Calculate summaries
        summary = {}
        for name, values in metrics_by_name.items():
            summary[name] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1] if values else None
            }
        
        # Add operation summaries
        operation_summary = {}
        for op_name, stats in self.operation_stats.items():
            if stats.last_call and stats.last_call >= cutoff_time:
                operation_summary[op_name] = {
                    "call_count": stats.call_count,
                    "avg_time": stats.avg_time,
                    "min_time": stats.min_time,
                    "max_time": stats.max_time,
                    "error_rate": stats.error_count / stats.call_count if stats.call_count > 0 else 0,
                    "last_call": stats.last_call.isoformat()
                }
        
        return {
            "metrics": summary,
            "operations": operation_summary,
            "system": dict(self.system_metrics),
            "period_hours": hours,
            "total_metrics": len(recent_metrics)
        }
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        with self._lock:
            # Check system metrics
            if self.system_metrics.get('cpu_percent', 0) > 80:
                recommendations.append("High CPU usage detected. Consider reducing concurrent operations.")
            
            if self.system_metrics.get('memory_percent', 0) > 85:
                recommendations.append("High memory usage detected. Consider implementing more aggressive caching policies.")
            
            if self.system_metrics.get('disk_percent', 0) > 90:
                recommendations.append("Low disk space. Consider cleaning up temporary files and logs.")
            
            # Check operation performance
            for op_name, stats in self.operation_stats.items():
                if stats.avg_time > 5.0:  # 5 second threshold
                    recommendations.append(f"Operation '{op_name}' is slow (avg: {stats.avg_time:.2f}s). Consider optimization.")
                
                if stats.error_count / stats.call_count > 0.1:  # 10% error rate threshold
                    error_rate = stats.error_count / stats.call_count * 100
                    recommendations.append(f"Operation '{op_name}' has high error rate ({error_rate:.1f}%). Check error handling.")
        
        return recommendations
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        with self._lock:
            self.metrics_history.clear()
            self.operation_stats.clear()
            self.system_metrics.clear()
        logger.info("Performance metrics cleared")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def timed_operation(operation_name: str):
    """Decorator to time function execution and record metrics."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    execution_time = time.time() - start_time
                    performance_monitor.record_operation(operation_name, execution_time, success)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    execution_time = time.time() - start_time
                    performance_monitor.record_operation(operation_name, execution_time, success)
            return sync_wrapper
    return decorator


@asynccontextmanager
async def async_timer(operation_name: str):
    """Async context manager for timing operations."""
    start_time = time.time()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        execution_time = time.time() - start_time
        performance_monitor.record_operation(operation_name, execution_time, success)


@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.time()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        execution_time = time.time() - start_time
        performance_monitor.record_operation(operation_name, execution_time, success)


class PerformanceOptimizer:
    """Performance optimization utilities."""
    
    @staticmethod
    def optimize_chunk_size(document_length: int, target_chunks: int = 10) -> int:
        """Calculate optimal chunk size based on document length."""
        base_chunk_size = document_length // target_chunks
        
        # Apply constraints
        min_chunk_size = 200
        max_chunk_size = 2000
        
        optimal_size = max(min_chunk_size, min(max_chunk_size, base_chunk_size))
        
        # Round to nearest 50 for consistency
        return round(optimal_size / 50) * 50
    
    @staticmethod
    def calculate_optimal_batch_size(item_count: int, processing_time_per_item: float, 
                                   target_batch_time: float = 30.0) -> int:
        """Calculate optimal batch size for processing operations."""
        if processing_time_per_item <= 0:
            return min(item_count, 10)  # Default batch size
        
        optimal_batch_size = int(target_batch_time / processing_time_per_item)
        
        # Apply constraints
        min_batch_size = 1
        max_batch_size = 50
        
        return max(min_batch_size, min(max_batch_size, min(optimal_batch_size, item_count)))
    
    @staticmethod
    def should_use_cache(operation_frequency: float, cache_hit_rate: float, 
                        cache_overhead: float = 0.1) -> bool:
        """Determine if caching would be beneficial for an operation."""
        # Simple heuristic: cache if frequently used and has good hit rate
        benefit_threshold = cache_overhead * 2  # Must overcome overhead by 2x
        expected_benefit = operation_frequency * cache_hit_rate
        
        return expected_benefit > benefit_threshold
    
    @staticmethod
    def estimate_memory_usage(chunk_count: int, avg_chunk_size: int, 
                            embedding_dimension: int = 384) -> Dict[str, float]:
        """Estimate memory usage for vector storage."""
        # Text storage (assuming UTF-8, roughly 1 byte per character)
        text_memory_mb = (chunk_count * avg_chunk_size) / (1024 * 1024)
        
        # Embedding storage (float32 = 4 bytes)
        embedding_memory_mb = (chunk_count * embedding_dimension * 4) / (1024 * 1024)
        
        # Metadata overhead (estimated)
        metadata_memory_mb = chunk_count * 0.001  # 1KB per chunk metadata
        
        total_memory_mb = text_memory_mb + embedding_memory_mb + metadata_memory_mb
        
        return {
            "text_memory_mb": text_memory_mb,
            "embedding_memory_mb": embedding_memory_mb,
            "metadata_memory_mb": metadata_memory_mb,
            "total_memory_mb": total_memory_mb,
            "total_memory_gb": total_memory_mb / 1024
        }


class AdaptiveOptimizer:
    """Adaptive optimization that learns from performance metrics."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def suggest_chunk_size_optimization(self, current_chunk_size: int, 
                                      operation_name: str = "document_chunking") -> Optional[int]:
        """Suggest optimal chunk size based on performance history."""
        stats = self.monitor.get_operation_stats(operation_name).get(operation_name)
        
        if not stats or stats.call_count < 5:
            return None  # Not enough data
        
        # If performance is poor, suggest smaller chunks
        if stats.avg_time > 3.0:  # 3 second threshold
            suggested_size = int(current_chunk_size * 0.8)
            return max(200, suggested_size)  # Minimum 200 characters
        
        # If performance is very good, suggest larger chunks
        if stats.avg_time < 0.5 and stats.error_count == 0:
            suggested_size = int(current_chunk_size * 1.2)
            return min(2000, suggested_size)  # Maximum 2000 characters
        
        return None  # Current size is acceptable
    
    def suggest_batch_size_optimization(self, current_batch_size: int,
                                      operation_name: str) -> Optional[int]:
        """Suggest optimal batch size based on performance."""
        stats = self.monitor.get_operation_stats(operation_name).get(operation_name)
        
        if not stats or stats.call_count < 3:
            return None
        
        # If processing is slow, reduce batch size
        if stats.avg_time > 10.0:
            return max(1, int(current_batch_size * 0.7))
        
        # If processing is fast and error-free, increase batch size
        if stats.avg_time < 2.0 and stats.error_count == 0:
            return min(50, int(current_batch_size * 1.3))
        
        return None
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        summary = self.monitor.get_metrics_summary()
        recommendations = self.monitor.get_performance_recommendations()
        
        # Add adaptive suggestions
        adaptive_suggestions = []
        
        for op_name in summary.get("operations", {}):
            # Chunk size suggestions
            chunk_suggestion = self.suggest_chunk_size_optimization(1000, op_name)
            if chunk_suggestion:
                adaptive_suggestions.append(
                    f"Consider adjusting chunk size to {chunk_suggestion} for operation '{op_name}'"
                )
            
            # Batch size suggestions
            batch_suggestion = self.suggest_batch_size_optimization(10, op_name)
            if batch_suggestion:
                adaptive_suggestions.append(
                    f"Consider adjusting batch size to {batch_suggestion} for operation '{op_name}'"
                )
        
        return {
            "performance_summary": summary,
            "static_recommendations": recommendations,
            "adaptive_suggestions": adaptive_suggestions,
            "optimization_history": dict(self.optimization_history),
            "generated_at": datetime.utcnow().isoformat()
        }


# Global adaptive optimizer
adaptive_optimizer = AdaptiveOptimizer(performance_monitor)