# backend/core/performance_optimizer.py
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import psutil
import time
from collections import defaultdict, deque
import numpy as np

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK = "network"
    GPU = "gpu"
    DATABASE = "database"
    CACHE = "cache"
    AI_APIS = "ai_apis"

class OptimizationStrategy(Enum):
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"

class TaskComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ResourceMetrics:
    resource_type: ResourceType
    current_usage: float
    average_usage: float
    peak_usage: float
    available_capacity: float
    bottleneck_score: float
    optimization_potential: float
    last_updated: datetime

@dataclass
class PerformanceProfile:
    profile_id: str
    hardware_specs: Dict[str, Any]
    optimal_configurations: Dict[str, Any]
    resource_limits: Dict[ResourceType, float]
    performance_targets: Dict[str, float]
    optimization_preferences: Dict[str, Any]

@dataclass
class OptimizationRecommendation:
    recommendation_id: str
    resource_type: ResourceType
    current_state: Dict[str, Any]
    recommended_changes: Dict[str, Any]
    expected_improvement: float
    implementation_cost: float
    priority_score: float
    implementation_steps: List[str]

class PerformanceIntelligenceOptimizer:
    def __init__(self):
        self.resource_monitors = {}
        self.performance_history = deque(maxlen=10000)
        self.optimization_results = defaultdict(list)
        self.system_profile = None
        self.adaptive_thresholds = defaultdict(float)
        self.prediction_models = {}

        # Initialize monitoring (lazy initialization)
        self._monitoring_initialized = False

    async def _ensure_monitoring_initialized(self):
        """Ensure monitoring is initialized (lazy initialization)"""
        if not self._monitoring_initialized:
            await self._initialize_monitoring()
            self._monitoring_initialized = True

    async def optimize_system_performance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive system performance optimization"""

        # Ensure monitoring is initialized
        await self._ensure_monitoring_initialized()

        # Step 1: Collect current metrics
        current_metrics = await self._collect_comprehensive_metrics()

        # Step 2: Analyze performance bottlenecks
        bottlenecks = await self._identify_bottlenecks(current_metrics)

        # Step 3: Generate optimization recommendations
        recommendations = await self._generate_optimization_recommendations(
            current_metrics, bottlenecks, context
        )

        # Step 4: Apply safe optimizations automatically
        auto_applied = await self._apply_automatic_optimizations(recommendations)

        # Step 5: Predict performance impact
        predicted_impact = await self._predict_optimization_impact(recommendations)

        # Step 6: Resource allocation optimization
        resource_allocation = await self._optimize_resource_allocation(current_metrics, context)

        # Step 7: Caching strategy optimization
        cache_optimization = await self._optimize_caching_strategy(current_metrics, context)

        optimization_result = {
            "current_performance": current_metrics,
            "identified_bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "auto_applied_optimizations": auto_applied,
            "predicted_improvements": predicted_impact,
            "resource_allocation": resource_allocation,
            "cache_optimization": cache_optimization,
            "next_review_time": datetime.now() + timedelta(hours=1)
        }

        # Store for learning
        await self._store_optimization_results(optimization_result, context)

        return optimization_result

    async def adaptive_resource_management(self, workload_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptively manage resources based on predicted workload"""

        # Analyze predicted workload
        workload_analysis = await self._analyze_predicted_workload(workload_prediction)

        # Calculate resource requirements
        resource_requirements = await self._calculate_resource_requirements(workload_analysis)

        # Optimize resource allocation
        allocation_plan = await self._create_allocation_plan(resource_requirements)

        # Pre-allocate critical resources
        pre_allocation = await self._pre_allocate_resources(allocation_plan)

        # Set up dynamic scaling
        scaling_config = await self._configure_dynamic_scaling(workload_analysis)

        # Monitor and adjust
        monitoring_plan = await self._create_monitoring_plan(allocation_plan)

        return {
            "workload_analysis": workload_analysis,
            "resource_requirements": resource_requirements,
            "allocation_plan": allocation_plan,
            "pre_allocation_status": pre_allocation,
            "scaling_configuration": scaling_config,
            "monitoring_plan": monitoring_plan
        }

    async def intelligent_caching_optimization(self, access_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize caching based on access patterns and content types"""

        # Analyze access patterns
        pattern_analysis = await self._analyze_access_patterns(access_patterns)

        # Identify high-value cache candidates
        cache_candidates = await self._identify_cache_candidates(pattern_analysis)

        # Optimize cache hierarchy
        cache_hierarchy = await self._optimize_cache_hierarchy(cache_candidates)

        # Configure cache policies
        cache_policies = await self._configure_cache_policies(pattern_analysis)

        # Pre-warm caches
        prewarming_plan = await self._create_cache_prewarming_plan(cache_candidates)

        # Set up cache invalidation strategies
        invalidation_strategy = await self._configure_cache_invalidation(pattern_analysis)

        return {
            "pattern_analysis": pattern_analysis,
            "cache_candidates": cache_candidates,
            "cache_hierarchy": cache_hierarchy,
            "cache_policies": cache_policies,
            "prewarming_plan": prewarming_plan,
            "invalidation_strategy": invalidation_strategy
        }

    async def _collect_comprehensive_metrics(self) -> Dict[str, ResourceMetrics]:
        """Collect comprehensive system metrics"""

        metrics = {}

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]

        metrics[ResourceType.CPU] = ResourceMetrics(
            resource_type=ResourceType.CPU,
            current_usage=cpu_percent / 100.0,
            average_usage=await self._get_average_usage(ResourceType.CPU),
            peak_usage=await self._get_peak_usage(ResourceType.CPU),
            available_capacity=(100 - cpu_percent) / 100.0,
            bottleneck_score=await self._calculate_bottleneck_score(ResourceType.CPU, cpu_percent),
            optimization_potential=await self._calculate_optimization_potential(ResourceType.CPU),
            last_updated=datetime.now()
        )

        # Memory metrics
        memory = psutil.virtual_memory()
        metrics[ResourceType.MEMORY] = ResourceMetrics(
            resource_type=ResourceType.MEMORY,
            current_usage=memory.percent / 100.0,
            average_usage=await self._get_average_usage(ResourceType.MEMORY),
            peak_usage=await self._get_peak_usage(ResourceType.MEMORY),
            available_capacity=memory.available / memory.total,
            bottleneck_score=await self._calculate_bottleneck_score(ResourceType.MEMORY, memory.percent),
            optimization_potential=await self._calculate_optimization_potential(ResourceType.MEMORY),
            last_updated=datetime.now()
        )

        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        if disk_io:
            disk_usage = await self._calculate_disk_usage_percentage()
            metrics[ResourceType.DISK_IO] = ResourceMetrics(
                resource_type=ResourceType.DISK_IO,
                current_usage=disk_usage,
                average_usage=await self._get_average_usage(ResourceType.DISK_IO),
                peak_usage=await self._get_peak_usage(ResourceType.DISK_IO),
                available_capacity=1.0 - disk_usage,
                bottleneck_score=await self._calculate_bottleneck_score(ResourceType.DISK_IO, disk_usage * 100),
                optimization_potential=await self._calculate_optimization_potential(ResourceType.DISK_IO),
                last_updated=datetime.now()
            )

        # Database metrics
        db_metrics = await self._collect_database_metrics()
        if db_metrics:
            metrics[ResourceType.DATABASE] = db_metrics

        # Cache metrics
        cache_metrics = await self._collect_cache_metrics()
        if cache_metrics:
            metrics[ResourceType.CACHE] = cache_metrics

        # AI API metrics
        ai_metrics = await self._collect_ai_api_metrics()
        if ai_metrics:
            metrics[ResourceType.AI_APIS] = ai_metrics

        return metrics

    async def _identify_bottlenecks(self, metrics: Dict[str, ResourceMetrics]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from metrics"""

        bottlenecks = []

        for resource_type, metric in metrics.items():
            if metric.bottleneck_score > 0.7:  # High bottleneck threshold
                bottleneck = {
                    "resource_type": resource_type,
                    "severity": "high" if metric.bottleneck_score > 0.9 else "medium",
                    "current_usage": metric.current_usage,
                    "bottleneck_score": metric.bottleneck_score,
                    "impact_analysis": await self._analyze_bottleneck_impact(resource_type, metric),
                    "root_causes": await self._identify_root_causes(resource_type, metric),
                    "affected_operations": await self._identify_affected_operations(resource_type)
                }
                bottlenecks.append(bottleneck)

        # Sort by severity and impact
        bottlenecks.sort(key=lambda x: x["bottleneck_score"], reverse=True)

        return bottlenecks

    async def _generate_optimization_recommendations(self, metrics: Dict[str, ResourceMetrics],
                                                   bottlenecks: List[Dict[str, Any]],
                                                   context: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate specific optimization recommendations"""

        recommendations = []

        # CPU optimization recommendations
        if ResourceType.CPU in metrics:
            cpu_recommendations = await self._generate_cpu_optimizations(
                metrics[ResourceType.CPU], context
            )
            recommendations.extend(cpu_recommendations)

        # Memory optimization recommendations
        if ResourceType.MEMORY in metrics:
            memory_recommendations = await self._generate_memory_optimizations(
                metrics[ResourceType.MEMORY], context
            )
            recommendations.extend(memory_recommendations)

        # Database optimization recommendations
        if ResourceType.DATABASE in metrics:
            db_recommendations = await self._generate_database_optimizations(
                metrics[ResourceType.DATABASE], context
            )
            recommendations.extend(db_recommendations)

        # Cache optimization recommendations
        if ResourceType.CACHE in metrics:
            cache_recommendations = await self._generate_cache_optimizations(
                metrics[ResourceType.CACHE], context
            )
            recommendations.extend(cache_recommendations)

        # AI API optimization recommendations
        if ResourceType.AI_APIS in metrics:
            ai_recommendations = await self._generate_ai_api_optimizations(
                metrics[ResourceType.AI_APIS], context
            )
            recommendations.extend(ai_recommendations)

        # Sort by priority and expected impact
        recommendations.sort(
            key=lambda x: x.priority_score * x.expected_improvement,
            reverse=True
        )

        return recommendations

    async def _generate_cpu_optimizations(self, cpu_metrics: ResourceMetrics,
                                        context: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate CPU-specific optimization recommendations"""

        recommendations = []

        if cpu_metrics.current_usage > 0.8:  # High CPU usage
            # Process prioritization recommendation
            if context.get("has_background_tasks", True):
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"cpu_process_priority_{datetime.now().timestamp()}",
                    resource_type=ResourceType.CPU,
                    current_state={"cpu_usage": cpu_metrics.current_usage},
                    recommended_changes={
                        "action": "adjust_process_priorities",
                        "background_task_priority": "low",
                        "ai_task_priority": "high",
                        "user_task_priority": "realtime"
                    },
                    expected_improvement=0.2,
                    implementation_cost=0.1,
                    priority_score=0.9,
                    implementation_steps=[
                        "Identify CPU-intensive background tasks",
                        "Lower priority of non-critical processes",
                        "Implement task queuing for background operations"
                    ]
                ))

            # Async processing recommendation
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"cpu_async_processing_{datetime.now().timestamp()}",
                resource_type=ResourceType.CPU,
                current_state={"cpu_usage": cpu_metrics.current_usage},
                recommended_changes={
                    "action": "increase_async_processing",
                    "ai_api_calls": "async",
                    "file_operations": "async",
                    "database_queries": "async"
                },
                expected_improvement=0.3,
                implementation_cost=0.2,
                priority_score=0.8,
                implementation_steps=[
                    "Convert synchronous AI API calls to async",
                    "Implement async file processing",
                    "Use async database connections"
                ]
            ))

        return recommendations

    async def _generate_memory_optimizations(self, memory_metrics: ResourceMetrics,
                                           context: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate memory-specific optimization recommendations"""

        recommendations = []

        if memory_metrics.current_usage > 0.75:  # High memory usage
            # Memory cleanup recommendation
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"memory_cleanup_{datetime.now().timestamp()}",
                resource_type=ResourceType.MEMORY,
                current_state={"memory_usage": memory_metrics.current_usage},
                recommended_changes={
                    "action": "implement_memory_cleanup",
                    "cache_cleanup_interval": 300,  # 5 minutes
                    "garbage_collection_frequency": "aggressive",
                    "object_pooling": "enabled"
                },
                expected_improvement=0.25,
                implementation_cost=0.1,
                priority_score=0.9,
                implementation_steps=[
                    "Implement automatic cache cleanup",
                    "Enable aggressive garbage collection",
                    "Add object pooling for frequently created objects"
                ]
            ))

            # Data structure optimization
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"memory_data_structures_{datetime.now().timestamp()}",
                resource_type=ResourceType.MEMORY,
                current_state={"memory_usage": memory_metrics.current_usage},
                recommended_changes={
                    "action": "optimize_data_structures",
                    "use_generators": True,
                    "lazy_loading": True,
                    "memory_mapped_files": True
                },
                expected_improvement=0.2,
                implementation_cost=0.3,
                priority_score=0.7,
                implementation_steps=[
                    "Replace lists with generators where possible",
                    "Implement lazy loading for large datasets",
                    "Use memory-mapped files for large file operations"
                ]
            ))

        return recommendations

    async def _apply_automatic_optimizations(self, recommendations: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Apply safe optimizations automatically"""

        auto_applied = []

        for recommendation in recommendations:
            # Only apply low-cost, high-impact optimizations automatically
            if (recommendation.implementation_cost < 0.3 and
                recommendation.expected_improvement > 0.2 and
                recommendation.priority_score > 0.7):

                try:
                    result = await self._apply_optimization(recommendation)
                    if result["success"]:
                        auto_applied.append({
                            "recommendation_id": recommendation.recommendation_id,
                            "applied_changes": recommendation.recommended_changes,
                            "result": result,
                            "applied_at": datetime.now()
                        })
                except Exception as e:
                    # Log error but continue with other optimizations
                    print(f"Failed to apply optimization {recommendation.recommendation_id}: {e}")

        return auto_applied

    async def _apply_optimization(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """Apply a specific optimization recommendation"""

        result = {"success": False, "details": {}}

        try:
            if recommendation.resource_type == ResourceType.CPU:
                result = await self._apply_cpu_optimization(recommendation)
            elif recommendation.resource_type == ResourceType.MEMORY:
                result = await self._apply_memory_optimization(recommendation)
            elif recommendation.resource_type == ResourceType.CACHE:
                result = await self._apply_cache_optimization(recommendation)
            elif recommendation.resource_type == ResourceType.DATABASE:
                result = await self._apply_database_optimization(recommendation)

        except Exception as e:
            result = {"success": False, "error": str(e)}

        return result

    async def predict_performance_trends(self, time_horizon: timedelta) -> Dict[str, Any]:
        """Predict performance trends over specified time horizon"""

        # Analyze historical data
        historical_analysis = await self._analyze_historical_performance()

        # Identify patterns and cycles
        patterns = await self._identify_performance_patterns(historical_analysis)

        # Build prediction models
        prediction_models = await self._build_prediction_models(historical_analysis, patterns)

        # Generate predictions
        predictions = {}
        for resource_type in ResourceType:
            if resource_type in prediction_models:
                predictions[resource_type] = await self._predict_resource_usage(
                    resource_type, time_horizon, prediction_models[resource_type]
                )

        # Identify potential issues
        predicted_issues = await self._predict_performance_issues(predictions, time_horizon)

        # Generate proactive recommendations
        proactive_recommendations = await self._generate_proactive_recommendations(
            predictions, predicted_issues
        )

        return {
            "time_horizon": time_horizon,
            "predictions": predictions,
            "predicted_issues": predicted_issues,
            "proactive_recommendations": proactive_recommendations,
            "confidence_levels": await self._calculate_prediction_confidence(predictions),
            "update_frequency": await self._recommend_update_frequency(predictions)
        }

    async def _initialize_monitoring(self):
        """Initialize continuous performance monitoring"""

        while True:
            try:
                # Collect metrics
                metrics = await self._collect_lightweight_metrics()

                # Store in history
                self.performance_history.append({
                    "timestamp": datetime.now(),
                    "metrics": metrics
                })

                # Check for anomalies
                anomalies = await self._detect_performance_anomalies(metrics)
                if anomalies:
                    await self._handle_performance_anomalies(anomalies)

                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds

            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error

# Global performance optimizer instance
performance_optimizer = PerformanceIntelligenceOptimizer()