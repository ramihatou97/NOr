# backend/core/workflow_intelligence.py
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from collections import defaultdict, deque

class TaskType(Enum):
    CREATIVE_WRITING = "creative_writing"
    RESEARCH = "research"
    EDITING = "editing"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    REFERENCE_MANAGEMENT = "reference_management"
    CONFLICT_RESOLUTION = "conflict_resolution"
    QUALITY_REVIEW = "quality_review"

class CognitiveLoadLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class WorkflowTask:
    task_id: str
    task_type: TaskType
    description: str
    estimated_duration: timedelta
    cognitive_load: CognitiveLoadLevel
    priority: TaskPriority
    dependencies: List[str]
    optimal_conditions: Dict[str, Any]
    deadline: Optional[datetime]
    context_requirements: Dict[str, Any]

@dataclass
class ProductivityPattern:
    user_id: str
    peak_hours: List[int]
    low_energy_hours: List[int]
    optimal_task_sequence: List[TaskType]
    context_switch_cost: float
    sustained_focus_duration: timedelta
    preferred_break_patterns: List[Dict[str, Any]]

@dataclass
class WorkflowOptimization:
    optimized_schedule: List[Dict[str, Any]]
    predicted_productivity: float
    energy_management: Dict[str, Any]
    context_switching_plan: List[Dict[str, Any]]
    break_recommendations: List[Dict[str, Any]]
    resource_allocation: Dict[str, float]

class WorkflowIntelligenceManager:
    def __init__(self):
        self.user_patterns = {}
        self.task_queue = deque()
        self.completed_tasks = deque(maxlen=1000)
        self.productivity_metrics = defaultdict(list)
        self.context_switching_costs = defaultdict(float)
        self.energy_patterns = defaultdict(list)

    async def optimize_workflow(self, user_id: str, current_context: Dict[str, Any],
                              available_time: timedelta) -> WorkflowOptimization:
        """Generate optimal workflow for the user based on patterns and context"""

        # Get user productivity patterns
        user_pattern = await self._get_user_patterns(user_id)

        # Analyze current context
        context_analysis = await self._analyze_current_context(current_context)

        # Get pending tasks
        pending_tasks = await self._get_pending_tasks(user_id)

        # Optimize task scheduling
        optimized_schedule = await self._optimize_task_schedule(
            pending_tasks, user_pattern, context_analysis, available_time
        )

        # Predict productivity
        predicted_productivity = await self._predict_session_productivity(
            optimized_schedule, user_pattern, context_analysis
        )

        # Generate energy management plan
        energy_management = await self._generate_energy_management_plan(
            optimized_schedule, user_pattern
        )

        # Plan context switches
        context_switching_plan = await self._plan_context_switches(
            optimized_schedule, user_pattern
        )

        # Recommend breaks
        break_recommendations = await self._recommend_breaks(
            optimized_schedule, user_pattern, available_time
        )

        # Allocate resources
        resource_allocation = await self._allocate_resources(
            optimized_schedule, context_analysis
        )

        return WorkflowOptimization(
            optimized_schedule=optimized_schedule,
            predicted_productivity=predicted_productivity,
            energy_management=energy_management,
            context_switching_plan=context_switching_plan,
            break_recommendations=break_recommendations,
            resource_allocation=resource_allocation
        )

    async def adaptive_task_scheduling(self, user_id: str,
                                     real_time_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adaptively schedule tasks based on real-time user state"""

        # Assess current cognitive state
        cognitive_state = await self._assess_cognitive_state(real_time_context)

        # Get appropriate tasks for current state
        suitable_tasks = await self._get_tasks_for_cognitive_state(user_id, cognitive_state)

        # Prioritize based on urgency and cognitive fit
        prioritized_tasks = await self._prioritize_tasks_by_cognitive_fit(
            suitable_tasks, cognitive_state
        )

        # Generate adaptive schedule
        adaptive_schedule = await self._generate_adaptive_schedule(
            prioritized_tasks, cognitive_state, real_time_context
        )

        return adaptive_schedule

    async def intelligent_interruption_management(self, user_id: str,
                                                interruption: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently manage interruptions to minimize productivity loss"""

        # Assess current task context
        current_task = await self._get_current_task(user_id)

        # Calculate interruption cost
        interruption_cost = await self._calculate_interruption_cost(
            current_task, interruption
        )

        # Determine optimal handling strategy
        handling_strategy = await self._determine_interruption_strategy(
            current_task, interruption, interruption_cost
        )

        # Generate recommendations
        recommendations = await self._generate_interruption_recommendations(
            handling_strategy, current_task, interruption
        )

        return {
            "should_interrupt": handling_strategy["should_interrupt"],
            "delay_recommendation": handling_strategy.get("delay_duration"),
            "context_preservation": handling_strategy["context_preservation"],
            "resumption_strategy": handling_strategy["resumption_strategy"],
            "recommendations": recommendations
        }

    async def learn_from_session(self, user_id: str, session_data: Dict[str, Any]):
        """Learn from completed work session to improve future optimization"""

        # Extract productivity metrics
        productivity_metrics = await self._extract_productivity_metrics(session_data)

        # Analyze task performance
        task_performance = await self._analyze_task_performance(session_data)

        # Update user patterns
        await self._update_user_patterns(user_id, productivity_metrics, task_performance)

        # Learn context switching costs
        await self._learn_context_switching_costs(user_id, session_data)

        # Update energy patterns
        await self._update_energy_patterns(user_id, session_data)

        # Improve prediction models
        await self._improve_prediction_models(user_id, session_data)

    async def _get_user_patterns(self, user_id: str) -> ProductivityPattern:
        """Get or initialize user productivity patterns"""

        if user_id not in self.user_patterns:
            # Initialize with default patterns
            self.user_patterns[user_id] = ProductivityPattern(
                user_id=user_id,
                peak_hours=[9, 10, 14, 15],  # Default peak hours
                low_energy_hours=[13, 16, 17],  # Default low energy hours
                optimal_task_sequence=[
                    TaskType.RESEARCH,
                    TaskType.CREATIVE_WRITING,
                    TaskType.EDITING,
                    TaskType.VALIDATION
                ],
                context_switch_cost=0.3,  # 30% productivity loss
                sustained_focus_duration=timedelta(minutes=45),
                preferred_break_patterns=[
                    {"type": "short", "duration": timedelta(minutes=5), "frequency": 45},
                    {"type": "medium", "duration": timedelta(minutes=15), "frequency": 120}
                ]
            )

        return self.user_patterns[user_id]

    async def _analyze_current_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current work context"""

        analysis = {
            "cognitive_load": context.get("cognitive_load", 0.5),
            "time_of_day": datetime.now().hour,
            "available_time": context.get("available_time", timedelta(hours=2)),
            "current_chapter": context.get("current_chapter"),
            "recent_activity": context.get("recent_activity", []),
            "external_factors": context.get("external_factors", {}),
            "energy_level": await self._estimate_energy_level(context),
            "focus_quality": await self._estimate_focus_quality(context)
        }

        return analysis

    async def _optimize_task_schedule(self, tasks: List[WorkflowTask],
                                    user_pattern: ProductivityPattern,
                                    context: Dict[str, Any],
                                    available_time: timedelta) -> List[Dict[str, Any]]:
        """Optimize task scheduling for maximum productivity"""

        # Filter tasks by available time
        feasible_tasks = [task for task in tasks
                         if task.estimated_duration <= available_time]

        # Sort by optimal sequence and priority
        sorted_tasks = await self._sort_tasks_optimally(
            feasible_tasks, user_pattern, context
        )

        # Schedule tasks with timing optimization
        schedule = []
        current_time = datetime.now()
        remaining_time = available_time

        for task in sorted_tasks:
            if task.estimated_duration <= remaining_time:
                # Check if current time is optimal for this task
                optimal_start_time = await self._find_optimal_start_time(
                    task, current_time, user_pattern, context
                )

                schedule_item = {
                    "task": task,
                    "start_time": optimal_start_time,
                    "end_time": optimal_start_time + task.estimated_duration,
                    "productivity_prediction": await self._predict_task_productivity(
                        task, optimal_start_time, user_pattern, context
                    ),
                    "cognitive_load_impact": await self._calculate_cognitive_load_impact(
                        task, context
                    ),
                    "preparation_requirements": await self._get_preparation_requirements(task)
                }

                schedule.append(schedule_item)
                current_time = schedule_item["end_time"]
                remaining_time -= task.estimated_duration

        return schedule

    async def _assess_cognitive_state(self, real_time_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current cognitive state from real-time indicators"""

        # Analyze typing patterns
        typing_speed = real_time_context.get("typing_speed", 0)
        typing_consistency = real_time_context.get("typing_consistency", 0)

        # Analyze interaction patterns
        click_precision = real_time_context.get("click_precision", 0)
        navigation_efficiency = real_time_context.get("navigation_efficiency", 0)

        # Analyze work patterns
        task_switching_frequency = real_time_context.get("task_switching_frequency", 0)
        focus_duration = real_time_context.get("focus_duration", 0)

        # Calculate cognitive state scores
        attention_level = (typing_consistency + click_precision + focus_duration) / 3
        processing_speed = (typing_speed + navigation_efficiency) / 2
        mental_fatigue = min(1.0, task_switching_frequency * 0.5)

        cognitive_state = {
            "attention_level": attention_level,
            "processing_speed": processing_speed,
            "mental_fatigue": mental_fatigue,
            "overall_cognitive_capacity": (attention_level + processing_speed) * (1 - mental_fatigue),
            "recommended_task_complexity": await self._recommend_task_complexity(
                attention_level, processing_speed, mental_fatigue
            )
        }

        return cognitive_state

    async def _get_tasks_for_cognitive_state(self, user_id: str,
                                           cognitive_state: Dict[str, Any]) -> List[WorkflowTask]:
        """Get tasks appropriate for current cognitive state"""

        cognitive_capacity = cognitive_state["overall_cognitive_capacity"]

        # Get all pending tasks
        all_tasks = await self._get_pending_tasks(user_id)

        # Filter by cognitive requirements
        suitable_tasks = []
        for task in all_tasks:
            task_cognitive_requirement = await self._get_task_cognitive_requirement(task)

            # Match task complexity to cognitive capacity
            if self._is_cognitive_match(task_cognitive_requirement, cognitive_capacity):
                suitable_tasks.append(task)

        return suitable_tasks

    async def _calculate_interruption_cost(self, current_task: Optional[WorkflowTask],
                                         interruption: Dict[str, Any]) -> float:
        """Calculate the cost of interrupting current work"""

        if not current_task:
            return 0.0

        cost_factors = {
            "task_complexity": 0.0,
            "current_progress": 0.0,
            "context_depth": 0.0,
            "resumption_difficulty": 0.0
        }

        # Task complexity factor
        if current_task.cognitive_load == CognitiveLoadLevel.HIGH:
            cost_factors["task_complexity"] = 0.8
        elif current_task.cognitive_load == CognitiveLoadLevel.MEDIUM:
            cost_factors["task_complexity"] = 0.5
        else:
            cost_factors["task_complexity"] = 0.2

        # Current progress factor (more progress = higher interruption cost)
        task_progress = interruption.get("task_progress", 0.5)
        cost_factors["current_progress"] = task_progress * 0.7

        # Context depth factor
        context_depth = interruption.get("context_depth", 0.5)
        cost_factors["context_depth"] = context_depth * 0.6

        # Resumption difficulty
        if current_task.task_type == TaskType.CREATIVE_WRITING:
            cost_factors["resumption_difficulty"] = 0.9
        elif current_task.task_type == TaskType.SYNTHESIS:
            cost_factors["resumption_difficulty"] = 0.7
        else:
            cost_factors["resumption_difficulty"] = 0.4

        # Calculate weighted interruption cost
        weights = {
            "task_complexity": 0.3,
            "current_progress": 0.2,
            "context_depth": 0.2,
            "resumption_difficulty": 0.3
        }

        total_cost = sum(cost_factors[factor] * weights[factor]
                        for factor in cost_factors)

        return min(1.0, total_cost)

    async def _determine_interruption_strategy(self, current_task: Optional[WorkflowTask],
                                             interruption: Dict[str, Any],
                                             interruption_cost: float) -> Dict[str, Any]:
        """Determine optimal strategy for handling interruption"""

        interruption_urgency = interruption.get("urgency", 0.5)
        interruption_importance = interruption.get("importance", 0.5)

        # Decision matrix
        if interruption_urgency > 0.8 and interruption_importance > 0.7:
            # Critical interruption - must handle immediately
            strategy = {
                "should_interrupt": True,
                "context_preservation": "full",
                "resumption_strategy": "immediate_context_restore"
            }
        elif interruption_cost < 0.3:
            # Low cost interruption - safe to handle
            strategy = {
                "should_interrupt": True,
                "context_preservation": "minimal",
                "resumption_strategy": "natural_transition"
            }
        elif interruption_urgency > 0.6:
            # Moderate urgency - delay until natural break point
            delay_duration = await self._calculate_optimal_delay(current_task)
            strategy = {
                "should_interrupt": False,
                "delay_duration": delay_duration,
                "context_preservation": "checkpoint",
                "resumption_strategy": "checkpoint_restore"
            }
        else:
            # Low urgency - defer until task completion or scheduled break
            strategy = {
                "should_interrupt": False,
                "defer_until": "task_completion",
                "context_preservation": "none",
                "resumption_strategy": "new_context"
            }

        return strategy

    async def predict_optimal_work_session(self, user_id: str,
                                         session_goals: List[str],
                                         constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal work session structure"""

        user_pattern = await self._get_user_patterns(user_id)

        # Analyze session requirements
        session_analysis = await self._analyze_session_requirements(session_goals, constraints)

        # Generate optimal session structure
        session_structure = {
            "warm_up_period": await self._design_warm_up_period(user_pattern, session_analysis),
            "core_work_blocks": await self._design_core_work_blocks(user_pattern, session_analysis),
            "transition_periods": await self._design_transition_periods(user_pattern),
            "break_schedule": await self._design_break_schedule(user_pattern, session_analysis),
            "cool_down_period": await self._design_cool_down_period(user_pattern),
            "contingency_plans": await self._design_contingency_plans(session_analysis)
        }

        # Predict session outcomes
        predicted_outcomes = {
            "productivity_score": await self._predict_session_productivity_score(session_structure),
            "completion_probability": await self._predict_goal_completion_probability(
                session_goals, session_structure
            ),
            "energy_depletion": await self._predict_energy_depletion(session_structure),
            "quality_expectations": await self._predict_output_quality(session_structure)
        }

        return {
            "session_structure": session_structure,
            "predicted_outcomes": predicted_outcomes,
            "optimization_recommendations": await self._generate_optimization_recommendations(
                session_structure, predicted_outcomes
            )
        }

# Global workflow intelligence instance
workflow_intelligence = WorkflowIntelligenceManager()