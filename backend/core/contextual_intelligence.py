# backend/core/contextual_intelligence.py
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum

class ExpertiseLevel(Enum):
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"
    SPECIALIST = "specialist"

class WorkflowState(Enum):
    RESEARCH = "research"
    WRITING = "writing"
    REVIEW = "review"
    SYNTHESIS = "synthesis"

@dataclass
class UserContext:
    current_chapter: Optional[str]
    session_focus: str
    expertise_map: Dict[str, ExpertiseLevel]
    workflow_state: WorkflowState
    urgency_level: int  # 1-5
    cognitive_load: float  # 0.0-1.0
    preferred_sources: List[str]
    time_constraints: Optional[datetime]

@dataclass
class EnhancedQuery:
    original_query: str
    contextual_expansion: str
    predicted_intent: str
    expertise_calibration: ExpertiseLevel
    priority_score: float
    resource_allocation: Dict[str, float]

class ContextualIntelligenceEngine:
    def __init__(self):
        self.user_context = UserContext(
            current_chapter=None,
            session_focus="",
            expertise_map={},
            workflow_state=WorkflowState.RESEARCH,
            urgency_level=3,
            cognitive_load=0.5,
            preferred_sources=[],
            time_constraints=None
        )

        self.session_history = []
        self.learning_patterns = {}
        self.prediction_cache = {}

    async def update_context(self, activity_data: Dict[str, Any]):
        """Continuously update user context based on activities"""

        # Track current work
        if activity_data.get("chapter_id"):
            self.user_context.current_chapter = activity_data["chapter_id"]

        # Detect workflow state
        if activity_data.get("action") == "search":
            self.user_context.workflow_state = WorkflowState.RESEARCH
        elif activity_data.get("action") == "write":
            self.user_context.workflow_state = WorkflowState.WRITING
        elif activity_data.get("action") == "review":
            self.user_context.workflow_state = WorkflowState.REVIEW

        # Calculate cognitive load
        self.user_context.cognitive_load = await self._calculate_cognitive_load(activity_data)

        # Update expertise mapping
        await self._update_expertise_map(activity_data)

        # Store for pattern learning
        self.session_history.append({
            "timestamp": datetime.now(),
            "context": self.user_context.__dict__.copy(),
            "activity": activity_data
        })

    async def enhance_query(self, query: str, task_context: Dict[str, Any]) -> EnhancedQuery:
        """Transform simple queries into contextually rich requests"""

        # Analyze query intent
        intent_analysis = await self._analyze_query_intent(query, task_context)

        # Expand query with context
        contextual_expansion = await self._expand_with_context(query, intent_analysis)

        # Calibrate for expertise level
        domain = await self._detect_medical_domain(query)
        expertise_level = self.user_context.expertise_map.get(domain, ExpertiseLevel.INTERMEDIATE)

        # Calculate priority and resource allocation
        priority_score = await self._calculate_priority(query, task_context)
        resource_allocation = await self._optimize_resource_allocation(priority_score, expertise_level)

        return EnhancedQuery(
            original_query=query,
            contextual_expansion=contextual_expansion,
            predicted_intent=intent_analysis["primary_intent"],
            expertise_calibration=expertise_level,
            priority_score=priority_score,
            resource_allocation=resource_allocation
        )

    async def predict_next_needs(self) -> Dict[str, Any]:
        """Predict what user will need next based on patterns"""

        current_patterns = await self._analyze_current_patterns()
        historical_patterns = await self._analyze_historical_patterns()

        predictions = {
            "next_likely_queries": await self._predict_queries(current_patterns),
            "upcoming_sections": await self._predict_content_needs(current_patterns),
            "resource_requirements": await self._predict_resource_needs(current_patterns),
            "optimal_timing": await self._predict_optimal_work_times(historical_patterns),
            "quality_concerns": await self._predict_quality_issues(current_patterns)
        }

        return predictions

    async def _calculate_cognitive_load(self, activity_data: Dict[str, Any]) -> float:
        """Calculate current cognitive load based on activity patterns"""

        factors = {
            "task_complexity": activity_data.get("complexity", 0.5),
            "time_pressure": min(1.0, activity_data.get("urgency", 3) / 5.0),
            "multitasking": len(activity_data.get("concurrent_tasks", [])) * 0.2,
            "session_duration": min(1.0, activity_data.get("session_minutes", 30) / 120.0)
        }

        # Weighted cognitive load calculation
        load = (
            factors["task_complexity"] * 0.4 +
            factors["time_pressure"] * 0.3 +
            factors["multitasking"] * 0.2 +
            factors["session_duration"] * 0.1
        )

        return min(1.0, load)

    async def _update_expertise_map(self, activity_data: Dict[str, Any]):
        """Update expertise mapping based on user decisions and corrections"""

        if "domain" in activity_data and "outcome" in activity_data:
            domain = activity_data["domain"]
            outcome_quality = activity_data["outcome"].get("quality", 0.5)

            current_level = self.user_context.expertise_map.get(domain, ExpertiseLevel.INTERMEDIATE)

            # Adjust expertise based on successful outcomes
            if outcome_quality > 0.8:
                if current_level == ExpertiseLevel.NOVICE:
                    self.user_context.expertise_map[domain] = ExpertiseLevel.INTERMEDIATE
                elif current_level == ExpertiseLevel.INTERMEDIATE:
                    self.user_context.expertise_map[domain] = ExpertiseLevel.EXPERT
            elif outcome_quality < 0.4:
                # Lower confidence in expertise if poor outcomes
                if current_level == ExpertiseLevel.EXPERT:
                    self.user_context.expertise_map[domain] = ExpertiseLevel.INTERMEDIATE

    async def _analyze_query_intent(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the underlying intent behind a query"""

        # Intent classification patterns
        intent_patterns = {
            "research": ["search", "find", "literature", "studies", "papers"],
            "synthesis": ["combine", "synthesize", "summary", "overview"],
            "validation": ["check", "verify", "validate", "confirm"],
            "creation": ["write", "create", "draft", "compose"],
            "analysis": ["analyze", "compare", "evaluate", "assess"]
        }

        # Score each intent
        intent_scores = {}
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword.lower() in query.lower())
            intent_scores[intent] = score

        primary_intent = max(intent_scores, key=intent_scores.get)

        return {
            "primary_intent": primary_intent,
            "intent_scores": intent_scores,
            "confidence": max(intent_scores.values()) / len(query.split())
        }

    async def _expand_with_context(self, query: str, intent_analysis: Dict[str, Any]) -> str:
        """Expand query with contextual information"""

        expansions = []

        # Add current chapter context
        if self.user_context.current_chapter:
            expansions.append(f"related to {self.user_context.current_chapter}")

        # Add expertise context
        if intent_analysis["primary_intent"] in ["research", "validation"]:
            expertise_domains = [domain for domain, level in self.user_context.expertise_map.items()
                               if level in [ExpertiseLevel.EXPERT, ExpertiseLevel.SPECIALIST]]
            if expertise_domains:
                expansions.append(f"focus on {', '.join(expertise_domains)}")

        # Add urgency context
        if self.user_context.urgency_level >= 4:
            expansions.append("high priority recent findings")

        # Add workflow context
        if self.user_context.workflow_state == WorkflowState.WRITING:
            expansions.append("for content creation")
        elif self.user_context.workflow_state == WorkflowState.REVIEW:
            expansions.append("for validation and review")

        expanded_query = query
        if expansions:
            expanded_query += " (" + ", ".join(expansions) + ")"

        return expanded_query

    async def _detect_medical_domain(self, query: str) -> str:
        """Detect the medical domain/specialty of a query"""

        domain_keywords = {
            "neurosurgery": ["brain", "neurosurg", "cranial", "spinal", "neurological"],
            "cardiology": ["heart", "cardiac", "cardiovascular", "coronary"],
            "oncology": ["cancer", "tumor", "oncology", "chemotherapy", "radiation"],
            "pediatrics": ["pediatric", "child", "children", "infant", "neonatal"],
            "orthopedics": ["bone", "joint", "orthopedic", "fracture", "musculoskeletal"]
        }

        query_lower = query.lower()
        domain_scores = {}

        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            domain_scores[domain] = score

        return max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else "general"

    async def _calculate_priority(self, query: str, context: Dict[str, Any]) -> float:
        """Calculate priority score for resource allocation"""

        base_priority = 0.5

        # Urgency factor
        urgency_factor = self.user_context.urgency_level / 5.0

        # Cognitive load factor (higher load = higher priority for quick answers)
        load_factor = self.user_context.cognitive_load

        # Time constraint factor
        time_factor = 0.0
        if self.user_context.time_constraints:
            time_remaining = (self.user_context.time_constraints - datetime.now()).total_seconds()
            if time_remaining < 3600:  # Less than 1 hour
                time_factor = 1.0
            elif time_remaining < 86400:  # Less than 1 day
                time_factor = 0.7

        # Query complexity factor
        complexity_factor = min(1.0, len(query.split()) / 20.0)

        priority = base_priority + (urgency_factor * 0.3) + (load_factor * 0.2) + (time_factor * 0.3) + (complexity_factor * 0.2)

        return min(1.0, priority)

    async def _optimize_resource_allocation(self, priority: float, expertise: ExpertiseLevel) -> Dict[str, float]:
        """Determine optimal resource allocation for query processing"""

        allocation = {
            "local_processing": 0.3,
            "external_apis": 0.4,
            "cache_usage": 0.3,
            "quality_threshold": 0.7
        }

        # High priority queries get more external API usage
        if priority > 0.8:
            allocation["external_apis"] = 0.7
            allocation["local_processing"] = 0.2
            allocation["quality_threshold"] = 0.9

        # Expert users get higher quality thresholds
        if expertise in [ExpertiseLevel.EXPERT, ExpertiseLevel.SPECIALIST]:
            allocation["quality_threshold"] = min(1.0, allocation["quality_threshold"] + 0.2)

        # Novice users get more local processing (faster, simpler responses)
        if expertise == ExpertiseLevel.NOVICE:
            allocation["local_processing"] = 0.6
            allocation["external_apis"] = 0.2

        return allocation

# Global context manager instance
contextual_intelligence = ContextualIntelligenceEngine()