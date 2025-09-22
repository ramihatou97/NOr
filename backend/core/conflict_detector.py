# backend/core/conflict_detector.py
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import re
from collections import defaultdict
import numpy as np

class ConflictType(Enum):
    CONTRADICTION = "contradiction"
    INCONSISTENCY = "inconsistency"
    OUTDATED_INFORMATION = "outdated_information"
    METHODOLOGY_CONFLICT = "methodology_conflict"
    SCOPE_MISMATCH = "scope_mismatch"
    DEFINITIONAL_DISCREPANCY = "definitional_discrepancy"
    STATISTICAL_DISAGREEMENT = "statistical_disagreement"

class ConflictSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ConflictResolutionStrategy(Enum):
    MANUAL_REVIEW = "manual_review"
    EXPERT_CONSULTATION = "expert_consultation"
    ADDITIONAL_RESEARCH = "additional_research"
    TEMPORAL_PRECEDENCE = "temporal_precedence"
    SOURCE_AUTHORITY = "source_authority"
    CONSENSUS_BUILDING = "consensus_building"

@dataclass
class ConflictEvidence:
    conflicting_statement: str
    source_id: str
    confidence_level: float
    evidence_strength: float
    publication_date: Optional[datetime]
    methodology_type: Optional[str]

@dataclass
class DetectedConflict:
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    conflicting_evidences: List[ConflictEvidence]
    affected_concepts: List[str]
    confidence_score: float
    resolution_suggestions: List[str]
    resolution_strategy: ConflictResolutionStrategy
    temporal_factor: float
    detected_at: datetime

@dataclass
class ConflictResolution:
    conflict_id: str
    resolution_method: str
    chosen_evidence: ConflictEvidence
    reasoning: str
    confidence_in_resolution: float
    alternative_perspectives: List[str]
    follow_up_actions: List[str]
    resolved_at: datetime

class AdvancedConflictDetector:
    def __init__(self):
        self.conflict_patterns = {}
        self.resolution_history = defaultdict(list)
        self.domain_expertise_weights = {}
        self.temporal_validity_models = {}
        self.source_credibility_scores = defaultdict(float)

        # Initialize conflict detection patterns
        self._initialize_conflict_patterns()

    async def detect_conflicts(self, content_pieces: List[Dict[str, Any]],
                             context: Dict[str, Any]) -> List[DetectedConflict]:
        """Detect conflicts between multiple pieces of content"""

        conflicts = []

        # Pairwise conflict detection
        for i, content1 in enumerate(content_pieces):
            for j, content2 in enumerate(content_pieces[i+1:], i+1):
                conflict = await self._detect_pairwise_conflict(content1, content2, context)
                if conflict:
                    conflicts.append(conflict)

        # Multi-way conflict detection
        multi_conflicts = await self._detect_multi_way_conflicts(content_pieces, context)
        conflicts.extend(multi_conflicts)

        # Temporal conflict detection
        temporal_conflicts = await self._detect_temporal_conflicts(content_pieces, context)
        conflicts.extend(temporal_conflicts)

        # Contextual conflict detection
        contextual_conflicts = await self._detect_contextual_conflicts(content_pieces, context)
        conflicts.extend(contextual_conflicts)

        # Rank conflicts by severity and confidence
        ranked_conflicts = await self._rank_conflicts(conflicts)

        return ranked_conflicts

    async def analyze_contradiction(self, statement1: str, statement2: str,
                                  context: Dict[str, Any]) -> Optional[DetectedConflict]:
        """Deep analysis of potential contradiction between two statements"""

        # Extract claims from statements
        claims1 = await self._extract_claims(statement1)
        claims2 = await self._extract_claims(statement2)

        # Semantic analysis
        semantic_conflict = await self._analyze_semantic_conflict(claims1, claims2)

        # Logical analysis
        logical_conflict = await self._analyze_logical_conflict(claims1, claims2)

        # Quantitative analysis
        quantitative_conflict = await self._analyze_quantitative_conflict(claims1, claims2)

        # Temporal analysis
        temporal_conflict = await self._analyze_temporal_conflict(statement1, statement2, context)

        # Combine analyses
        if any([semantic_conflict, logical_conflict, quantitative_conflict, temporal_conflict]):
            conflict = await self._synthesize_conflict_analysis(
                statement1, statement2, context,
                semantic_conflict, logical_conflict, quantitative_conflict, temporal_conflict
            )
            return conflict

        return None

    async def intelligent_conflict_resolution(self, conflict: DetectedConflict,
                                            additional_context: Dict[str, Any]) -> ConflictResolution:
        """Intelligently resolve detected conflicts using multiple strategies"""

        # Analyze resolution strategies
        strategy_scores = await self._evaluate_resolution_strategies(conflict, additional_context)

        # Choose optimal strategy
        optimal_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]

        # Apply resolution strategy
        resolution = await self._apply_resolution_strategy(
            conflict, optimal_strategy, additional_context
        )

        # Learn from resolution
        await self._learn_from_resolution(conflict, resolution)

        return resolution

    async def _detect_pairwise_conflict(self, content1: Dict[str, Any],
                                      content2: Dict[str, Any],
                                      context: Dict[str, Any]) -> Optional[DetectedConflict]:
        """Detect conflicts between two pieces of content"""

        # Extract text content
        text1 = content1.get("content", "")
        text2 = content2.get("content", "")

        # Skip if either is empty
        if not text1 or not text2:
            return None

        # Check for different types of conflicts
        contradiction = await self._check_direct_contradiction(text1, text2, context)
        if contradiction:
            return contradiction

        inconsistency = await self._check_logical_inconsistency(text1, text2, context)
        if inconsistency:
            return inconsistency

        methodology_conflict = await self._check_methodology_conflict(content1, content2, context)
        if methodology_conflict:
            return methodology_conflict

        definitional_conflict = await self._check_definitional_conflict(text1, text2, context)
        if definitional_conflict:
            return definitional_conflict

        return None

    async def _check_direct_contradiction(self, text1: str, text2: str,
                                        context: Dict[str, Any]) -> Optional[DetectedConflict]:
        """Check for direct contradictions between statements"""

        # Extract medical claims
        claims1 = await self._extract_medical_claims(text1)
        claims2 = await self._extract_medical_claims(text2)

        contradictions = []

        for claim1 in claims1:
            for claim2 in claims2:
                if await self._are_contradictory(claim1, claim2):
                    contradictions.append((claim1, claim2))

        if contradictions:
            # Create conflict evidence
            evidence1 = ConflictEvidence(
                conflicting_statement=text1,
                source_id=context.get("source1_id", "unknown"),
                confidence_level=context.get("source1_confidence", 0.5),
                evidence_strength=await self._calculate_evidence_strength(text1),
                publication_date=context.get("source1_date"),
                methodology_type=context.get("source1_methodology")
            )

            evidence2 = ConflictEvidence(
                conflicting_statement=text2,
                source_id=context.get("source2_id", "unknown"),
                confidence_level=context.get("source2_confidence", 0.5),
                evidence_strength=await self._calculate_evidence_strength(text2),
                publication_date=context.get("source2_date"),
                methodology_type=context.get("source2_methodology")
            )

            # Calculate conflict severity
            severity = await self._calculate_conflict_severity(contradictions, evidence1, evidence2)

            # Generate conflict
            conflict = DetectedConflict(
                conflict_id=f"contradiction_{datetime.now().timestamp()}",
                conflict_type=ConflictType.CONTRADICTION,
                severity=severity,
                description=f"Direct contradiction found: {contradictions[0][0]['claim']} vs {contradictions[0][1]['claim']}",
                conflicting_evidences=[evidence1, evidence2],
                affected_concepts=await self._extract_affected_concepts(contradictions),
                confidence_score=await self._calculate_conflict_confidence(contradictions, evidence1, evidence2),
                resolution_suggestions=await self._generate_resolution_suggestions(contradictions, evidence1, evidence2),
                resolution_strategy=await self._recommend_resolution_strategy(severity, evidence1, evidence2),
                temporal_factor=await self._calculate_temporal_factor(evidence1, evidence2),
                detected_at=datetime.now()
            )

            return conflict

        return None

    async def _are_contradictory(self, claim1: Dict[str, Any], claim2: Dict[str, Any]) -> bool:
        """Determine if two medical claims are contradictory"""

        # Check for opposite assertions
        if await self._check_opposite_assertions(claim1, claim2):
            return True

        # Check for mutually exclusive statements
        if await self._check_mutual_exclusivity(claim1, claim2):
            return True

        # Check for quantitative contradictions
        if await self._check_quantitative_contradictions(claim1, claim2):
            return True

        # Check for temporal contradictions
        if await self._check_temporal_contradictions(claim1, claim2):
            return True

        return False

    async def _check_opposite_assertions(self, claim1: Dict[str, Any], claim2: Dict[str, Any]) -> bool:
        """Check for opposite assertions (X is effective vs X is ineffective)"""

        # Define opposite assertion patterns
        opposite_patterns = [
            ("effective", "ineffective"),
            ("safe", "unsafe"),
            ("beneficial", "harmful"),
            ("increases", "decreases"),
            ("improves", "worsens"),
            ("prevents", "causes"),
            ("cures", "exacerbates"),
            ("recommended", "contraindicated")
        ]

        claim1_text = claim1.get("text", "").lower()
        claim2_text = claim2.get("text", "").lower()

        # Check if the same subject has opposite predicates
        subject1 = claim1.get("subject", "")
        subject2 = claim2.get("subject", "")

        if await self._are_same_subject(subject1, subject2):
            for positive, negative in opposite_patterns:
                if positive in claim1_text and negative in claim2_text:
                    return True
                if negative in claim1_text and positive in claim2_text:
                    return True

        return False

    async def _check_quantitative_contradictions(self, claim1: Dict[str, Any], claim2: Dict[str, Any]) -> bool:
        """Check for contradictory quantitative claims"""

        # Extract numerical values and ranges
        numbers1 = await self._extract_numerical_claims(claim1.get("text", ""))
        numbers2 = await self._extract_numerical_claims(claim2.get("text", ""))

        for num1 in numbers1:
            for num2 in numbers2:
                if await self._are_contradictory_numbers(num1, num2):
                    return True

        return False

    async def _extract_numerical_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract numerical claims from text"""

        numerical_claims = []

        # Patterns for different types of numerical claims
        patterns = [
            r"(\d+(?:\.\d+)?)\s*%",  # Percentages
            r"(\d+(?:\.\d+)?)\s*(mg|g|kg|ml|l)",  # Dosages
            r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)",  # Ranges
            r"(less than|greater than|more than)\s*(\d+(?:\.\d+)?)",  # Comparisons
            r"(\d+(?:\.\d+)?)\s*(times|fold)",  # Multipliers
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                numerical_claims.append({
                    "value": match.group(1),
                    "unit": match.group(2) if len(match.groups()) > 1 else None,
                    "context": text[max(0, match.start()-50):match.end()+50],
                    "type": await self._classify_numerical_claim(match.group(0))
                })

        return numerical_claims

    async def _check_methodology_conflict(self, content1: Dict[str, Any],
                                        content2: Dict[str, Any],
                                        context: Dict[str, Any]) -> Optional[DetectedConflict]:
        """Check for conflicts in research methodology that might explain different results"""

        method1 = content1.get("methodology", {})
        method2 = content2.get("methodology", {})

        if not method1 or not method2:
            return None

        # Compare study designs
        design_conflict = await self._compare_study_designs(method1, method2)

        # Compare sample sizes
        sample_conflict = await self._compare_sample_sizes(method1, method2)

        # Compare inclusion/exclusion criteria
        criteria_conflict = await self._compare_study_criteria(method1, method2)

        # Compare outcome measures
        outcome_conflict = await self._compare_outcome_measures(method1, method2)

        if any([design_conflict, sample_conflict, criteria_conflict, outcome_conflict]):
            conflict = DetectedConflict(
                conflict_id=f"methodology_{datetime.now().timestamp()}",
                conflict_type=ConflictType.METHODOLOGY_CONFLICT,
                severity=ConflictSeverity.MEDIUM,
                description="Methodological differences may explain conflicting results",
                conflicting_evidences=[
                    ConflictEvidence(
                        conflicting_statement=content1.get("content", ""),
                        source_id=content1.get("source_id", "unknown"),
                        confidence_level=0.7,
                        evidence_strength=0.6,
                        methodology_type=method1.get("study_type")
                    ),
                    ConflictEvidence(
                        conflicting_statement=content2.get("content", ""),
                        source_id=content2.get("source_id", "unknown"),
                        confidence_level=0.7,
                        evidence_strength=0.6,
                        methodology_type=method2.get("study_type")
                    )
                ],
                affected_concepts=[content1.get("main_concept", ""), content2.get("main_concept", "")],
                confidence_score=0.8,
                resolution_suggestions=[
                    "Consider methodological differences when interpreting results",
                    "Look for meta-analyses that account for study heterogeneity",
                    "Evaluate which methodology is more appropriate for the research question"
                ],
                resolution_strategy=ConflictResolutionStrategy.ADDITIONAL_RESEARCH,
                temporal_factor=0.5,
                detected_at=datetime.now()
            )

            return conflict

        return None

    async def _evaluate_resolution_strategies(self, conflict: DetectedConflict,
                                            context: Dict[str, Any]) -> Dict[ConflictResolutionStrategy, float]:
        """Evaluate different resolution strategies for a conflict"""

        strategy_scores = {}

        # Manual review strategy
        strategy_scores[ConflictResolutionStrategy.MANUAL_REVIEW] = await self._score_manual_review(conflict, context)

        # Expert consultation strategy
        strategy_scores[ConflictResolutionStrategy.EXPERT_CONSULTATION] = await self._score_expert_consultation(conflict, context)

        # Additional research strategy
        strategy_scores[ConflictResolutionStrategy.ADDITIONAL_RESEARCH] = await self._score_additional_research(conflict, context)

        # Temporal precedence strategy
        strategy_scores[ConflictResolutionStrategy.TEMPORAL_PRECEDENCE] = await self._score_temporal_precedence(conflict, context)

        # Source authority strategy
        strategy_scores[ConflictResolutionStrategy.SOURCE_AUTHORITY] = await self._score_source_authority(conflict, context)

        # Consensus building strategy
        strategy_scores[ConflictResolutionStrategy.CONSENSUS_BUILDING] = await self._score_consensus_building(conflict, context)

        return strategy_scores

    async def _apply_resolution_strategy(self, conflict: DetectedConflict,
                                       strategy: ConflictResolutionStrategy,
                                       context: Dict[str, Any]) -> ConflictResolution:
        """Apply the chosen resolution strategy"""

        if strategy == ConflictResolutionStrategy.TEMPORAL_PRECEDENCE:
            return await self._resolve_by_temporal_precedence(conflict, context)
        elif strategy == ConflictResolutionStrategy.SOURCE_AUTHORITY:
            return await self._resolve_by_source_authority(conflict, context)
        elif strategy == ConflictResolutionStrategy.ADDITIONAL_RESEARCH:
            return await self._resolve_by_additional_research(conflict, context)
        elif strategy == ConflictResolutionStrategy.CONSENSUS_BUILDING:
            return await self._resolve_by_consensus_building(conflict, context)
        else:
            return await self._resolve_by_manual_review(conflict, context)

    async def _resolve_by_temporal_precedence(self, conflict: DetectedConflict,
                                            context: Dict[str, Any]) -> ConflictResolution:
        """Resolve conflict by giving precedence to more recent evidence"""

        # Find the most recent evidence
        most_recent_evidence = max(
            conflict.conflicting_evidences,
            key=lambda e: e.publication_date or datetime.min
        )

        resolution = ConflictResolution(
            conflict_id=conflict.conflict_id,
            resolution_method="temporal_precedence",
            chosen_evidence=most_recent_evidence,
            reasoning=f"Chose more recent evidence from {most_recent_evidence.publication_date}",
            confidence_in_resolution=0.8,
            alternative_perspectives=[
                e.conflicting_statement for e in conflict.conflicting_evidences
                if e != most_recent_evidence
            ],
            follow_up_actions=[
                "Monitor for newer research",
                "Review methodology of older studies",
                "Check for systematic reviews"
            ],
            resolved_at=datetime.now()
        )

        return resolution

    async def get_conflict_trends(self, domain: str, time_period: timedelta) -> Dict[str, Any]:
        """Analyze conflict trends in a specific domain over time"""

        cutoff_date = datetime.now() - time_period
        recent_conflicts = [
            conflict for conflict in self.resolution_history[domain]
            if conflict.get("detected_at", datetime.min) > cutoff_date
        ]

        trends = {
            "total_conflicts": len(recent_conflicts),
            "conflict_types_distribution": await self._analyze_conflict_types(recent_conflicts),
            "resolution_success_rate": await self._calculate_resolution_success_rate(recent_conflicts),
            "common_conflict_patterns": await self._identify_common_patterns(recent_conflicts),
            "resolution_time_trends": await self._analyze_resolution_times(recent_conflicts),
            "recurring_conflicts": await self._identify_recurring_conflicts(recent_conflicts)
        }

        return trends

    def _initialize_conflict_patterns(self):
        """Initialize patterns for conflict detection"""

        self.conflict_patterns = {
            "contradictory_terms": [
                ("effective", "ineffective"),
                ("safe", "dangerous"),
                ("recommended", "contraindicated"),
                ("increases", "decreases"),
                ("improves", "worsens")
            ],
            "quantitative_thresholds": {
                "significant_difference": 0.2,  # 20% difference is significant
                "confidence_interval_overlap": 0.1
            },
            "temporal_significance": {
                "outdated_threshold": timedelta(days=1825),  # 5 years
                "recent_precedence_weight": 0.8
            }
        }

# Global conflict detector instance
conflict_detector = AdvancedConflictDetector()