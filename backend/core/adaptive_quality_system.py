# backend/core/adaptive_quality_system.py
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from collections import defaultdict, deque
import numpy as np

class QualityDimension(Enum):
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CURRENCY = "currency"
    RELIABILITY = "reliability"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    EVIDENCE_STRENGTH = "evidence_strength"

class ContentType(Enum):
    MEDICAL_FACT = "medical_fact"
    TREATMENT_PROTOCOL = "treatment_protocol"
    DIAGNOSTIC_CRITERIA = "diagnostic_criteria"
    RESEARCH_FINDING = "research_finding"
    CLINICAL_GUIDELINE = "clinical_guideline"
    PROCEDURAL_INSTRUCTION = "procedural_instruction"

@dataclass
class QualityAssessment:
    content_id: str
    content_type: ContentType
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    confidence_level: float
    assessment_reasoning: str
    improvement_suggestions: List[str]
    risk_factors: List[str]
    validation_requirements: List[str]
    assessed_at: datetime

@dataclass
class QualityEvolution:
    content_id: str
    previous_score: float
    new_score: float
    improvement_delta: float
    contributing_factors: List[str]
    user_feedback_impact: float
    outcome_validation_impact: float
    temporal_decay_impact: float

@dataclass
class OutcomeValidation:
    content_id: str
    predicted_quality: float
    actual_outcome: str
    outcome_success: bool
    user_corrections: List[str]
    clinical_effectiveness: Optional[float]
    validation_timestamp: datetime

class AdaptiveQualitySystem:
    def __init__(self):
        self.quality_models = {}
        self.outcome_history = deque(maxlen=10000)
        self.user_feedback_patterns = defaultdict(list)
        self.content_performance_tracking = {}
        self.quality_calibration_factors = defaultdict(float)
        self.learning_rate = 0.1

        # Initialize domain-specific quality models
        self._initialize_quality_models()

    async def assess_content_quality(self, content: str, content_type: ContentType,
                                   context: Dict[str, Any]) -> QualityAssessment:
        """Perform comprehensive quality assessment that adapts based on outcomes"""

        # Step 1: Multi-dimensional quality analysis
        dimension_scores = await self._assess_quality_dimensions(content, content_type, context)

        # Step 2: Apply learned calibrations
        calibrated_scores = await self._apply_calibrations(dimension_scores, content_type, context)

        # Step 3: Calculate overall quality score
        overall_score = await self._calculate_weighted_score(calibrated_scores, content_type)

        # Step 4: Assess confidence in the assessment
        confidence_level = await self._calculate_assessment_confidence(
            calibrated_scores, content_type, context
        )

        # Step 5: Generate improvement suggestions
        improvement_suggestions = await self._generate_improvement_suggestions(
            content, calibrated_scores, context
        )

        # Step 6: Identify risk factors
        risk_factors = await self._identify_risk_factors(content, calibrated_scores, context)

        # Step 7: Determine validation requirements
        validation_requirements = await self._determine_validation_requirements(
            overall_score, content_type, risk_factors
        )

        # Step 8: Generate reasoning
        assessment_reasoning = await self._generate_assessment_reasoning(
            calibrated_scores, improvement_suggestions, risk_factors
        )

        assessment = QualityAssessment(
            content_id=context.get("content_id", "unknown"),
            content_type=content_type,
            overall_score=overall_score,
            dimension_scores=calibrated_scores,
            confidence_level=confidence_level,
            assessment_reasoning=assessment_reasoning,
            improvement_suggestions=improvement_suggestions,
            risk_factors=risk_factors,
            validation_requirements=validation_requirements,
            assessed_at=datetime.now()
        )

        # Store for learning
        await self._store_assessment_for_learning(assessment, content, context)

        return assessment

    async def learn_from_outcome(self, content_id: str, outcome_data: OutcomeValidation):
        """Learn from real-world outcomes to improve future assessments"""

        # Store outcome for analysis
        self.outcome_history.append(outcome_data)

        # Calculate prediction accuracy
        prediction_error = abs(outcome_data.predicted_quality -
                             (1.0 if outcome_data.outcome_success else 0.0))

        # Update calibration factors
        await self._update_calibration_factors(outcome_data, prediction_error)

        # Learn from user corrections
        if outcome_data.user_corrections:
            await self._learn_from_user_corrections(outcome_data)

        # Update content performance tracking
        await self._update_content_performance(outcome_data)

        # Evolve quality models
        await self._evolve_quality_models(outcome_data, prediction_error)

    async def predict_content_longevity(self, content: str, content_type: ContentType,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict how long content will remain accurate and valuable"""

        # Analyze temporal factors
        temporal_analysis = await self._analyze_temporal_factors(content, content_type)

        # Assess domain stability
        domain_stability = await self._assess_domain_stability(content_type, context)

        # Calculate decay rate
        decay_rate = await self._calculate_quality_decay_rate(
            content, content_type, temporal_analysis, domain_stability
        )

        # Predict maintenance needs
        maintenance_schedule = await self._predict_maintenance_schedule(
            content, content_type, decay_rate
        )

        # Identify update triggers
        update_triggers = await self._identify_update_triggers(content, content_type, context)

        return {
            "predicted_validity_period": await self._calculate_validity_period(decay_rate),
            "decay_rate": decay_rate,
            "stability_factors": domain_stability,
            "maintenance_schedule": maintenance_schedule,
            "update_triggers": update_triggers,
            "confidence_degradation_timeline": await self._model_confidence_degradation(decay_rate)
        }

    async def _assess_quality_dimensions(self, content: str, content_type: ContentType,
                                       context: Dict[str, Any]) -> Dict[QualityDimension, float]:
        """Assess content across multiple quality dimensions"""

        scores = {}

        # Accuracy assessment
        scores[QualityDimension.ACCURACY] = await self._assess_accuracy(content, content_type, context)

        # Completeness assessment
        scores[QualityDimension.COMPLETENESS] = await self._assess_completeness(content, content_type)

        # Currency assessment (how up-to-date)
        scores[QualityDimension.CURRENCY] = await self._assess_currency(content, content_type, context)

        # Reliability assessment
        scores[QualityDimension.RELIABILITY] = await self._assess_reliability(content, context)

        # Relevance assessment
        scores[QualityDimension.RELEVANCE] = await self._assess_relevance(content, context)

        # Clarity assessment
        scores[QualityDimension.CLARITY] = await self._assess_clarity(content, content_type)

        # Evidence strength assessment
        scores[QualityDimension.EVIDENCE_STRENGTH] = await self._assess_evidence_strength(content, context)

        return scores

    async def _assess_accuracy(self, content: str, content_type: ContentType,
                             context: Dict[str, Any]) -> float:
        """Assess factual accuracy of content"""

        accuracy_factors = {
            "source_credibility": 0.0,
            "fact_verification": 0.0,
            "internal_consistency": 0.0,
            "expert_consensus": 0.0
        }

        # Check source credibility
        sources = context.get("sources", [])
        if sources:
            credibility_scores = []
            for source in sources:
                credibility_scores.append(await self._evaluate_source_credibility(source))
            accuracy_factors["source_credibility"] = sum(credibility_scores) / len(credibility_scores)

        # Fact verification against knowledge base
        accuracy_factors["fact_verification"] = await self._verify_facts_against_knowledge_base(content)

        # Internal consistency check
        accuracy_factors["internal_consistency"] = await self._check_internal_consistency(content)

        # Expert consensus alignment
        accuracy_factors["expert_consensus"] = await self._check_expert_consensus_alignment(
            content, content_type
        )

        # Weighted accuracy score
        weights = {"source_credibility": 0.3, "fact_verification": 0.3,
                  "internal_consistency": 0.2, "expert_consensus": 0.2}

        accuracy_score = sum(accuracy_factors[factor] * weights[factor]
                           for factor in accuracy_factors)

        return min(1.0, accuracy_score)

    async def _assess_completeness(self, content: str, content_type: ContentType) -> float:
        """Assess how complete the content is for its type"""

        # Define expected elements for each content type
        required_elements = await self._get_required_elements(content_type)

        # Check presence of required elements
        present_elements = []
        for element in required_elements:
            if await self._check_element_presence(content, element):
                present_elements.append(element)

        # Calculate completeness score
        completeness_score = len(present_elements) / len(required_elements) if required_elements else 1.0

        # Adjust for content depth
        depth_factor = await self._assess_content_depth(content, content_type)
        adjusted_score = completeness_score * depth_factor

        return min(1.0, adjusted_score)

    async def _assess_currency(self, content: str, content_type: ContentType,
                             context: Dict[str, Any]) -> float:
        """Assess how current/up-to-date the content is"""

        currency_factors = {
            "reference_recency": 0.0,
            "guideline_alignment": 0.0,
            "terminology_currency": 0.0,
            "practice_standards": 0.0
        }

        # Check reference recency
        references = context.get("references", [])
        if references:
            reference_dates = []
            for ref in references:
                ref_date = await self._extract_reference_date(ref)
                if ref_date:
                    reference_dates.append(ref_date)

            if reference_dates:
                avg_age = sum((datetime.now() - date).days for date in reference_dates) / len(reference_dates)
                currency_factors["reference_recency"] = max(0.0, 1.0 - (avg_age / 1825))  # 5 years

        # Check alignment with current guidelines
        currency_factors["guideline_alignment"] = await self._check_guideline_alignment(content, content_type)

        # Check terminology currency
        currency_factors["terminology_currency"] = await self._check_terminology_currency(content)

        # Check practice standards alignment
        currency_factors["practice_standards"] = await self._check_practice_standards_alignment(
            content, content_type
        )

        # Weighted currency score
        weights = {"reference_recency": 0.3, "guideline_alignment": 0.3,
                  "terminology_currency": 0.2, "practice_standards": 0.2}

        currency_score = sum(currency_factors[factor] * weights[factor]
                           for factor in currency_factors)

        return min(1.0, currency_score)

    async def _apply_calibrations(self, dimension_scores: Dict[QualityDimension, float],
                                content_type: ContentType, context: Dict[str, Any]) -> Dict[QualityDimension, float]:
        """Apply learned calibrations based on historical outcomes"""

        calibrated_scores = {}

        for dimension, score in dimension_scores.items():
            # Get calibration factor for this dimension and content type
            calibration_key = f"{dimension.value}_{content_type.value}"
            calibration_factor = self.quality_calibration_factors.get(calibration_key, 1.0)

            # Apply user expertise calibration
            user_expertise = context.get("user_expertise_level", "intermediate")
            expertise_calibration = await self._get_expertise_calibration(dimension, user_expertise)

            # Apply domain-specific calibration
            domain = context.get("medical_domain", "general")
            domain_calibration = await self._get_domain_calibration(dimension, domain)

            # Combined calibration
            final_calibration = calibration_factor * expertise_calibration * domain_calibration

            calibrated_scores[dimension] = min(1.0, max(0.0, score * final_calibration))

        return calibrated_scores

    async def _calculate_weighted_score(self, dimension_scores: Dict[QualityDimension, float],
                                      content_type: ContentType) -> float:
        """Calculate weighted overall quality score"""

        # Content-type specific weights
        weights = await self._get_content_type_weights(content_type)

        # Calculate weighted score
        weighted_score = sum(
            dimension_scores.get(dimension, 0.5) * weight
            for dimension, weight in weights.items()
        )

        return min(1.0, weighted_score)

    async def _get_content_type_weights(self, content_type: ContentType) -> Dict[QualityDimension, float]:
        """Get dimension weights specific to content type"""

        weight_mappings = {
            ContentType.MEDICAL_FACT: {
                QualityDimension.ACCURACY: 0.35,
                QualityDimension.EVIDENCE_STRENGTH: 0.25,
                QualityDimension.CURRENCY: 0.15,
                QualityDimension.RELIABILITY: 0.15,
                QualityDimension.CLARITY: 0.05,
                QualityDimension.COMPLETENESS: 0.05
            },
            ContentType.TREATMENT_PROTOCOL: {
                QualityDimension.ACCURACY: 0.25,
                QualityDimension.COMPLETENESS: 0.25,
                QualityDimension.CURRENCY: 0.20,
                QualityDimension.EVIDENCE_STRENGTH: 0.15,
                QualityDimension.CLARITY: 0.10,
                QualityDimension.RELIABILITY: 0.05
            },
            ContentType.RESEARCH_FINDING: {
                QualityDimension.EVIDENCE_STRENGTH: 0.30,
                QualityDimension.ACCURACY: 0.25,
                QualityDimension.RELIABILITY: 0.20,
                QualityDimension.CURRENCY: 0.15,
                QualityDimension.COMPLETENESS: 0.05,
                QualityDimension.CLARITY: 0.05
            }
        }

        return weight_mappings.get(content_type, {
            dimension: 1.0/len(QualityDimension) for dimension in QualityDimension
        })

    async def _update_calibration_factors(self, outcome_data: OutcomeValidation, prediction_error: float):
        """Update calibration factors based on outcome validation"""

        if prediction_error < 0.1:  # Good prediction
            learning_factor = self.learning_rate * 0.5
        elif prediction_error < 0.3:  # Moderate prediction
            learning_factor = self.learning_rate
        else:  # Poor prediction
            learning_factor = self.learning_rate * 2.0

        # Update calibration factors
        content_type = await self._get_content_type_from_outcome(outcome_data)

        for dimension in QualityDimension:
            calibration_key = f"{dimension.value}_{content_type.value}"

            current_factor = self.quality_calibration_factors.get(calibration_key, 1.0)

            if outcome_data.outcome_success and prediction_error > 0.2:
                # We underestimated quality
                adjustment = learning_factor * (1.0 + prediction_error)
            elif not outcome_data.outcome_success and prediction_error > 0.2:
                # We overestimated quality
                adjustment = learning_factor * (1.0 - prediction_error)
            else:
                adjustment = 1.0

            self.quality_calibration_factors[calibration_key] = (
                current_factor * (1 - learning_factor) + adjustment * learning_factor
            )

    async def get_quality_insights(self, content_id: str) -> Dict[str, Any]:
        """Get comprehensive quality insights for content"""

        insights = {
            "quality_trends": await self._analyze_quality_trends(content_id),
            "improvement_opportunities": await self._identify_improvement_opportunities(content_id),
            "risk_assessment": await self._assess_quality_risks(content_id),
            "comparative_analysis": await self._compare_with_similar_content(content_id),
            "user_perception_analysis": await self._analyze_user_perception(content_id),
            "predicted_quality_evolution": await self._predict_quality_evolution(content_id)
        }

        return insights

    def _initialize_quality_models(self):
        """Initialize domain-specific quality assessment models"""

        # Medical specialties with their specific quality criteria
        self.quality_models = {
            "neurosurgery": {
                "accuracy_weight": 0.4,
                "evidence_weight": 0.3,
                "currency_weight": 0.2,
                "clarity_weight": 0.1
            },
            "general_medicine": {
                "accuracy_weight": 0.3,
                "completeness_weight": 0.25,
                "currency_weight": 0.25,
                "clarity_weight": 0.2
            },
            "research": {
                "evidence_weight": 0.4,
                "accuracy_weight": 0.3,
                "reliability_weight": 0.2,
                "currency_weight": 0.1
            }
        }

# Global adaptive quality system instance
adaptive_quality_system = AdaptiveQualitySystem()