# backend/core/nuance_merge_engine.py
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import re
import hashlib
import difflib
from collections import defaultdict
# Core dependencies with graceful fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
import logging

logger = logging.getLogger(__name__)

class NuanceType(Enum):
    ENHANCEMENT = "enhancement"
    REFINEMENT = "refinement"
    EXPANSION = "expansion"
    CLARIFICATION = "clarification"
    PRECISION_IMPROVEMENT = "precision_improvement"
    CLINICAL_SPECIFICITY = "clinical_specificity"
    MEDICAL_ACCURACY = "medical_accuracy"
    TERMINOLOGY_UPGRADE = "terminology_upgrade"

class MergeCategory(Enum):
    CONTENT_IMPROVEMENT = "content_improvement"
    MEDICAL_PRECISION = "medical_precision"
    STRUCTURAL_ENHANCEMENT = "structural_enhancement"
    TERMINOLOGY_REFINEMENT = "terminology_refinement"
    CLINICAL_CONTEXT = "clinical_context"
    EVIDENCE_STRENGTHENING = "evidence_strengthening"

class NuanceStatus(Enum):
    DETECTED = "detected"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    APPLIED = "applied"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"

class SimilarityAlgorithm(Enum):
    SEMANTIC_TRANSFORMER = "semantic_transformer"
    JACCARD_COEFFICIENT = "jaccard_coefficient"
    LEVENSHTEIN_DISTANCE = "levenshtein_distance"
    COSINE_SIMILARITY = "cosine_similarity"
    HYBRID_ANALYSIS = "hybrid_analysis"

@dataclass
class SimilarityMetrics:
    semantic_similarity: float
    jaccard_similarity: float
    levenshtein_distance: int
    cosine_similarity: float
    normalized_levenshtein: float
    word_overlap_ratio: float
    sentence_structure_similarity: float

@dataclass
class MedicalContext:
    medical_concepts_added: List[str] = field(default_factory=list)
    anatomical_references: List[str] = field(default_factory=list)
    procedure_references: List[str] = field(default_factory=list)
    drug_references: List[str] = field(default_factory=list)
    specialty_context: Optional[str] = None
    clinical_relevance_score: float = 0.0

@dataclass
class SentenceAnalysis:
    original_sentence: str
    enhanced_sentence: str
    sentence_position: int
    paragraph_position: int
    added_parts: List[str] = field(default_factory=list)
    modified_parts: List[str] = field(default_factory=list)
    removed_parts: List[str] = field(default_factory=list)
    word_level_changes: Dict[str, Any] = field(default_factory=dict)
    sentence_similarity: float = 0.0
    medical_concept_density: float = 0.0
    clinical_importance_score: float = 0.0
    change_type: Optional[str] = None
    impact_category: Optional[str] = None

@dataclass
class DetectedNuance:
    nuance_id: str
    chapter_id: str
    section_id: Optional[str]
    original_content: str
    updated_content: str

    # Similarity analysis
    similarity_metrics: SimilarityMetrics

    # Classification
    nuance_type: NuanceType
    merge_category: MergeCategory
    confidence_score: float

    # Medical context
    medical_context: MedicalContext

    # Sentence-level analysis
    sentence_analyses: List[SentenceAnalysis] = field(default_factory=list)

    # Processing metadata
    detection_algorithm: str = "hybrid_semantic_analysis"
    processing_time_ms: int = 0
    memory_usage_mb: float = 0.0
    algorithm_version: str = "1.0"

    # AI analysis
    ai_analysis: Dict[str, Any] = field(default_factory=dict)
    processing_models: List[str] = field(default_factory=list)
    ai_recommendations: Dict[str, Any] = field(default_factory=dict)

    # Workflow
    status: NuanceStatus = NuanceStatus.DETECTED
    priority_level: int = 5
    auto_apply_eligible: bool = False
    manual_review_required: bool = True

    # Audit trail
    detected_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class NuanceDetectionConfig:
    specialty: str
    exact_duplicate_threshold: float = 0.98
    nuance_threshold_high: float = 0.90
    nuance_threshold_medium: float = 0.75
    nuance_threshold_low: float = 0.60
    significant_change_threshold: float = 0.50
    auto_apply_threshold: float = 0.95
    auto_apply_enabled: bool = False
    require_review_threshold: float = 0.80
    manual_approval_required: bool = True
    primary_similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_semantic_analysis: bool = True
    enable_medical_concept_analysis: bool = True
    enable_clinical_validation: bool = True
    max_content_length: int = 50000
    max_processing_time_ms: int = 30000
    minimum_confidence_threshold: float = 0.70

class AdvancedNuanceMergeEngine:
    def __init__(self):
        self.similarity_cache = {}
        self.medical_terminologies = {}
        self.specialty_configs = {}
        self.processing_metrics = defaultdict(list)
        self.sentence_transformer_model = None

        # Initialize models and configurations
        self._initialize_models()
        self._load_medical_terminologies()
        self._load_specialty_configurations()

    async def detect_nuances(self, original_content: str, updated_content: str,
                           chapter_id: str, context: Dict[str, Any]) -> Optional[DetectedNuance]:
        """Advanced nuance detection with multi-layered analysis"""

        start_time = datetime.utcnow()

        try:
            # Input validation and preprocessing
            if not self._validate_input(original_content, updated_content):
                return None

            # Get configuration for specialty
            specialty = context.get('specialty', 'general_medicine')
            config = self._get_specialty_config(specialty)

            # Quick duplicate check
            if await self._is_exact_duplicate(original_content, updated_content, config):
                return None

            # Calculate comprehensive similarity metrics
            similarity_metrics = await self._calculate_similarity_metrics(
                original_content, updated_content, config
            )

            # Determine if this qualifies as a nuance
            if not self._qualifies_as_nuance(similarity_metrics, config):
                return None

            # Generate unique ID
            nuance_id = self._generate_nuance_id(chapter_id, original_content, updated_content)

            # Perform detailed content analysis
            nuance_type = await self._classify_nuance_type(
                original_content, updated_content, similarity_metrics, context
            )

            merge_category = await self._determine_merge_category(
                original_content, updated_content, nuance_type, context
            )

            # Calculate confidence score
            confidence_score = await self._calculate_confidence_score(
                similarity_metrics, nuance_type, merge_category, context
            )

            # Analyze medical context
            medical_context = await self._analyze_medical_context(
                original_content, updated_content, specialty
            )

            # Sentence-level analysis
            sentence_analyses = await self._perform_sentence_analysis(
                original_content, updated_content
            )

            # AI-powered analysis
            ai_analysis = await self._perform_ai_analysis(
                original_content, updated_content, context
            )

            # Processing metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Create detected nuance
            detected_nuance = DetectedNuance(
                nuance_id=nuance_id,
                chapter_id=chapter_id,
                section_id=context.get('section_id'),
                original_content=original_content,
                updated_content=updated_content,
                similarity_metrics=similarity_metrics,
                nuance_type=nuance_type,
                merge_category=merge_category,
                confidence_score=confidence_score,
                medical_context=medical_context,
                sentence_analyses=sentence_analyses,
                processing_time_ms=int(processing_time),
                ai_analysis=ai_analysis,
                processing_models=["sentence-transformers", "medical-nlp"],
                ai_recommendations=await self._generate_ai_recommendations(
                    original_content, updated_content, confidence_score
                )
            )

            # Determine workflow status
            detected_nuance.auto_apply_eligible = (
                confidence_score >= config.auto_apply_threshold and
                config.auto_apply_enabled
            )

            detected_nuance.manual_review_required = (
                confidence_score < config.require_review_threshold or
                config.manual_approval_required
            )

            # Record processing metrics
            await self._record_processing_metrics(detected_nuance, True, None)

            return detected_nuance

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._record_processing_metrics(None, False, str(e))
            logger.error(f"Nuance detection failed: {e}")
            raise

    async def _calculate_similarity_metrics(self, content1: str, content2: str,
                                          config: NuanceDetectionConfig) -> SimilarityMetrics:
        """Calculate comprehensive similarity metrics"""

        # Check cache first
        cache_key = self._generate_cache_key(content1, content2, "all_metrics")
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Semantic similarity using transformer models
        semantic_similarity = await self._calculate_semantic_similarity(content1, content2)

        # Jaccard similarity
        jaccard_similarity = self._calculate_jaccard_similarity(content1, content2)

        # Levenshtein distance
        levenshtein_distance = self._calculate_levenshtein_distance(content1, content2)
        normalized_levenshtein = 1 - (levenshtein_distance / max(len(content1), len(content2)))

        # Cosine similarity
        cosine_similarity = await self._calculate_cosine_similarity(content1, content2)

        # Word overlap ratio
        word_overlap_ratio = self._calculate_word_overlap_ratio(content1, content2)

        # Sentence structure similarity
        sentence_structure_similarity = self._calculate_sentence_structure_similarity(content1, content2)

        metrics = SimilarityMetrics(
            semantic_similarity=semantic_similarity,
            jaccard_similarity=jaccard_similarity,
            levenshtein_distance=levenshtein_distance,
            cosine_similarity=cosine_similarity,
            normalized_levenshtein=normalized_levenshtein,
            word_overlap_ratio=word_overlap_ratio,
            sentence_structure_similarity=sentence_structure_similarity
        )

        # Cache the result
        self.similarity_cache[cache_key] = metrics

        return metrics

    async def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity using available methods"""

        if SENTENCE_TRANSFORMERS_AVAILABLE and NUMPY_AVAILABLE:
            # Use sentence transformers if available
            if not self.sentence_transformer_model:
                self.sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Generate embeddings
            embeddings = self.sentence_transformer_model.encode([content1, content2])

            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )

            return float(similarity)

        elif SKLEARN_AVAILABLE:
            # Fallback to TF-IDF similarity
            return await self._calculate_cosine_similarity(content1, content2)

        else:
            # Fallback to simple word overlap
            return self._calculate_word_overlap_ratio(content1, content2)

    def _calculate_jaccard_similarity(self, content1: str, content2: str) -> float:
        """Calculate Jaccard similarity coefficient"""

        # Tokenize into words
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        # Calculate Jaccard coefficient
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _calculate_levenshtein_distance(self, content1: str, content2: str) -> int:
        """Calculate Levenshtein distance"""

        # Use difflib for efficient calculation
        sequence_matcher = difflib.SequenceMatcher(None, content1, content2)

        # Calculate edit distance
        operations = sequence_matcher.get_opcodes()
        distance = sum(abs(j-i) + abs(l-k) for tag, i, j, k, l in operations if tag != 'equal')

        return distance

    async def _calculate_cosine_similarity(self, content1: str, content2: str) -> float:
        """Calculate cosine similarity using TF-IDF vectors"""

        if SKLEARN_AVAILABLE:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([content1, content2])

            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            return float(similarity)
        else:
            # Fallback to word overlap
            return self._calculate_word_overlap_ratio(content1, content2)

    def _calculate_word_overlap_ratio(self, content1: str, content2: str) -> float:
        """Calculate word overlap ratio"""

        words1 = content1.lower().split()
        words2 = content2.lower().split()

        overlapping_words = len(set(words1).intersection(set(words2)))
        total_unique_words = len(set(words1).union(set(words2)))

        return overlapping_words / total_unique_words if total_unique_words > 0 else 0.0

    def _calculate_sentence_structure_similarity(self, content1: str, content2: str) -> float:
        """Analyze sentence structure similarity"""

        # Split into sentences
        sentences1 = re.split(r'[.!?]+', content1)
        sentences2 = re.split(r'[.!?]+', content2)

        # Calculate structure metrics
        avg_sentence_length1 = np.mean([len(s.split()) for s in sentences1 if s.strip()])
        avg_sentence_length2 = np.mean([len(s.split()) for s in sentences2 if s.strip()])

        # Structure similarity based on sentence length patterns
        length_similarity = 1 - abs(avg_sentence_length1 - avg_sentence_length2) / max(avg_sentence_length1, avg_sentence_length2)

        return float(length_similarity)

    def _qualifies_as_nuance(self, metrics: SimilarityMetrics, config: NuanceDetectionConfig) -> bool:
        """Determine if similarity metrics qualify as a nuance"""

        # Must be similar enough to be a nuance but different enough to matter
        semantic_qualifies = (
            config.nuance_threshold_low <= metrics.semantic_similarity <= config.nuance_threshold_high
        )

        jaccard_qualifies = (
            config.nuance_threshold_low <= metrics.jaccard_similarity <= config.nuance_threshold_high
        )

        # At least one metric should qualify
        return semantic_qualifies or jaccard_qualifies

    async def _classify_nuance_type(self, original: str, updated: str,
                                   metrics: SimilarityMetrics, context: Dict[str, Any]) -> NuanceType:
        """Classify the type of nuance detected"""

        # Analyze the nature of changes
        original_words = set(original.lower().split())
        updated_words = set(updated.lower().split())

        added_words = updated_words - original_words
        removed_words = original_words - updated_words

        # Medical terminology analysis
        medical_terms_added = await self._identify_medical_terms(list(added_words))

        # Classification logic
        if len(medical_terms_added) > 0:
            if len(medical_terms_added) > len(added_words) * 0.5:
                return NuanceType.MEDICAL_ACCURACY
            else:
                return NuanceType.CLINICAL_SPECIFICITY

        if len(updated) > len(original) * 1.2:
            return NuanceType.EXPANSION
        elif metrics.semantic_similarity > 0.85:
            return NuanceType.REFINEMENT
        elif len(added_words) > len(removed_words):
            return NuanceType.ENHANCEMENT
        else:
            return NuanceType.CLARIFICATION

    async def _determine_merge_category(self, original: str, updated: str,
                                       nuance_type: NuanceType, context: Dict[str, Any]) -> MergeCategory:
        """Determine the category of merge"""

        # Map nuance types to merge categories
        category_mapping = {
            NuanceType.MEDICAL_ACCURACY: MergeCategory.MEDICAL_PRECISION,
            NuanceType.CLINICAL_SPECIFICITY: MergeCategory.CLINICAL_CONTEXT,
            NuanceType.TERMINOLOGY_UPGRADE: MergeCategory.TERMINOLOGY_REFINEMENT,
            NuanceType.EXPANSION: MergeCategory.CONTENT_IMPROVEMENT,
            NuanceType.ENHANCEMENT: MergeCategory.CONTENT_IMPROVEMENT,
            NuanceType.REFINEMENT: MergeCategory.STRUCTURAL_ENHANCEMENT,
            NuanceType.CLARIFICATION: MergeCategory.CONTENT_IMPROVEMENT
        }

        return category_mapping.get(nuance_type, MergeCategory.CONTENT_IMPROVEMENT)

    async def _calculate_confidence_score(self, metrics: SimilarityMetrics,
                                        nuance_type: NuanceType, merge_category: MergeCategory,
                                        context: Dict[str, Any]) -> float:
        """Calculate confidence score for the detected nuance"""

        # Base confidence from similarity metrics
        base_confidence = (
            metrics.semantic_similarity * 0.4 +
            metrics.jaccard_similarity * 0.2 +
            metrics.normalized_levenshtein * 0.2 +
            metrics.cosine_similarity * 0.2
        )

        # Adjust based on nuance type reliability
        type_multipliers = {
            NuanceType.MEDICAL_ACCURACY: 1.1,
            NuanceType.CLINICAL_SPECIFICITY: 1.05,
            NuanceType.ENHANCEMENT: 1.0,
            NuanceType.REFINEMENT: 0.95,
            NuanceType.CLARIFICATION: 0.9
        }

        confidence = base_confidence * type_multipliers.get(nuance_type, 1.0)

        # Ensure within bounds
        return max(0.0, min(1.0, confidence))

    async def _analyze_medical_context(self, original: str, updated: str, specialty: str) -> MedicalContext:
        """Analyze medical context of the changes"""

        # Extract medical concepts
        original_words = set(original.lower().split())
        updated_words = set(updated.lower().split())
        added_words = updated_words - original_words

        medical_concepts = await self._identify_medical_terms(list(added_words))
        anatomical_refs = await self._identify_anatomical_references(list(added_words))
        procedure_refs = await self._identify_procedures(list(added_words))
        drug_refs = await self._identify_drugs(list(added_words))

        # Calculate clinical relevance
        clinical_relevance = self._calculate_clinical_relevance(
            medical_concepts, anatomical_refs, procedure_refs, specialty
        )

        return MedicalContext(
            medical_concepts_added=medical_concepts,
            anatomical_references=anatomical_refs,
            procedure_references=procedure_refs,
            drug_references=drug_refs,
            specialty_context=specialty,
            clinical_relevance_score=clinical_relevance
        )

    async def _perform_sentence_analysis(self, original: str, updated: str) -> List[SentenceAnalysis]:
        """Perform detailed sentence-level analysis"""

        # Split into sentences
        original_sentences = re.split(r'[.!?]+', original)
        updated_sentences = re.split(r'[.!?]+', updated)

        analyses = []

        # Use difflib to find matching sentences
        sequence_matcher = difflib.SequenceMatcher(None, original_sentences, updated_sentences)

        for tag, i1, i2, j1, j2 in sequence_matcher.get_opcodes():
            if tag == 'replace':
                # Analyze replaced sentences
                for idx, (orig_sent, upd_sent) in enumerate(zip(original_sentences[i1:i2], updated_sentences[j1:j2])):
                    analysis = await self._analyze_sentence_pair(orig_sent, upd_sent, i1 + idx)
                    analyses.append(analysis)
            elif tag == 'insert':
                # New sentences added
                for idx, upd_sent in enumerate(updated_sentences[j1:j2]):
                    analysis = SentenceAnalysis(
                        original_sentence="",
                        enhanced_sentence=upd_sent,
                        sentence_position=j1 + idx,
                        paragraph_position=0,
                        change_type="addition",
                        impact_category="medium"
                    )
                    analyses.append(analysis)

        return analyses

    async def _analyze_sentence_pair(self, original_sent: str, updated_sent: str, position: int) -> SentenceAnalysis:
        """Analyze a pair of sentences"""

        # Calculate sentence-level similarity
        similarity = await self._calculate_semantic_similarity(original_sent, updated_sent)

        # Analyze word-level changes
        original_words = original_sent.split()
        updated_words = updated_sent.split()

        sequence_matcher = difflib.SequenceMatcher(None, original_words, updated_words)

        added_parts = []
        modified_parts = []
        removed_parts = []

        for tag, i1, i2, j1, j2 in sequence_matcher.get_opcodes():
            if tag == 'insert':
                added_parts.extend(updated_words[j1:j2])
            elif tag == 'delete':
                removed_parts.extend(original_words[i1:i2])
            elif tag == 'replace':
                modified_parts.append({
                    'original': ' '.join(original_words[i1:i2]),
                    'updated': ' '.join(updated_words[j1:j2])
                })

        # Calculate medical concept density
        medical_concepts = await self._identify_medical_terms(updated_words)
        concept_density = len(medical_concepts) / len(updated_words) if updated_words else 0

        return SentenceAnalysis(
            original_sentence=original_sent,
            enhanced_sentence=updated_sent,
            sentence_position=position,
            paragraph_position=0,
            added_parts=added_parts,
            modified_parts=modified_parts,
            removed_parts=removed_parts,
            sentence_similarity=similarity,
            medical_concept_density=concept_density,
            clinical_importance_score=concept_density * 0.8,  # Simple heuristic
            change_type="modification" if modified_parts else "enhancement"
        )

    async def _perform_ai_analysis(self, original: str, updated: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI-powered analysis of the changes"""

        # This would integrate with actual AI models
        # For now, providing structure for the analysis

        analysis = {
            "content_quality_improvement": await self._assess_quality_improvement(original, updated),
            "medical_accuracy_enhancement": await self._assess_medical_accuracy(original, updated),
            "readability_improvement": await self._assess_readability_improvement(original, updated),
            "clinical_relevance": await self._assess_clinical_relevance_ai(original, updated),
            "suggested_improvements": await self._generate_improvement_suggestions(original, updated),
            "risk_assessment": await self._assess_potential_risks(original, updated)
        }

        return analysis

    # Helper methods for AI analysis
    async def _assess_quality_improvement(self, original: str, updated: str) -> float:
        """Assess overall quality improvement"""
        # Placeholder for actual AI assessment
        return 0.75

    async def _assess_medical_accuracy(self, original: str, updated: str) -> float:
        """Assess medical accuracy improvement"""
        # Placeholder for medical AI assessment
        return 0.80

    async def _assess_readability_improvement(self, original: str, updated: str) -> float:
        """Assess readability improvement"""
        # Simple readability metrics
        original_complexity = self._calculate_readability_score(original)
        updated_complexity = self._calculate_readability_score(updated)

        return max(0, updated_complexity - original_complexity)

    def _calculate_readability_score(self, text: str) -> float:
        """Calculate simple readability score"""
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        avg_words_per_sentence = words / sentences if sentences > 0 else 0

        # Simple inverse relationship (shorter sentences = more readable)
        return max(0, 1 - (avg_words_per_sentence / 30))

    # Utility methods
    def _validate_input(self, original: str, updated: str) -> bool:
        """Validate input content"""
        return bool(original and updated and original.strip() and updated.strip())

    async def _is_exact_duplicate(self, original: str, updated: str, config: NuanceDetectionConfig) -> bool:
        """Check if content is an exact duplicate"""
        similarity = await self._calculate_semantic_similarity(original, updated)
        return similarity >= config.exact_duplicate_threshold

    def _generate_nuance_id(self, chapter_id: str, original: str, updated: str) -> str:
        """Generate unique nuance ID"""
        content_hash = hashlib.md5(f"{original}::{updated}".encode()).hexdigest()[:8]
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"nuance_{chapter_id}_{timestamp}_{content_hash}"

    def _generate_cache_key(self, content1: str, content2: str, algorithm: str) -> str:
        """Generate cache key for similarity calculations"""
        combined = f"{content1}::{content2}::{algorithm}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_specialty_config(self, specialty: str) -> NuanceDetectionConfig:
        """Get configuration for medical specialty"""
        if specialty in self.specialty_configs:
            return self.specialty_configs[specialty]

        # Return default configuration
        return NuanceDetectionConfig(specialty=specialty)

    def _initialize_models(self):
        """Initialize AI models and configurations"""
        # Load sentence transformer model lazily
        self.sentence_transformer_model = None

    def _load_medical_terminologies(self):
        """Load medical terminology databases"""
        # Placeholder for medical terminology loading
        self.medical_terminologies = {
            'anatomy': [],
            'procedures': [],
            'drugs': [],
            'conditions': []
        }

    def _load_specialty_configurations(self):
        """Load specialty-specific configurations"""
        # Default configurations for different specialties
        self.specialty_configs = {
            'neurosurgery': NuanceDetectionConfig(
                specialty='neurosurgery',
                nuance_threshold_high=0.92,
                nuance_threshold_medium=0.78,
                require_review_threshold=0.85
            ),
            'general_medicine': NuanceDetectionConfig(
                specialty='general_medicine',
                nuance_threshold_high=0.85,
                nuance_threshold_medium=0.68,
                require_review_threshold=0.75
            )
        }

    async def _identify_medical_terms(self, words: List[str]) -> List[str]:
        """Identify medical terms in word list"""
        # Placeholder for medical term identification
        medical_terms = []
        medical_indicators = ['surgical', 'clinical', 'medical', 'anatomy', 'procedure', 'diagnosis']

        for word in words:
            if any(indicator in word.lower() for indicator in medical_indicators):
                medical_terms.append(word)

        return medical_terms

    async def _identify_anatomical_references(self, words: List[str]) -> List[str]:
        """Identify anatomical references"""
        anatomical_terms = ['brain', 'spine', 'nerve', 'tissue', 'organ', 'muscle', 'bone']
        return [word for word in words if word.lower() in anatomical_terms]

    async def _identify_procedures(self, words: List[str]) -> List[str]:
        """Identify medical procedures"""
        procedure_terms = ['surgery', 'operation', 'procedure', 'treatment', 'therapy']
        return [word for word in words if word.lower() in procedure_terms]

    async def _identify_drugs(self, words: List[str]) -> List[str]:
        """Identify drug references"""
        drug_indicators = ['medication', 'drug', 'treatment', 'therapy', 'pharmaceutical']
        return [word for word in words if word.lower() in drug_indicators]

    def _calculate_clinical_relevance(self, medical_concepts: List[str],
                                    anatomical_refs: List[str],
                                    procedure_refs: List[str],
                                    specialty: str) -> float:
        """Calculate clinical relevance score"""
        total_medical_terms = len(medical_concepts) + len(anatomical_refs) + len(procedure_refs)

        # Simple scoring based on medical term density
        if total_medical_terms == 0:
            return 0.0
        elif total_medical_terms <= 2:
            return 0.4
        elif total_medical_terms <= 5:
            return 0.7
        else:
            return 0.9

    async def _assess_clinical_relevance_ai(self, original: str, updated: str) -> float:
        """AI-powered clinical relevance assessment"""
        # Placeholder for AI assessment
        return 0.75

    async def _generate_improvement_suggestions(self, original: str, updated: str) -> List[str]:
        """Generate AI-powered improvement suggestions"""
        return [
            "Consider adding more specific medical terminology",
            "Enhance clinical context with procedural details",
            "Include relevant anatomical references"
        ]

    async def _assess_potential_risks(self, original: str, updated: str) -> Dict[str, Any]:
        """Assess potential risks of the changes"""
        return {
            "medical_accuracy_risk": "low",
            "information_loss_risk": "minimal",
            "clarity_impact": "positive",
            "overall_risk_level": "low"
        }

    async def _generate_ai_recommendations(self, original: str, updated: str, confidence: float) -> Dict[str, Any]:
        """Generate AI recommendations for the nuance"""

        recommendations = {
            "apply_automatically": confidence > 0.9,
            "require_review": confidence < 0.8,
            "suggested_action": "approve" if confidence > 0.85 else "review",
            "improvement_priority": "high" if confidence > 0.9 else "medium",
            "additional_validation_needed": confidence < 0.7
        }

        return recommendations

    async def _record_processing_metrics(self, nuance: Optional[DetectedNuance],
                                       success: bool, error_message: Optional[str]):
        """Record processing metrics for analysis"""

        metric = {
            "timestamp": datetime.utcnow(),
            "success": success,
            "processing_time_ms": nuance.processing_time_ms if nuance else 0,
            "memory_usage_mb": nuance.memory_usage_mb if nuance else 0,
            "error_message": error_message,
            "content_length": len(nuance.original_content) if nuance else 0,
            "confidence_score": nuance.confidence_score if nuance else 0
        }

        self.processing_metrics["nuance_detection"].append(metric)

# Global instance
nuance_merge_engine = AdvancedNuanceMergeEngine()