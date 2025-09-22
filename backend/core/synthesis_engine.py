# backend/core/synthesis_engine.py
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from collections import defaultdict
import numpy as np

class SynthesisType(Enum):
    COMPREHENSIVE_REVIEW = "comprehensive_review"
    CLINICAL_SUMMARY = "clinical_summary"
    RESEARCH_OVERVIEW = "research_overview"
    TREATMENT_GUIDELINES = "treatment_guidelines"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    CASE_STUDY_SYNTHESIS = "case_study_synthesis"
    META_ANALYSIS_SUMMARY = "meta_analysis_summary"

class EvidenceLevel(Enum):
    LEVEL_1 = "systematic_review_meta_analysis"
    LEVEL_2 = "randomized_controlled_trial"
    LEVEL_3 = "cohort_study"
    LEVEL_4 = "case_control_study"
    LEVEL_5 = "case_series_report"
    EXPERT_OPINION = "expert_opinion"

@dataclass
class SynthesisSource:
    source_id: str
    content: str
    evidence_level: EvidenceLevel
    confidence_score: float
    publication_date: datetime
    authors: List[str]
    journal: str
    methodology: Dict[str, Any]
    key_findings: List[str]
    limitations: List[str]
    sample_size: Optional[int]

@dataclass
class SynthesisResult:
    synthesis_id: str
    synthesis_type: SynthesisType
    topic: str
    synthesized_content: str
    evidence_hierarchy: List[Dict[str, Any]]
    consensus_points: List[str]
    conflicting_evidence: List[Dict[str, Any]]
    research_gaps: List[str]
    clinical_implications: List[str]
    quality_assessment: Dict[str, float]
    confidence_level: float
    sources_used: List[str]
    methodology_notes: str
    limitations: List[str]
    recommendations: List[str]
    created_at: datetime

class IntelligentSynthesisEngine:
    def __init__(self):
        self.synthesis_templates = {}
        self.quality_weights = {}
        self.evidence_hierarchies = {}
        self.synthesis_history = defaultdict(list)
        self.consensus_algorithms = {}

        # Initialize synthesis framework
        self._initialize_synthesis_framework()

    async def synthesize_knowledge(self, sources: List[SynthesisSource],
                                 synthesis_type: SynthesisType,
                                 topic: str,
                                 context: Dict[str, Any]) -> SynthesisResult:
        """Create intelligent synthesis of multiple knowledge sources"""

        # Step 1: Source analysis and preprocessing
        processed_sources = await self._preprocess_sources(sources, context)

        # Step 2: Evidence hierarchy construction
        evidence_hierarchy = await self._construct_evidence_hierarchy(processed_sources)

        # Step 3: Conflict identification and resolution
        conflicts, resolved_evidence = await self._identify_and_resolve_conflicts(processed_sources)

        # Step 4: Consensus analysis
        consensus_points = await self._analyze_consensus(resolved_evidence)

        # Step 5: Gap analysis
        research_gaps = await self._identify_research_gaps(resolved_evidence, topic)

        # Step 6: Content synthesis
        synthesized_content = await self._generate_synthesized_content(
            resolved_evidence, synthesis_type, topic, context
        )

        # Step 7: Clinical implications extraction
        clinical_implications = await self._extract_clinical_implications(
            resolved_evidence, synthesis_type
        )

        # Step 8: Quality assessment
        quality_assessment = await self._assess_synthesis_quality(
            synthesized_content, resolved_evidence, synthesis_type
        )

        # Step 9: Generate recommendations
        recommendations = await self._generate_recommendations(
            resolved_evidence, synthesis_type, clinical_implications
        )

        # Step 10: Methodology documentation
        methodology_notes = await self._document_methodology(
            sources, synthesis_type, evidence_hierarchy
        )

        # Step 11: Limitation identification
        limitations = await self._identify_synthesis_limitations(
            sources, conflicts, quality_assessment
        )

        synthesis_result = SynthesisResult(
            synthesis_id=f"synthesis_{datetime.now().timestamp()}",
            synthesis_type=synthesis_type,
            topic=topic,
            synthesized_content=synthesized_content,
            evidence_hierarchy=evidence_hierarchy,
            consensus_points=consensus_points,
            conflicting_evidence=conflicts,
            research_gaps=research_gaps,
            clinical_implications=clinical_implications,
            quality_assessment=quality_assessment,
            confidence_level=await self._calculate_overall_confidence(quality_assessment),
            sources_used=[source.source_id for source in sources],
            methodology_notes=methodology_notes,
            limitations=limitations,
            recommendations=recommendations,
            created_at=datetime.now()
        )

        # Store for learning
        await self._store_synthesis_for_learning(synthesis_result, context)

        return synthesis_result

    async def adaptive_synthesis(self, sources: List[SynthesisSource],
                               user_expertise: str,
                               target_audience: str,
                               context: Dict[str, Any]) -> SynthesisResult:
        """Generate synthesis adapted to user expertise and target audience"""

        # Determine optimal synthesis approach
        synthesis_approach = await self._determine_optimal_approach(
            sources, user_expertise, target_audience, context
        )

        # Adapt content complexity
        complexity_level = await self._determine_complexity_level(user_expertise, target_audience)

        # Customize synthesis type
        synthesis_type = await self._select_synthesis_type(sources, target_audience, context)

        # Generate adapted synthesis
        adapted_synthesis = await self.synthesize_knowledge(
            sources, synthesis_type, context.get("topic", ""), context
        )

        # Post-process for audience
        adapted_synthesis.synthesized_content = await self._adapt_for_audience(
            adapted_synthesis.synthesized_content, user_expertise, target_audience
        )

        return adapted_synthesis

    async def _preprocess_sources(self, sources: List[SynthesisSource],
                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Preprocess sources for synthesis"""

        processed_sources = []

        for source in sources:
            # Extract key information
            key_info = await self._extract_key_information(source)

            # Assess source quality
            quality_score = await self._assess_source_quality(source)

            # Extract methodological details
            methodology_details = await self._extract_methodology_details(source)

            # Identify main claims
            main_claims = await self._extract_main_claims(source)

            # Assess evidence strength
            evidence_strength = await self._assess_evidence_strength(source)

            processed_source = {
                "original_source": source,
                "key_information": key_info,
                "quality_score": quality_score,
                "methodology_details": methodology_details,
                "main_claims": main_claims,
                "evidence_strength": evidence_strength,
                "relevance_score": await self._calculate_relevance_score(source, context)
            }

            processed_sources.append(processed_source)

        # Sort by quality and relevance
        processed_sources.sort(
            key=lambda x: x["quality_score"] * x["relevance_score"],
            reverse=True
        )

        return processed_sources

    async def _construct_evidence_hierarchy(self, processed_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Construct evidence hierarchy based on quality and type"""

        hierarchy = []

        # Group by evidence level
        evidence_groups = defaultdict(list)
        for source in processed_sources:
            evidence_level = source["original_source"].evidence_level
            evidence_groups[evidence_level].append(source)

        # Process each evidence level
        level_order = [
            EvidenceLevel.LEVEL_1,
            EvidenceLevel.LEVEL_2,
            EvidenceLevel.LEVEL_3,
            EvidenceLevel.LEVEL_4,
            EvidenceLevel.LEVEL_5,
            EvidenceLevel.EXPERT_OPINION
        ]

        for evidence_level in level_order:
            if evidence_level in evidence_groups:
                level_sources = evidence_groups[evidence_level]

                # Sort within level by quality
                level_sources.sort(key=lambda x: x["quality_score"], reverse=True)

                # Create level summary
                level_summary = {
                    "evidence_level": evidence_level,
                    "source_count": len(level_sources),
                    "sources": level_sources,
                    "consensus_strength": await self._calculate_level_consensus(level_sources),
                    "overall_quality": sum(s["quality_score"] for s in level_sources) / len(level_sources),
                    "key_findings": await self._synthesize_level_findings(level_sources)
                }

                hierarchy.append(level_summary)

        return hierarchy

    async def _identify_and_resolve_conflicts(self, processed_sources: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Identify conflicts between sources and resolve them"""

        conflicts = []
        resolved_evidence = []

        # Pairwise conflict detection
        for i, source1 in enumerate(processed_sources):
            for j, source2 in enumerate(processed_sources[i+1:], i+1):
                conflict = await self._detect_source_conflict(source1, source2)
                if conflict:
                    conflicts.append(conflict)

        # Group sources by conflicting claims
        conflict_groups = await self._group_conflicting_sources(conflicts, processed_sources)

        # Resolve each conflict group
        for group in conflict_groups:
            resolved_claim = await self._resolve_conflict_group(group)
            resolved_evidence.append(resolved_claim)

        # Add non-conflicting sources
        non_conflicting = [
            source for source in processed_sources
            if not any(source in group["sources"] for group in conflict_groups)
        ]
        resolved_evidence.extend(non_conflicting)

        return conflicts, resolved_evidence

    async def _analyze_consensus(self, resolved_evidence: List[Dict[str, Any]]) -> List[str]:
        """Analyze consensus across resolved evidence"""

        consensus_points = []

        # Extract all claims
        all_claims = []
        for evidence in resolved_evidence:
            if "main_claims" in evidence:
                all_claims.extend(evidence["main_claims"])

        # Group similar claims
        claim_groups = await self._group_similar_claims(all_claims)

        # Identify consensus claims (supported by multiple high-quality sources)
        for group in claim_groups:
            consensus_strength = await self._calculate_claim_consensus(group, resolved_evidence)
            if consensus_strength > 0.7:  # Strong consensus threshold
                consensus_points.append(await self._formulate_consensus_statement(group))

        return consensus_points

    async def _generate_synthesized_content(self, resolved_evidence: List[Dict[str, Any]],
                                          synthesis_type: SynthesisType,
                                          topic: str,
                                          context: Dict[str, Any]) -> str:
        """Generate the main synthesized content"""

        # Get synthesis template
        template = self.synthesis_templates.get(synthesis_type, self.synthesis_templates["default"])

        # Generate sections based on template
        sections = {}

        # Introduction section
        sections["introduction"] = await self._generate_introduction(topic, resolved_evidence, context)

        # Background section
        sections["background"] = await self._generate_background(topic, resolved_evidence)

        # Main findings section
        sections["main_findings"] = await self._generate_main_findings(resolved_evidence, synthesis_type)

        # Evidence analysis section
        sections["evidence_analysis"] = await self._generate_evidence_analysis(resolved_evidence)

        # Discussion section
        sections["discussion"] = await self._generate_discussion(resolved_evidence, topic, context)

        # Conclusion section
        sections["conclusion"] = await self._generate_conclusion(resolved_evidence, topic)

        # Assemble final content
        synthesized_content = await self._assemble_content(sections, template, synthesis_type)

        return synthesized_content

    async def _generate_main_findings(self, resolved_evidence: List[Dict[str, Any]],
                                    synthesis_type: SynthesisType) -> str:
        """Generate main findings section"""

        findings_content = []

        # Group findings by theme
        thematic_groups = await self._group_findings_by_theme(resolved_evidence)

        findings_content.append("## Key Findings\n")

        for theme, findings in thematic_groups.items():
            findings_content.append(f"\n### {theme.replace('_', ' ').title()}\n")

            # Prioritize findings by evidence quality
            prioritized_findings = sorted(
                findings,
                key=lambda x: x.get("evidence_strength", 0),
                reverse=True
            )

            for finding in prioritized_findings[:5]:  # Top 5 findings per theme
                # Format finding with evidence level
                evidence_level = finding.get("evidence_level", "Unknown")
                finding_text = finding.get("text", "")
                source_count = finding.get("source_count", 1)

                findings_content.append(
                    f"- **{finding_text}** "
                    f"({evidence_level}, {source_count} source{'s' if source_count > 1 else ''})\n"
                )

        return "".join(findings_content)

    async def _generate_evidence_analysis(self, resolved_evidence: List[Dict[str, Any]]) -> str:
        """Generate evidence analysis section"""

        analysis_content = []
        analysis_content.append("## Evidence Analysis\n")

        # Quality assessment
        analysis_content.append("\n### Quality of Evidence\n")
        quality_distribution = await self._analyze_quality_distribution(resolved_evidence)
        for level, count in quality_distribution.items():
            analysis_content.append(f"- {level}: {count} studies\n")

        # Methodology assessment
        analysis_content.append("\n### Methodological Considerations\n")
        methodology_analysis = await self._analyze_methodologies(resolved_evidence)
        for consideration in methodology_analysis:
            analysis_content.append(f"- {consideration}\n")

        # Sample size analysis
        analysis_content.append("\n### Sample Size Analysis\n")
        sample_analysis = await self._analyze_sample_sizes(resolved_evidence)
        analysis_content.append(f"- Total participants across studies: {sample_analysis['total']}\n")
        analysis_content.append(f"- Average sample size: {sample_analysis['average']}\n")
        analysis_content.append(f"- Range: {sample_analysis['range']}\n")

        return "".join(analysis_content)

    async def _extract_clinical_implications(self, resolved_evidence: List[Dict[str, Any]],
                                           synthesis_type: SynthesisType) -> List[str]:
        """Extract clinical implications from synthesized evidence"""

        implications = []

        # Direct clinical implications from sources
        for evidence in resolved_evidence:
            source_implications = await self._extract_source_implications(evidence)
            implications.extend(source_implications)

        # Inferred implications from synthesis
        inferred_implications = await self._infer_clinical_implications(resolved_evidence, synthesis_type)
        implications.extend(inferred_implications)

        # Practice change implications
        practice_implications = await self._identify_practice_implications(resolved_evidence)
        implications.extend(practice_implications)

        # Remove duplicates and prioritize
        unique_implications = await self._deduplicate_and_prioritize_implications(implications)

        return unique_implications[:10]  # Top 10 implications

    async def _assess_synthesis_quality(self, synthesized_content: str,
                                      resolved_evidence: List[Dict[str, Any]],
                                      synthesis_type: SynthesisType) -> Dict[str, float]:
        """Assess the quality of the synthesis"""

        quality_metrics = {}

        # Comprehensiveness
        quality_metrics["comprehensiveness"] = await self._assess_comprehensiveness(
            synthesized_content, resolved_evidence
        )

        # Accuracy
        quality_metrics["accuracy"] = await self._assess_synthesis_accuracy(
            synthesized_content, resolved_evidence
        )

        # Clarity
        quality_metrics["clarity"] = await self._assess_content_clarity(synthesized_content)

        # Evidence integration
        quality_metrics["evidence_integration"] = await self._assess_evidence_integration(
            synthesized_content, resolved_evidence
        )

        # Bias assessment
        quality_metrics["bias_minimization"] = await self._assess_bias_minimization(
            synthesized_content, resolved_evidence
        )

        # Logical consistency
        quality_metrics["logical_consistency"] = await self._assess_logical_consistency(
            synthesized_content
        )

        return quality_metrics

    async def update_synthesis_with_new_evidence(self, synthesis_id: str,
                                               new_sources: List[SynthesisSource]) -> SynthesisResult:
        """Update existing synthesis with new evidence"""

        # Retrieve original synthesis
        original_synthesis = await self._retrieve_synthesis(synthesis_id)

        # Combine with new sources
        all_sources = original_synthesis.sources_used + new_sources

        # Re-synthesize with updated evidence
        updated_synthesis = await self.synthesize_knowledge(
            all_sources,
            original_synthesis.synthesis_type,
            original_synthesis.topic,
            {"update_mode": True, "original_synthesis": original_synthesis}
        )

        # Track changes
        changes = await self._track_synthesis_changes(original_synthesis, updated_synthesis)
        updated_synthesis.methodology_notes += f"\n\nUpdate Summary:\n{changes}"

        return updated_synthesis

    def _initialize_synthesis_framework(self):
        """Initialize synthesis templates and frameworks"""

        self.synthesis_templates = {
            SynthesisType.COMPREHENSIVE_REVIEW: {
                "sections": ["introduction", "background", "methods", "results", "discussion", "conclusion"],
                "emphasis": "thoroughness",
                "evidence_weighting": "balanced"
            },
            SynthesisType.CLINICAL_SUMMARY: {
                "sections": ["clinical_question", "evidence_summary", "recommendations", "implementation"],
                "emphasis": "practicality",
                "evidence_weighting": "clinical_relevance"
            },
            SynthesisType.TREATMENT_GUIDELINES: {
                "sections": ["scope", "recommendations", "evidence_basis", "implementation_considerations"],
                "emphasis": "actionability",
                "evidence_weighting": "evidence_level"
            },
            "default": {
                "sections": ["introduction", "main_findings", "discussion", "conclusion"],
                "emphasis": "clarity",
                "evidence_weighting": "quality"
            }
        }

        # Initialize quality weights for different synthesis types
        self.quality_weights = {
            SynthesisType.CLINICAL_SUMMARY: {
                "clinical_relevance": 0.4,
                "evidence_quality": 0.3,
                "practical_applicability": 0.3
            },
            SynthesisType.RESEARCH_OVERVIEW: {
                "methodological_rigor": 0.4,
                "evidence_quality": 0.4,
                "comprehensiveness": 0.2
            }
        }

# Global synthesis engine instance
synthesis_engine = IntelligentSynthesisEngine()