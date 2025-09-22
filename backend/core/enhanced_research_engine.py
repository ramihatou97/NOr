# backend/core/enhanced_research_engine.py
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import aiohttp
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import re

class ResearchSource(Enum):
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PERPLEXITY = "perplexity"
    LOCAL_KNOWLEDGE = "local_knowledge"
    CACHED_RESULTS = "cached_results"

@dataclass
class ResearchQuery:
    query: str
    domain: str
    urgency: int  # 1-5
    quality_threshold: float
    max_results: int
    source_preferences: List[ResearchSource]
    context: Dict[str, Any]

@dataclass
class ResearchResult:
    id: str
    title: str
    authors: List[str]
    abstract: str
    full_text: Optional[str]
    publication_date: datetime
    journal: str
    doi: Optional[str]
    pmid: Optional[str]
    source: ResearchSource
    relevance_score: float
    quality_score: float
    citation_count: int
    key_findings: List[str]
    medical_concepts: List[str]
    conflicts_detected: List[Dict[str, Any]]

@dataclass
class ResearchSynthesis:
    topic: str
    total_sources: int
    synthesis_text: str
    key_findings: List[str]
    consensus_points: List[str]
    contradictions: List[Dict[str, Any]]
    research_gaps: List[str]
    confidence_level: float
    last_updated: datetime

class EnhancedResearchEngine:
    def __init__(self):
        self.source_adapters = {
            ResearchSource.PUBMED: PubMedAdapter(),
            ResearchSource.SEMANTIC_SCHOLAR: SemanticScholarAdapter(),
            ResearchSource.PERPLEXITY: PerplexityAdapter(),
            ResearchSource.LOCAL_KNOWLEDGE: LocalKnowledgeAdapter(),
            ResearchSource.CACHED_RESULTS: CachedResultsAdapter()
        }

        self.result_cache = {}
        self.synthesis_cache = {}
        self.quality_filters = QualityFilterEngine()
        self.conflict_detector = ConflictDetectionEngine()

    async def intelligent_search(self, query: ResearchQuery) -> List[ResearchResult]:
        """Perform intelligent multi-source research with context awareness"""

        # Step 1: Query enhancement and optimization
        enhanced_query = await self._enhance_query(query)

        # Step 2: Source selection and prioritization
        optimal_sources = await self._select_optimal_sources(enhanced_query)

        # Step 3: Parallel search across sources
        search_tasks = []
        for source in optimal_sources:
            if source in self.source_adapters:
                task = self._search_source(source, enhanced_query)
                search_tasks.append(task)

        source_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Step 4: Combine and deduplicate results
        combined_results = await self._combine_results(source_results, enhanced_query)

        # Step 5: Quality filtering and ranking
        filtered_results = await self._filter_and_rank_results(combined_results, query)

        # Step 6: Conflict detection
        results_with_conflicts = await self._detect_conflicts(filtered_results)

        # Step 7: Cache results for future use
        await self._cache_results(query, results_with_conflicts)

        return results_with_conflicts

    async def synthesize_research(self, results: List[ResearchResult], topic: str,
                                synthesis_type: str = "comprehensive") -> ResearchSynthesis:
        """Create intelligent synthesis of research results"""

        # Step 1: Group results by themes
        thematic_groups = await self._group_by_themes(results)

        # Step 2: Extract key findings
        key_findings = await self._extract_key_findings(results)

        # Step 3: Identify consensus and contradictions
        consensus_analysis = await self._analyze_consensus(results, thematic_groups)

        # Step 4: Generate synthesis text
        synthesis_text = await self._generate_synthesis_text(
            results, thematic_groups, consensus_analysis, synthesis_type
        )

        # Step 5: Identify research gaps
        research_gaps = await self._identify_research_gaps(results, topic)

        # Step 6: Calculate overall confidence
        confidence_level = await self._calculate_synthesis_confidence(results, consensus_analysis)

        synthesis = ResearchSynthesis(
            topic=topic,
            total_sources=len(results),
            synthesis_text=synthesis_text,
            key_findings=key_findings,
            consensus_points=consensus_analysis["consensus"],
            contradictions=consensus_analysis["contradictions"],
            research_gaps=research_gaps,
            confidence_level=confidence_level,
            last_updated=datetime.now()
        )

        # Cache synthesis
        await self._cache_synthesis(topic, synthesis)

        return synthesis

    async def _enhance_query(self, query: ResearchQuery) -> ResearchQuery:
        """Enhance query with contextual information and domain knowledge"""

        enhanced_query = query.query

        # Add domain-specific terms
        domain_terms = await self._get_domain_terms(query.domain)
        if domain_terms:
            enhanced_query += f" ({' OR '.join(domain_terms[:3])})"

        # Add context-based filters
        if query.context.get("current_chapter"):
            chapter_concepts = await self._extract_chapter_concepts(query.context["current_chapter"])
            if chapter_concepts:
                enhanced_query += f" AND ({' OR '.join(chapter_concepts[:2])})"

        # Add temporal filters based on urgency
        if query.urgency >= 4:
            current_year = datetime.now().year
            enhanced_query += f" AND {current_year-2}:{current_year}[dp]"  # Recent papers only

        return ResearchQuery(
            query=enhanced_query,
            domain=query.domain,
            urgency=query.urgency,
            quality_threshold=query.quality_threshold,
            max_results=query.max_results,
            source_preferences=query.source_preferences,
            context=query.context
        )

    async def _select_optimal_sources(self, query: ResearchQuery) -> List[ResearchSource]:
        """Select optimal sources based on query characteristics"""

        sources = []

        # Always check cache first
        sources.append(ResearchSource.CACHED_RESULTS)

        # Local knowledge for basic medical concepts
        if any(term in query.query.lower() for term in ["anatomy", "physiology", "basic"]):
            sources.append(ResearchSource.LOCAL_KNOWLEDGE)

        # PubMed for medical research
        if query.domain in ["medicine", "neurosurgery", "surgery", "clinical"]:
            sources.append(ResearchSource.PUBMED)

        # Semantic Scholar for comprehensive coverage
        sources.append(ResearchSource.SEMANTIC_SCHOLAR)

        # Perplexity for recent insights and synthesis
        if query.urgency >= 3:
            sources.append(ResearchSource.PERPLEXITY)

        # Respect user preferences
        if query.source_preferences:
            # Prioritize user preferences while keeping essential sources
            preferred = [src for src in query.source_preferences if src not in sources]
            sources.extend(preferred)

        return sources[:4]  # Limit to 4 sources for performance

    async def _search_source(self, source: ResearchSource, query: ResearchQuery) -> List[ResearchResult]:
        """Search a specific source with error handling"""

        try:
            adapter = self.source_adapters[source]
            results = await adapter.search(query)

            # Add source information
            for result in results:
                result.source = source

            return results

        except Exception as e:
            print(f"Error searching {source}: {e}")
            return []

    async def _combine_results(self, source_results: List[List[ResearchResult]],
                             query: ResearchQuery) -> List[ResearchResult]:
        """Combine results from multiple sources and remove duplicates"""

        combined = []
        seen_titles = set()
        seen_dois = set()

        for results in source_results:
            if isinstance(results, list):  # Skip exceptions
                for result in results:
                    # Check for duplicates
                    is_duplicate = False

                    if result.doi and result.doi in seen_dois:
                        is_duplicate = True
                    elif result.title.lower() in seen_titles:
                        is_duplicate = True

                    if not is_duplicate:
                        combined.append(result)
                        seen_titles.add(result.title.lower())
                        if result.doi:
                            seen_dois.add(result.doi)

        return combined

    async def _filter_and_rank_results(self, results: List[ResearchResult],
                                     query: ResearchQuery) -> List[ResearchResult]:
        """Filter results by quality and rank by relevance"""

        # Quality filtering
        filtered_results = []
        for result in results:
            quality_score = await self.quality_filters.assess_quality(result)
            result.quality_score = quality_score

            if quality_score >= query.quality_threshold:
                filtered_results.append(result)

        # Relevance scoring
        for result in filtered_results:
            relevance_score = await self._calculate_relevance(result, query)
            result.relevance_score = relevance_score

        # Combined ranking
        for result in filtered_results:
            result.combined_score = (
                result.relevance_score * 0.4 +
                result.quality_score * 0.3 +
                min(1.0, result.citation_count / 100) * 0.2 +
                self._calculate_recency_score(result) * 0.1
            )

        # Sort by combined score
        filtered_results.sort(key=lambda x: x.combined_score, reverse=True)

        return filtered_results[:query.max_results]

    async def _detect_conflicts(self, results: List[ResearchResult]) -> List[ResearchResult]:
        """Detect conflicts between research results"""

        for i, result1 in enumerate(results):
            conflicts = []

            for j, result2 in enumerate(results[i+1:], i+1):
                conflict = await self.conflict_detector.detect_conflict(result1, result2)
                if conflict:
                    conflicts.append({
                        "conflicting_result_id": result2.id,
                        "conflict_type": conflict["type"],
                        "description": conflict["description"],
                        "severity": conflict["severity"]
                    })

            result1.conflicts_detected = conflicts

        return results

    async def _generate_synthesis_text(self, results: List[ResearchResult],
                                     thematic_groups: Dict[str, List[ResearchResult]],
                                     consensus_analysis: Dict[str, Any],
                                     synthesis_type: str) -> str:
        """Generate comprehensive synthesis text"""

        synthesis_parts = []

        # Introduction
        synthesis_parts.append(f"## Research Synthesis\n\n")
        synthesis_parts.append(f"Based on analysis of {len(results)} research sources, ")
        synthesis_parts.append(f"the following synthesis presents current understanding:\n\n")

        # Key findings by theme
        synthesis_parts.append("### Key Findings\n\n")
        for theme, theme_results in thematic_groups.items():
            synthesis_parts.append(f"**{theme.title()}:**\n")

            # Extract top findings for this theme
            theme_findings = []
            for result in theme_results[:3]:  # Top 3 results per theme
                if result.key_findings:
                    theme_findings.extend(result.key_findings[:2])

            for finding in theme_findings[:3]:  # Top 3 findings per theme
                synthesis_parts.append(f"- {finding}\n")

            synthesis_parts.append("\n")

        # Consensus points
        if consensus_analysis["consensus"]:
            synthesis_parts.append("### Areas of Consensus\n\n")
            for consensus in consensus_analysis["consensus"]:
                synthesis_parts.append(f"- {consensus}\n")
            synthesis_parts.append("\n")

        # Contradictions and gaps
        if consensus_analysis["contradictions"]:
            synthesis_parts.append("### Identified Contradictions\n\n")
            for contradiction in consensus_analysis["contradictions"]:
                synthesis_parts.append(f"- **{contradiction['topic']}**: {contradiction['description']}\n")
            synthesis_parts.append("\n")

        # Clinical implications (for medical domains)
        synthesis_parts.append("### Clinical Implications\n\n")
        clinical_implications = await self._extract_clinical_implications(results)
        for implication in clinical_implications:
            synthesis_parts.append(f"- {implication}\n")

        return "".join(synthesis_parts)

class QualityFilterEngine:
    """Engine for assessing research quality"""

    async def assess_quality(self, result: ResearchResult) -> float:
        """Assess the quality of a research result"""

        quality_factors = {
            "journal_impact": 0.0,
            "citation_count": 0.0,
            "author_credibility": 0.0,
            "methodology_quality": 0.0,
            "recency": 0.0
        }

        # Journal impact factor
        quality_factors["journal_impact"] = await self._assess_journal_quality(result.journal)

        # Citation count (normalized)
        quality_factors["citation_count"] = min(1.0, result.citation_count / 50)

        # Author credibility
        quality_factors["author_credibility"] = await self._assess_author_credibility(result.authors)

        # Methodology assessment from abstract
        quality_factors["methodology_quality"] = await self._assess_methodology(result.abstract)

        # Recency factor
        years_old = (datetime.now() - result.publication_date).days / 365
        quality_factors["recency"] = max(0.1, 1.0 - (years_old / 10))  # Decay over 10 years

        # Weighted quality score
        weights = {
            "journal_impact": 0.25,
            "citation_count": 0.20,
            "author_credibility": 0.15,
            "methodology_quality": 0.25,
            "recency": 0.15
        }

        quality_score = sum(
            quality_factors[factor] * weights[factor]
            for factor in quality_factors
        )

        return min(1.0, quality_score)

    async def _assess_journal_quality(self, journal: str) -> float:
        """Assess journal quality/impact factor"""

        # High-impact medical journals
        high_impact_journals = {
            "nature", "science", "cell", "lancet", "nejm", "jama",
            "nature medicine", "nature neuroscience", "neuron",
            "journal of neurosurgery", "neurosurgery"
        }

        journal_lower = journal.lower()

        for high_impact in high_impact_journals:
            if high_impact in journal_lower:
                return 1.0

        # Medium impact indicators
        if any(term in journal_lower for term in ["journal", "international", "clinical"]):
            return 0.7

        return 0.5  # Default for unknown journals

    async def _assess_methodology(self, abstract: str) -> float:
        """Assess methodology quality from abstract"""

        methodology_indicators = {
            "high_quality": ["randomized", "controlled", "systematic review", "meta-analysis",
                           "double-blind", "placebo-controlled", "prospective"],
            "medium_quality": ["cohort", "case-control", "retrospective", "observational"],
            "low_quality": ["case report", "case series", "opinion", "editorial"]
        }

        abstract_lower = abstract.lower()

        # Check for high quality indicators
        high_score = sum(1 for indicator in methodology_indicators["high_quality"]
                        if indicator in abstract_lower)
        if high_score > 0:
            return min(1.0, 0.8 + (high_score * 0.1))

        # Check for medium quality indicators
        medium_score = sum(1 for indicator in methodology_indicators["medium_quality"]
                          if indicator in abstract_lower)
        if medium_score > 0:
            return 0.6

        # Check for low quality indicators
        low_score = sum(1 for indicator in methodology_indicators["low_quality"]
                       if indicator in abstract_lower)
        if low_score > 0:
            return 0.3

        return 0.5  # Default score

class ConflictDetectionEngine:
    """Engine for detecting conflicts between research results"""

    async def detect_conflict(self, result1: ResearchResult, result2: ResearchResult) -> Optional[Dict[str, Any]]:
        """Detect conflicts between two research results"""

        # Extract claims from abstracts
        claims1 = await self._extract_claims(result1.abstract)
        claims2 = await self._extract_claims(result2.abstract)

        # Check for contradictory claims
        for claim1 in claims1:
            for claim2 in claims2:
                conflict = await self._compare_claims(claim1, claim2)
                if conflict:
                    return {
                        "type": conflict["type"],
                        "description": conflict["description"],
                        "severity": conflict["severity"],
                        "claim1": claim1,
                        "claim2": claim2
                    }

        return None

    async def _extract_claims(self, abstract: str) -> List[str]:
        """Extract factual claims from abstract"""

        # Simple claim extraction (would use NLP in practice)
        sentences = abstract.split('. ')

        claim_indicators = [
            "showed", "demonstrated", "found", "revealed", "indicated",
            "concluded", "suggested", "reported", "observed"
        ]

        claims = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in claim_indicators):
                claims.append(sentence.strip())

        return claims

    async def _compare_claims(self, claim1: str, claim2: str) -> Optional[Dict[str, Any]]:
        """Compare two claims for conflicts"""

        # Simple contradiction detection (would use advanced NLP)
        contradiction_patterns = [
            ("increase", "decrease"),
            ("improve", "worsen"),
            ("effective", "ineffective"),
            ("safe", "unsafe"),
            ("beneficial", "harmful")
        ]

        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()

        for positive, negative in contradiction_patterns:
            if positive in claim1_lower and negative in claim2_lower:
                return {
                    "type": "contradiction",
                    "description": f"Conflicting findings: '{positive}' vs '{negative}'",
                    "severity": "high"
                }
            elif negative in claim1_lower and positive in claim2_lower:
                return {
                    "type": "contradiction",
                    "description": f"Conflicting findings: '{negative}' vs '{positive}'",
                    "severity": "high"
                }

        return None

# Source adapter implementations
class PubMedAdapter:
    async def search(self, query: ResearchQuery) -> List[ResearchResult]:
        """Search PubMed database"""
        # Implementation would use actual PubMed API
        # This is a placeholder for the structure
        return []

class SemanticScholarAdapter:
    async def search(self, query: ResearchQuery) -> List[ResearchResult]:
        """Search Semantic Scholar database"""
        # Implementation would use actual Semantic Scholar API
        return []

class PerplexityAdapter:
    async def search(self, query: ResearchQuery) -> List[ResearchResult]:
        """Search using Perplexity AI"""
        # Implementation would use actual Perplexity API
        return []

class LocalKnowledgeAdapter:
    async def search(self, query: ResearchQuery) -> List[ResearchResult]:
        """Search local knowledge base"""
        # Implementation would search local medical knowledge
        return []

class CachedResultsAdapter:
    async def search(self, query: ResearchQuery) -> List[ResearchResult]:
        """Search cached results"""
        # Implementation would search cached research results
        return []

# Global research engine instance
research_engine = EnhancedResearchEngine()