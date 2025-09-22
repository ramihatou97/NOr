# backend/core/knowledge_graph.py
from typing import Dict, List, Set, Optional, Tuple, Any
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import asyncio
import json

@dataclass
class ConceptNode:
    concept_id: str
    name: str
    concept_type: str  # medical_condition, procedure, medication, anatomy, etc.
    confidence: float
    last_updated: datetime
    sources: List[str]
    metadata: Dict[str, Any]

@dataclass
class ConceptRelation:
    source_concept: str
    target_concept: str
    relation_type: str  # causes, treats, prevents, contraindicated, etc.
    strength: float
    confidence: float
    evidence_sources: List[str]
    last_validated: datetime

@dataclass
class KnowledgeEvolution:
    new_concepts: List[ConceptNode]
    updated_relations: List[ConceptRelation]
    deprecated_knowledge: List[str]
    confidence_adjustments: Dict[str, float]
    reasoning: str

class SelfEvolvingKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.concept_embeddings = {}
        self.relation_strengths = defaultdict(float)
        self.temporal_validity = {}
        self.evolution_history = []
        self.learning_rate = 0.1

    async def add_concept(self, concept: ConceptNode):
        """Add a new medical concept to the knowledge graph"""

        self.graph.add_node(
            concept.concept_id,
            name=concept.name,
            concept_type=concept.concept_type,
            confidence=concept.confidence,
            last_updated=concept.last_updated,
            sources=concept.sources,
            metadata=concept.metadata
        )

        # Generate embeddings for semantic relationships
        self.concept_embeddings[concept.concept_id] = await self._generate_concept_embedding(concept)

        # Set temporal validity
        self.temporal_validity[concept.concept_id] = await self._calculate_temporal_validity(concept)

    async def add_relation(self, relation: ConceptRelation):
        """Add a relationship between medical concepts"""

        self.graph.add_edge(
            relation.source_concept,
            relation.target_concept,
            relation_type=relation.relation_type,
            strength=relation.strength,
            confidence=relation.confidence,
            evidence_sources=relation.evidence_sources,
            last_validated=relation.last_validated
        )

        # Update relation strength tracking
        relation_key = f"{relation.source_concept}:{relation.target_concept}:{relation.relation_type}"
        self.relation_strengths[relation_key] = relation.strength

    async def evolve_knowledge(self, new_information: List[Dict[str, Any]]) -> KnowledgeEvolution:
        """Evolve the knowledge graph based on new information"""

        evolution = KnowledgeEvolution(
            new_concepts=[],
            updated_relations=[],
            deprecated_knowledge=[],
            confidence_adjustments={},
            reasoning=""
        )

        for info in new_information:
            # Process new concepts
            if info["type"] == "new_concept":
                concept = await self._process_new_concept(info)
                if concept:
                    evolution.new_concepts.append(concept)
                    await self.add_concept(concept)

            # Process relation updates
            elif info["type"] == "relation_update":
                relation = await self._process_relation_update(info)
                if relation:
                    evolution.updated_relations.append(relation)
                    await self.add_relation(relation)

            # Process contradictions
            elif info["type"] == "contradiction":
                deprecated = await self._process_contradiction(info)
                evolution.deprecated_knowledge.extend(deprecated)

            # Process confidence updates
            elif info["type"] == "confidence_update":
                adjustments = await self._process_confidence_update(info)
                evolution.confidence_adjustments.update(adjustments)

        # Discover implicit relationships
        implicit_relations = await self._discover_implicit_relationships()
        evolution.updated_relations.extend(implicit_relations)

        # Validate and clean outdated knowledge
        outdated_knowledge = await self._identify_outdated_knowledge()
        evolution.deprecated_knowledge.extend(outdated_knowledge)

        # Store evolution history
        self.evolution_history.append({
            "timestamp": datetime.now(),
            "evolution": evolution,
            "trigger": "new_information_processing"
        })

        return evolution

    async def semantic_search(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform semantic search on the knowledge graph"""

        # Generate query embedding
        query_embedding = await self._generate_query_embedding(query)

        # Calculate similarity scores
        similarities = {}
        for concept_id, embedding in self.concept_embeddings.items():
            similarity = await self._calculate_cosine_similarity(query_embedding, embedding)
            similarities[concept_id] = similarity

        # Rank by similarity and graph centrality
        ranked_concepts = []
        for concept_id, similarity in similarities.items():
            if similarity > 0.5:  # Threshold for relevance
                concept_data = self.graph.nodes[concept_id]
                centrality = await self._calculate_concept_centrality(concept_id)
                confidence = concept_data.get("confidence", 0.5)

                # Combined score
                score = similarity * 0.4 + centrality * 0.3 + confidence * 0.3

                ranked_concepts.append({
                    "concept_id": concept_id,
                    "name": concept_data["name"],
                    "concept_type": concept_data["concept_type"],
                    "similarity": similarity,
                    "centrality": centrality,
                    "confidence": confidence,
                    "combined_score": score,
                    "related_concepts": await self._get_related_concepts(concept_id, context)
                })

        # Sort by combined score
        ranked_concepts.sort(key=lambda x: x["combined_score"], reverse=True)

        return ranked_concepts[:10]  # Return top 10 results

    async def discover_knowledge_gaps(self, domain: str) -> List[Dict[str, Any]]:
        """Discover gaps in the knowledge graph for a specific domain"""

        gaps = []

        # Get all concepts in domain
        domain_concepts = [
            node_id for node_id, data in self.graph.nodes(data=True)
            if data.get("concept_type") == domain or domain.lower() in data.get("name", "").lower()
        ]

        # Analyze connectivity patterns
        for concept_id in domain_concepts:
            # Check for missing standard relationships
            expected_relations = await self._get_expected_relations(concept_id)
            actual_relations = list(self.graph.edges(concept_id, data=True))

            for expected in expected_relations:
                if not any(rel[2].get("relation_type") == expected["type"] for rel in actual_relations):
                    gaps.append({
                        "gap_type": "missing_relation",
                        "concept": concept_id,
                        "missing_relation": expected["type"],
                        "expected_target": expected.get("target"),
                        "confidence": expected.get("confidence", 0.7),
                        "reasoning": f"Standard {expected['type']} relation missing for {self.graph.nodes[concept_id]['name']}"
                    })

        # Identify isolated concepts
        isolated_concepts = [
            node_id for node_id in domain_concepts
            if self.graph.degree(node_id) < 2
        ]

        for concept_id in isolated_concepts:
            gaps.append({
                "gap_type": "isolated_concept",
                "concept": concept_id,
                "reasoning": f"Concept {self.graph.nodes[concept_id]['name']} has insufficient connections",
                "suggested_actions": ["find_related_concepts", "validate_relationships"]
            })

        # Identify concepts with low confidence
        low_confidence_concepts = [
            node_id for node_id in domain_concepts
            if self.graph.nodes[node_id].get("confidence", 1.0) < 0.6
        ]

        for concept_id in low_confidence_concepts:
            gaps.append({
                "gap_type": "low_confidence",
                "concept": concept_id,
                "current_confidence": self.graph.nodes[concept_id].get("confidence"),
                "reasoning": "Concept needs additional validation",
                "suggested_actions": ["find_additional_sources", "cross_validate"]
            })

        return gaps

    async def predict_concept_relationships(self, concept_id: str) -> List[Dict[str, Any]]:
        """Predict potential relationships for a concept"""

        predictions = []

        if concept_id not in self.graph:
            return predictions

        concept_data = self.graph.nodes[concept_id]
        concept_embedding = self.concept_embeddings.get(concept_id)

        if not concept_embedding:
            return predictions

        # Find similar concepts
        similar_concepts = []
        for other_id, other_embedding in self.concept_embeddings.items():
            if other_id != concept_id:
                similarity = await self._calculate_cosine_similarity(concept_embedding, other_embedding)
                if similarity > 0.7:
                    similar_concepts.append((other_id, similarity))

        # Predict relationships based on similar concepts
        for similar_id, similarity in similar_concepts:
            similar_relations = list(self.graph.edges(similar_id, data=True))

            for _, target, rel_data in similar_relations:
                if target != concept_id and not self.graph.has_edge(concept_id, target):
                    # Predict this relationship for our concept
                    confidence = similarity * rel_data.get("confidence", 0.5) * 0.8  # Reduced confidence for prediction

                    predictions.append({
                        "predicted_relation": rel_data.get("relation_type"),
                        "target_concept": target,
                        "target_name": self.graph.nodes[target]["name"],
                        "confidence": confidence,
                        "reasoning": f"Similar to {self.graph.nodes[similar_id]['name']} (similarity: {similarity:.2f})",
                        "evidence_strength": rel_data.get("strength", 0.5)
                    })

        # Remove duplicates and sort by confidence
        seen = set()
        unique_predictions = []
        for pred in predictions:
            key = f"{pred['predicted_relation']}:{pred['target_concept']}"
            if key not in seen:
                seen.add(key)
                unique_predictions.append(pred)

        unique_predictions.sort(key=lambda x: x["confidence"], reverse=True)

        return unique_predictions[:5]  # Return top 5 predictions

    async def validate_knowledge_consistency(self) -> Dict[str, Any]:
        """Validate the consistency of knowledge in the graph"""

        inconsistencies = {
            "contradictions": [],
            "circular_dependencies": [],
            "confidence_conflicts": [],
            "temporal_inconsistencies": []
        }

        # Check for contradictions
        for node_id in self.graph.nodes():
            outgoing_relations = list(self.graph.edges(node_id, data=True))

            for i, (_, target1, rel1) in enumerate(outgoing_relations):
                for j, (_, target2, rel2) in enumerate(outgoing_relations[i+1:], i+1):
                    contradiction = await self._check_relation_contradiction(rel1, rel2, target1, target2)
                    if contradiction:
                        inconsistencies["contradictions"].append(contradiction)

        # Check for circular dependencies
        try:
            cycles = list(nx.simple_cycles(self.graph))
            for cycle in cycles:
                if len(cycle) > 1:
                    inconsistencies["circular_dependencies"].append({
                        "cycle": cycle,
                        "cycle_names": [self.graph.nodes[node]["name"] for node in cycle],
                        "severity": "high" if len(cycle) <= 3 else "medium"
                    })
        except:
            pass  # Graph might be too large for cycle detection

        # Check confidence conflicts
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            source_confidence = self.graph.nodes[source].get("confidence", 0.5)
            target_confidence = self.graph.nodes[target].get("confidence", 0.5)
            relation_confidence = data.get("confidence", 0.5)

            # Relation confidence shouldn't exceed concept confidences
            max_concept_confidence = max(source_confidence, target_confidence)
            if relation_confidence > max_concept_confidence + 0.2:
                inconsistencies["confidence_conflicts"].append({
                    "source": source,
                    "target": target,
                    "relation_confidence": relation_confidence,
                    "max_concept_confidence": max_concept_confidence,
                    "issue": "Relation confidence exceeds concept confidence"
                })

        return inconsistencies

    async def _generate_concept_embedding(self, concept: ConceptNode) -> np.ndarray:
        """Generate embedding for a medical concept"""
        # This would use a medical language model in practice
        # For now, using a simplified approach

        text = f"{concept.name} {concept.concept_type} {' '.join(concept.sources)}"
        # Simplified embedding generation (would use actual model)
        embedding = np.random.random(384)  # Placeholder
        return embedding

    async def _calculate_temporal_validity(self, concept: ConceptNode) -> float:
        """Calculate how long this concept is likely to remain valid"""

        # Medical knowledge decay rates by type
        decay_rates = {
            "anatomy": 0.99,  # Very stable
            "physiology": 0.95,  # Mostly stable
            "treatment": 0.85,  # Changes with new research
            "medication": 0.80,  # Frequent updates
            "diagnosis": 0.90,  # Moderately stable
            "procedure": 0.85   # Updates with technology
        }

        base_validity = decay_rates.get(concept.concept_type, 0.85)

        # Adjust based on source quality and recency
        source_quality_multiplier = 1.0
        if any("pubmed" in source.lower() for source in concept.sources):
            source_quality_multiplier = 1.1

        # Recent concepts start with higher validity
        days_since_update = (datetime.now() - concept.last_updated).days
        recency_multiplier = max(0.8, 1.0 - (days_since_update / 365) * 0.1)

        return min(1.0, base_validity * source_quality_multiplier * recency_multiplier)

    async def _discover_implicit_relationships(self) -> List[ConceptRelation]:
        """Discover relationships that are implied but not explicitly stated"""

        implicit_relations = []

        # Transitivity analysis
        for node in self.graph.nodes():
            # Find paths of length 2
            for neighbor1 in self.graph.neighbors(node):
                for neighbor2 in self.graph.neighbors(neighbor1):
                    if neighbor2 != node and not self.graph.has_edge(node, neighbor2):

                        # Get relation types
                        rel1_data = self.graph.get_edge_data(node, neighbor1)
                        rel2_data = self.graph.get_edge_data(neighbor1, neighbor2)

                        if rel1_data and rel2_data:
                            # Check for transitive relationships
                            implicit_relation = await self._check_transitivity(
                                node, neighbor1, neighbor2, rel1_data, rel2_data
                            )

                            if implicit_relation:
                                implicit_relations.append(implicit_relation)

        return implicit_relations

    async def get_knowledge_summary(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of the knowledge graph state"""

        summary = {
            "total_concepts": self.graph.number_of_nodes(),
            "total_relations": self.graph.number_of_edges(),
            "concept_types": defaultdict(int),
            "relation_types": defaultdict(int),
            "average_confidence": 0.0,
            "coverage_gaps": [],
            "recent_updates": []
        }

        # Analyze concept types
        for node_id, data in self.graph.nodes(data=True):
            concept_type = data.get("concept_type", "unknown")
            summary["concept_types"][concept_type] += 1

        # Analyze relation types
        for _, _, data in self.graph.edges(data=True):
            relation_type = data.get("relation_type", "unknown")
            summary["relation_types"][relation_type] += 1

        # Calculate average confidence
        confidences = [data.get("confidence", 0.5) for _, data in self.graph.nodes(data=True)]
        summary["average_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0

        # Find coverage gaps
        if domain:
            summary["coverage_gaps"] = await self.discover_knowledge_gaps(domain)

        # Recent updates
        recent_threshold = datetime.now() - timedelta(days=7)
        recent_nodes = [
            {"id": node_id, "name": data["name"], "updated": data["last_updated"]}
            for node_id, data in self.graph.nodes(data=True)
            if data.get("last_updated", datetime.min) > recent_threshold
        ]
        summary["recent_updates"] = sorted(recent_nodes, key=lambda x: x["updated"], reverse=True)[:10]

        return summary

# Global knowledge graph instance
knowledge_graph = SelfEvolvingKnowledgeGraph()