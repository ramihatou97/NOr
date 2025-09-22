# backend/core/pdf_textbook_processor.py
"""
Advanced PDF Textbook Chapter Processing System
Intelligent extraction, indexing, and semantic processing of medical textbooks
"""

import asyncio
import fitz  # PyMuPDF for advanced PDF processing
import PyPDF2
import pdfplumber
import camelot  # For table extraction
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json
from datetime import datetime
import hashlib
import pytesseract
from PIL import Image
import openai
import tiktoken
from sentence_transformers import SentenceTransformer
import spacy
from pathlib import Path
import logging

class TextbookType(Enum):
    NEUROSURGERY = "neurosurgery"
    ANATOMY = "anatomy"
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"
    PHARMACOLOGY = "pharmacology"
    GENERAL_MEDICINE = "general_medicine"

class ContentType(Enum):
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    FIGURE = "figure"
    TABLE = "table"
    REFERENCE = "reference"
    CASE_STUDY = "case_study"
    ALGORITHM = "algorithm"

@dataclass
class TextbookMetadata:
    title: str
    authors: List[str]
    edition: str
    publisher: str
    publication_year: int
    isbn: Optional[str]
    specialty: TextbookType
    total_pages: int
    language: str = "en"
    doi: Optional[str] = None

@dataclass
class ChapterStructure:
    chapter_number: int
    title: str
    start_page: int
    end_page: int
    sections: List[Dict[str, Any]]
    subsections: List[Dict[str, Any]]
    learning_objectives: List[str]
    key_concepts: List[str]
    figures: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    references: List[str]
    case_studies: List[Dict[str, Any]]

@dataclass
class ExtractedContent:
    content_id: str
    content_type: ContentType
    title: str
    content: str
    page_numbers: List[int]
    hierarchy_level: int
    parent_section: Optional[str]
    medical_concepts: List[str]
    anatomical_references: List[str]
    procedures_mentioned: List[str]
    pathologies_discussed: List[str]
    semantic_embedding: Optional[List[float]]
    quality_score: float
    clinical_relevance: float
    educational_value: float

@dataclass
class ProcessedTextbook:
    textbook_id: str
    metadata: TextbookMetadata
    chapters: List[ChapterStructure]
    extracted_content: List[ExtractedContent]
    semantic_index: Dict[str, Any]
    medical_terminology_index: Dict[str, List[str]]
    cross_references: Dict[str, List[str]]
    processing_timestamp: datetime
    total_content_blocks: int

class PDFTextbookProcessor:
    def __init__(self):
        # Initialize NLP models
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

        # Medical entity recognition
        self.medical_ner = self._load_medical_ner_model()

        # Tokenizer for content chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Processing statistics
        self.processing_stats = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def process_textbook_pdf(self, pdf_path: str,
                                 metadata: TextbookMetadata,
                                 processing_options: Dict[str, Any] = None) -> ProcessedTextbook:
        """Comprehensive processing of medical textbook PDF"""

        if processing_options is None:
            processing_options = {
                "extract_images": True,
                "extract_tables": True,
                "ocr_scanned_pages": True,
                "semantic_analysis": True,
                "medical_entity_extraction": True,
                "create_cross_references": True,
                "quality_assessment": True
            }

        self.logger.info(f"Starting processing of textbook: {metadata.title}")

        # Generate unique textbook ID
        textbook_id = hashlib.md5(f"{pdf_path}{datetime.now()}".encode()).hexdigest()

        # Phase 1: PDF Structure Analysis
        pdf_structure = await self._analyze_pdf_structure(pdf_path)

        # Phase 2: Chapter Detection and Segmentation
        chapters = await self._detect_and_segment_chapters(pdf_path, pdf_structure, metadata)

        # Phase 3: Content Extraction
        extracted_content = await self._extract_all_content(
            pdf_path, chapters, processing_options
        )

        # Phase 4: Medical Entity Recognition and Processing
        processed_content = await self._process_medical_content(
            extracted_content, metadata.specialty
        )

        # Phase 5: Semantic Indexing
        semantic_index = await self._create_semantic_index(processed_content)

        # Phase 6: Cross-Reference Generation
        cross_references = await self._generate_cross_references(processed_content)

        # Phase 7: Medical Terminology Index
        terminology_index = await self._create_medical_terminology_index(processed_content)

        processed_textbook = ProcessedTextbook(
            textbook_id=textbook_id,
            metadata=metadata,
            chapters=chapters,
            extracted_content=processed_content,
            semantic_index=semantic_index,
            medical_terminology_index=terminology_index,
            cross_references=cross_references,
            processing_timestamp=datetime.now(),
            total_content_blocks=len(processed_content)
        )

        # Save processed textbook
        await self._save_processed_textbook(processed_textbook)

        self.logger.info(f"Completed processing: {len(chapters)} chapters, {len(processed_content)} content blocks")

        return processed_textbook

    async def _analyze_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze PDF structure to understand layout and organization"""

        structure = {
            "total_pages": 0,
            "page_layouts": [],
            "font_analysis": {},
            "heading_patterns": [],
            "table_of_contents": [],
            "index_pages": [],
            "reference_pages": []
        }

        # Open PDF with multiple libraries for comprehensive analysis
        with fitz.open(pdf_path) as pdf_doc:
            structure["total_pages"] = pdf_doc.page_count

            # Analyze first 50 pages for structure patterns
            analysis_pages = min(50, pdf_doc.page_count)

            for page_num in range(analysis_pages):
                page = pdf_doc[page_num]

                # Font analysis for heading detection
                blocks = page.get_text("dict")
                fonts_on_page = await self._analyze_page_fonts(blocks)

                # Update font analysis
                for font, properties in fonts_on_page.items():
                    if font not in structure["font_analysis"]:
                        structure["font_analysis"][font] = properties
                    else:
                        # Merge properties
                        structure["font_analysis"][font]["occurrences"] += properties["occurrences"]

                # Detect potential headings
                headings = await self._detect_headings_on_page(blocks, page_num)
                structure["heading_patterns"].extend(headings)

                # Check for table of contents patterns
                toc_candidates = await self._detect_toc_patterns(page.get_text(), page_num)
                if toc_candidates:
                    structure["table_of_contents"].extend(toc_candidates)

        return structure

    async def _detect_and_segment_chapters(self, pdf_path: str,
                                         pdf_structure: Dict[str, Any],
                                         metadata: TextbookMetadata) -> List[ChapterStructure]:
        """Detect chapter boundaries and create chapter structure"""

        chapters = []

        # Identify chapter heading patterns
        chapter_patterns = await self._identify_chapter_patterns(
            pdf_structure["heading_patterns"], metadata.specialty
        )

        with fitz.open(pdf_path) as pdf_doc:
            current_chapter = None

            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                page_text = page.get_text()

                # Check for chapter start
                chapter_match = await self._match_chapter_pattern(
                    page_text, chapter_patterns, page_num
                )

                if chapter_match:
                    # Save previous chapter if exists
                    if current_chapter:
                        current_chapter.end_page = page_num - 1
                        await self._finalize_chapter_structure(current_chapter, pdf_doc)
                        chapters.append(current_chapter)

                    # Start new chapter
                    current_chapter = ChapterStructure(
                        chapter_number=chapter_match["chapter_number"],
                        title=chapter_match["title"],
                        start_page=page_num,
                        end_page=pdf_doc.page_count - 1,  # Temporary
                        sections=[],
                        subsections=[],
                        learning_objectives=[],
                        key_concepts=[],
                        figures=[],
                        tables=[],
                        references=[],
                        case_studies=[]
                    )

                # Process current page for chapter content
                if current_chapter:
                    await self._process_chapter_page(
                        current_chapter, page, page_num, page_text
                    )

            # Finalize last chapter
            if current_chapter:
                await self._finalize_chapter_structure(current_chapter, pdf_doc)
                chapters.append(current_chapter)

        return chapters

    async def _extract_all_content(self, pdf_path: str,
                                 chapters: List[ChapterStructure],
                                 options: Dict[str, Any]) -> List[ExtractedContent]:
        """Extract all content from PDF chapters"""

        all_content = []

        with fitz.open(pdf_path) as pdf_doc:
            for chapter in chapters:
                chapter_content = await self._extract_chapter_content(
                    pdf_doc, chapter, options
                )
                all_content.extend(chapter_content)

        return all_content

    async def _extract_chapter_content(self, pdf_doc,
                                     chapter: ChapterStructure,
                                     options: Dict[str, Any]) -> List[ExtractedContent]:
        """Extract comprehensive content from a single chapter"""

        content_blocks = []

        for page_num in range(chapter.start_page, chapter.end_page + 1):
            page = pdf_doc[page_num]

            # Extract text content
            if options.get("extract_text", True):
                text_blocks = await self._extract_text_blocks(page, page_num, chapter)
                content_blocks.extend(text_blocks)

            # Extract images and figures
            if options.get("extract_images", True):
                image_blocks = await self._extract_image_blocks(page, page_num, chapter)
                content_blocks.extend(image_blocks)

            # Extract tables
            if options.get("extract_tables", True):
                table_blocks = await self._extract_table_blocks(page, page_num, chapter)
                content_blocks.extend(table_blocks)

            # OCR for scanned content
            if options.get("ocr_scanned_pages", True):
                ocr_blocks = await self._ocr_scanned_content(page, page_num, chapter)
                content_blocks.extend(ocr_blocks)

        return content_blocks

    async def _extract_text_blocks(self, page, page_num: int,
                                 chapter: ChapterStructure) -> List[ExtractedContent]:
        """Extract and process text blocks from page"""

        text_blocks = []

        # Get text with position information
        blocks = page.get_text("dict")

        for block in blocks.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")

                if len(block_text.strip()) > 50:  # Minimum content length
                    content_id = f"{chapter.chapter_number}_{page_num}_{len(text_blocks)}"

                    # Determine content type and hierarchy
                    content_type, hierarchy_level = await self._classify_text_block(
                        block_text, block, chapter
                    )

                    extracted_content = ExtractedContent(
                        content_id=content_id,
                        content_type=content_type,
                        title=await self._extract_section_title(block_text),
                        content=block_text.strip(),
                        page_numbers=[page_num],
                        hierarchy_level=hierarchy_level,
                        parent_section=await self._find_parent_section(hierarchy_level, text_blocks),
                        medical_concepts=[],  # To be filled by medical processing
                        anatomical_references=[],
                        procedures_mentioned=[],
                        pathologies_discussed=[],
                        semantic_embedding=None,  # To be generated
                        quality_score=0.0,  # To be calculated
                        clinical_relevance=0.0,
                        educational_value=0.0
                    )

                    text_blocks.append(extracted_content)

        return text_blocks

    async def _extract_image_blocks(self, page, page_num: int,
                                  chapter: ChapterStructure) -> List[ExtractedContent]:
        """Extract images and figures from page"""

        image_blocks = []

        # Get images from page
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            # Extract image
            xref = img[0]
            pix = fitz.Pixmap(page.parent, xref)

            if pix.n - pix.alpha < 4:  # GRAY or RGB
                img_data = pix.tobytes("png")

                # Process image for medical content
                image_analysis = await self._analyze_medical_image(img_data, page_num)

                content_id = f"{chapter.chapter_number}_{page_num}_img_{img_index}"

                image_content = ExtractedContent(
                    content_id=content_id,
                    content_type=ContentType.FIGURE,
                    title=image_analysis.get("caption", f"Figure {chapter.chapter_number}.{img_index + 1}"),
                    content=image_analysis.get("description", "Medical image or diagram"),
                    page_numbers=[page_num],
                    hierarchy_level=3,  # Figures are typically level 3
                    parent_section=None,  # To be linked later
                    medical_concepts=image_analysis.get("medical_concepts", []),
                    anatomical_references=image_analysis.get("anatomical_references", []),
                    procedures_mentioned=image_analysis.get("procedures", []),
                    pathologies_discussed=image_analysis.get("pathologies", []),
                    semantic_embedding=None,
                    quality_score=image_analysis.get("quality_score", 0.7),
                    clinical_relevance=image_analysis.get("clinical_relevance", 0.6),
                    educational_value=image_analysis.get("educational_value", 0.8)
                )

                image_blocks.append(image_content)

            pix = None  # Free memory

        return image_blocks

    async def _extract_table_blocks(self, page, page_num: int,
                                  chapter: ChapterStructure) -> List[ExtractedContent]:
        """Extract tables using camelot-py"""

        table_blocks = []

        try:
            # Use camelot to extract tables
            tables = camelot.read_pdf(
                page.parent.name,
                pages=str(page_num + 1),  # camelot uses 1-indexed pages
                flavor='lattice'  # or 'stream' for tables without borders
            )

            for table_index, table in enumerate(tables):
                if table.accuracy > 50:  # Minimum accuracy threshold
                    # Convert table to structured format
                    table_data = table.df.to_dict('records')
                    table_text = table.df.to_string()

                    # Analyze table for medical content
                    table_analysis = await self._analyze_medical_table(table_data, table_text)

                    content_id = f"{chapter.chapter_number}_{page_num}_table_{table_index}"

                    table_content = ExtractedContent(
                        content_id=content_id,
                        content_type=ContentType.TABLE,
                        title=table_analysis.get("title", f"Table {chapter.chapter_number}.{table_index + 1}"),
                        content=json.dumps(table_data, indent=2),
                        page_numbers=[page_num],
                        hierarchy_level=3,
                        parent_section=None,
                        medical_concepts=table_analysis.get("medical_concepts", []),
                        anatomical_references=table_analysis.get("anatomical_references", []),
                        procedures_mentioned=table_analysis.get("procedures", []),
                        pathologies_discussed=table_analysis.get("pathologies", []),
                        semantic_embedding=None,
                        quality_score=table.accuracy / 100,
                        clinical_relevance=table_analysis.get("clinical_relevance", 0.7),
                        educational_value=table_analysis.get("educational_value", 0.8)
                    )

                    table_blocks.append(table_content)

        except Exception as e:
            self.logger.warning(f"Table extraction failed for page {page_num}: {e}")

        return table_blocks

    async def _process_medical_content(self, content_blocks: List[ExtractedContent],
                                     specialty: TextbookType) -> List[ExtractedContent]:
        """Process extracted content for medical entities and concepts"""

        processed_content = []

        for content in content_blocks:
            # Extract medical entities
            medical_entities = await self._extract_medical_entities(
                content.content, specialty
            )

            # Update content with medical information
            content.medical_concepts = medical_entities.get("concepts", [])
            content.anatomical_references = medical_entities.get("anatomy", [])
            content.procedures_mentioned = medical_entities.get("procedures", [])
            content.pathologies_discussed = medical_entities.get("pathologies", [])

            # Generate semantic embedding
            content.semantic_embedding = await self._generate_semantic_embedding(
                content.content
            )

            # Calculate quality scores
            quality_scores = await self._calculate_content_quality_scores(
                content, specialty
            )

            content.quality_score = quality_scores["quality"]
            content.clinical_relevance = quality_scores["clinical_relevance"]
            content.educational_value = quality_scores["educational_value"]

            processed_content.append(content)

        return processed_content

    async def _create_semantic_index(self, content_blocks: List[ExtractedContent]) -> Dict[str, Any]:
        """Create semantic search index for rapid content retrieval"""

        semantic_index = {
            "embeddings": [],
            "content_ids": [],
            "metadata": [],
            "medical_concept_index": {},
            "anatomical_index": {},
            "procedure_index": {},
            "pathology_index": {}
        }

        for content in content_blocks:
            if content.semantic_embedding:
                semantic_index["embeddings"].append(content.semantic_embedding)
                semantic_index["content_ids"].append(content.content_id)
                semantic_index["metadata"].append({
                    "content_type": content.content_type.value,
                    "title": content.title,
                    "page_numbers": content.page_numbers,
                    "quality_score": content.quality_score
                })

                # Index by medical concepts
                for concept in content.medical_concepts:
                    if concept not in semantic_index["medical_concept_index"]:
                        semantic_index["medical_concept_index"][concept] = []
                    semantic_index["medical_concept_index"][concept].append(content.content_id)

                # Index by anatomical references
                for anatomy in content.anatomical_references:
                    if anatomy not in semantic_index["anatomical_index"]:
                        semantic_index["anatomical_index"][anatomy] = []
                    semantic_index["anatomical_index"][anatomy].append(content.content_id)

        return semantic_index

    async def semantic_search_within_textbook(self, textbook_id: str,
                                            query: str,
                                            search_options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform semantic search within a processed textbook"""

        if search_options is None:
            search_options = {
                "max_results": 10,
                "content_types": ["chapter", "section", "figure", "table"],
                "min_quality_score": 0.5,
                "include_embeddings": False
            }

        # Load processed textbook
        textbook = await self._load_processed_textbook(textbook_id)

        if not textbook:
            return []

        # Generate query embedding
        query_embedding = await self._generate_semantic_embedding(query)

        # Calculate similarity scores
        similarities = []
        for i, content_embedding in enumerate(textbook.semantic_index["embeddings"]):
            similarity = await self._calculate_cosine_similarity(
                query_embedding, content_embedding
            )

            content_id = textbook.semantic_index["content_ids"][i]
            metadata = textbook.semantic_index["metadata"][i]

            if metadata["quality_score"] >= search_options["min_quality_score"]:
                similarities.append({
                    "content_id": content_id,
                    "similarity_score": similarity,
                    "metadata": metadata
                })

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Get full content for top results
        results = []
        for sim in similarities[:search_options["max_results"]]:
            content = await self._get_content_by_id(textbook, sim["content_id"])
            if content:
                result = {
                    "content": content,
                    "similarity_score": sim["similarity_score"],
                    "relevance_explanation": await self._explain_relevance(query, content)
                }
                results.append(result)

        return results

    async def extract_chapter_by_topic(self, textbook_id: str,
                                     topic: str,
                                     extraction_depth: str = "comprehensive") -> Dict[str, Any]:
        """Extract specific chapter or section content by medical topic"""

        # Search for relevant content
        search_results = await self.semantic_search_within_textbook(
            textbook_id, topic, {"max_results": 50}
        )

        if not search_results:
            return {"error": "No content found for the specified topic"}

        # Group results by chapter/section
        grouped_content = await self._group_content_by_hierarchy(search_results)

        # Extract and synthesize relevant content
        extracted_content = await self._synthesize_topic_content(
            grouped_content, topic, extraction_depth
        )

        return {
            "topic": topic,
            "extracted_content": extracted_content,
            "source_textbook": textbook_id,
            "content_blocks_used": len(search_results),
            "extraction_timestamp": datetime.now()
        }

    async def _load_medical_ner_model(self):
        """Load medical named entity recognition model"""
        # This would load a specialized medical NER model
        # For now, using spaCy's general model with medical extensions
        return self.nlp

    async def _extract_medical_entities(self, text: str, specialty: TextbookType) -> Dict[str, List[str]]:
        """Extract medical entities from text"""

        doc = self.nlp(text)

        entities = {
            "concepts": [],
            "anatomy": [],
            "procedures": [],
            "pathologies": [],
            "medications": [],
            "symptoms": []
        }

        # Extract named entities
        for ent in doc.ents:
            entity_text = ent.text.lower()

            # Classify entity based on medical context
            if ent.label_ in ["PERSON", "ORG"]:
                continue  # Skip person/organization names

            # Use specialty-specific classification
            entity_category = await self._classify_medical_entity(
                entity_text, ent.label_, specialty
            )

            if entity_category and entity_category in entities:
                entities[entity_category].append(ent.text)

        # Additional medical concept extraction using patterns
        medical_patterns = await self._get_medical_patterns(specialty)
        for pattern_type, patterns in medical_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities[pattern_type].extend(matches)

        # Remove duplicates and clean
        for category in entities:
            entities[category] = list(set([
                entity.strip() for entity in entities[category]
                if len(entity.strip()) > 2
            ]))

        return entities

    async def _save_processed_textbook(self, textbook: ProcessedTextbook):
        """Save processed textbook to database and file system"""

        # Save to database (simplified - would use actual DB)
        textbook_data = {
            "textbook_id": textbook.textbook_id,
            "metadata": textbook.metadata.__dict__,
            "processing_timestamp": textbook.processing_timestamp,
            "total_content_blocks": textbook.total_content_blocks,
            "chapters_count": len(textbook.chapters)
        }

        # Save detailed content to file system for rapid access
        import pickle

        file_path = f"/app/data/textbooks/{textbook.textbook_id}.pkl"
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(textbook, f)

        self.logger.info(f"Saved processed textbook: {textbook.textbook_id}")

# Global PDF textbook processor instance
pdf_textbook_processor = PDFTextbookProcessor()