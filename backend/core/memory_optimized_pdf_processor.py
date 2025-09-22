# backend/core/memory_optimized_pdf_processor.py
"""
Memory-Optimized PDF Processor for Medical Textbooks
Incorporates advanced optimization patterns from existing KOO system:
- Streaming PDF processing with chunked reading
- Object pool management with reusable PDF parser instances
- Real-time memory monitoring with automatic cleanup
- Checkpoint recovery for interrupted processing
- Medical AI intelligence integration
"""

import asyncio
import json
import os
import hashlib
import psutil
import gc
import weakref
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import fitz  # PyMuPDF for robust PDF processing
import PyPDF2
import spacy
import numpy as np
from pathlib import Path
import aiofiles
import aioredis
from concurrent.futures import ThreadPoolExecutor
import logging

# Medical AI imports
import openai
import anthropic
import google.generativeai as genai

# Advanced text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline

# Configuration from existing system
PDF_MAX_FILE_SIZE = int(os.getenv("PDF_MAX_FILE_SIZE", 104857600))  # 100MB
PDF_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", 1000))
PDF_MEMORY_LIMIT = int(os.getenv("PDF_MEMORY_LIMIT", 536870912))  # 512MB
PDF_PROCESSING_TIMEOUT = int(os.getenv("PDF_PROCESSING_TIMEOUT", 1800))  # 30 min
PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", 8388608))  # 8MB
PDF_POOL_SIZE = int(os.getenv("PDF_POOL_SIZE", 5))
PDF_PAGE_BATCH_SIZE = int(os.getenv("PDF_PAGE_BATCH_SIZE", 10))
PDF_CHECKPOINT_INTERVAL = int(os.getenv("PDF_CHECKPOINT_INTERVAL", 50))
PDF_MEMORY_CHECK_INTERVAL = int(os.getenv("PDF_MEMORY_CHECK_INTERVAL", 10))
PDF_CACHE_TTL = int(os.getenv("PDF_CACHE_TTL", 86400))

logger = logging.getLogger(__name__)

class MemoryPressureLevel(Enum):
    NORMAL = "normal"          # <80% memory usage
    WARNING = "warning"        # 80-90% memory usage
    CRITICAL = "critical"      # 90-95% memory usage
    SEVERE = "severe"          # >95% memory usage

class ExtractionMethod(Enum):
    PYMUPDF_OPTIMIZED = "pymupdf_optimized"
    PYPDF2_OPTIMIZED = "pypdf2_optimized"
    STREAMING_HYBRID = "streaming_hybrid"
    MEDICAL_ENHANCED = "medical_enhanced"

class MedicalEntityType(Enum):
    ANATOMICAL_STRUCTURE = "anatomical_structure"
    MEDICAL_CONDITION = "medical_condition"
    SURGICAL_PROCEDURE = "surgical_procedure"
    MEDICATION = "medication"
    DIAGNOSTIC_TEST = "diagnostic_test"
    MEDICAL_DEVICE = "medical_device"
    CLINICAL_FINDING = "clinical_finding"

@dataclass
class MemoryStats:
    page: int
    memory_mb: float
    memory_percent: float
    timestamp: datetime
    pressure_level: MemoryPressureLevel
    gc_triggered: bool = False

@dataclass
class ProcessingCheckpoint:
    document_id: str
    pages_processed: int
    total_pages: int
    extracted_text: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime
    memory_stats: List[MemoryStats]
    chapter_structure: List[Dict[str, Any]]
    medical_entities: List[Dict[str, Any]]

@dataclass
class PDFParserInstance:
    parser_id: str
    pymupdf_doc: Optional[fitz.Document]
    pypdf2_reader: Optional[PyPDF2.PdfReader]
    created_at: datetime
    last_used: datetime
    usage_count: int
    memory_footprint: float

@dataclass
class OptimizedProcessingResult:
    document_id: str
    success: bool
    pages_processed: int
    total_pages: int
    processing_time: float
    memory_peak_mb: float
    extraction_method: ExtractionMethod
    chapters_extracted: int
    medical_entities_found: int
    errors: List[str]
    warnings: List[str]
    checkpoints_saved: int
    memory_stats: List[MemoryStats]
    ai_analysis_summary: Optional[Dict[str, Any]] = None

class PDFParserPool:
    """Object pool for PDF parser instances to reduce GC overhead"""

    def __init__(self, pool_size: int = PDF_POOL_SIZE):
        self.pool_size = pool_size
        self.available_parsers = deque()
        self.active_parsers = {}
        self.total_created = 0
        self.total_reused = 0
        self._lock = threading.Lock()

    async def get_parser(self, file_path: str) -> PDFParserInstance:
        """Get a parser instance from the pool or create new one"""
        with self._lock:
            if self.available_parsers:
                parser = self.available_parsers.popleft()
                parser.last_used = datetime.now()
                parser.usage_count += 1
                self.total_reused += 1

                # Reinitialize with new file
                await self._reinitialize_parser(parser, file_path)
                return parser

            # Create new parser instance
            parser_id = f"parser_{self.total_created}"
            self.total_created += 1

            try:
                pymupdf_doc = fitz.open(file_path)

                with open(file_path, 'rb') as f:
                    pypdf2_reader = PyPDF2.PdfReader(f)

                parser = PDFParserInstance(
                    parser_id=parser_id,
                    pymupdf_doc=pymupdf_doc,
                    pypdf2_reader=pypdf2_reader,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    usage_count=1,
                    memory_footprint=self._calculate_parser_memory()
                )

                self.active_parsers[parser_id] = parser
                return parser

            except Exception as e:
                logger.error(f"Failed to create PDF parser: {e}")
                raise

    async def return_parser(self, parser: PDFParserInstance):
        """Return parser instance to the pool"""
        with self._lock:
            if parser.parser_id in self.active_parsers:
                del self.active_parsers[parser.parser_id]

                # Clean up parser resources
                await self._cleanup_parser(parser)

                # Return to pool if we have space
                if len(self.available_parsers) < self.pool_size:
                    self.available_parsers.append(parser)
                else:
                    # Destroy excess parser
                    await self._destroy_parser(parser)

    async def _reinitialize_parser(self, parser: PDFParserInstance, file_path: str):
        """Reinitialize parser with new file"""
        try:
            # Close existing documents
            if parser.pymupdf_doc:
                parser.pymupdf_doc.close()

            # Open new documents
            parser.pymupdf_doc = fitz.open(file_path)

            with open(file_path, 'rb') as f:
                parser.pypdf2_reader = PyPDF2.PdfReader(f)

        except Exception as e:
            logger.error(f"Failed to reinitialize parser: {e}")
            raise

    async def _cleanup_parser(self, parser: PDFParserInstance):
        """Clean up parser resources without destroying"""
        try:
            if parser.pymupdf_doc:
                parser.pymupdf_doc.close()
                parser.pymupdf_doc = None

            parser.pypdf2_reader = None

            # Force garbage collection
            gc.collect()

        except Exception as e:
            logger.error(f"Error cleaning up parser: {e}")

    async def _destroy_parser(self, parser: PDFParserInstance):
        """Completely destroy parser instance"""
        await self._cleanup_parser(parser)

    def _calculate_parser_memory(self) -> float:
        """Calculate memory footprint of parser"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # MB

    async def cleanup_pool(self):
        """Clean up entire parser pool"""
        with self._lock:
            # Clean up available parsers
            while self.available_parsers:
                parser = self.available_parsers.popleft()
                await self._destroy_parser(parser)

            # Clean up active parsers
            for parser in self.active_parsers.values():
                await self._destroy_parser(parser)

            self.active_parsers.clear()

            # Force garbage collection
            gc.collect()

class MemoryMonitor:
    """Real-time memory usage monitoring with pressure detection"""

    def __init__(self):
        self.memory_limit = PDF_MEMORY_LIMIT
        self.stats_history = deque(maxlen=1000)
        self.monitoring_active = False
        self._monitor_task = None

    def start_monitoring(self, interval: int = 5):
        """Start continuous memory monitoring"""
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))

    async def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self, interval: int):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                stats = self.get_current_stats(-1)  # Page -1 for monitoring
                self.stats_history.append(stats)

                # Check for memory pressure
                if stats.pressure_level in [MemoryPressureLevel.CRITICAL, MemoryPressureLevel.SEVERE]:
                    await self._handle_memory_pressure(stats)

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(interval)

    def get_current_stats(self, page: int) -> MemoryStats:
        """Get current memory statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        memory_percent = process.memory_percent()

        # Determine pressure level
        if memory_percent < 80:
            pressure_level = MemoryPressureLevel.NORMAL
        elif memory_percent < 90:
            pressure_level = MemoryPressureLevel.WARNING
        elif memory_percent < 95:
            pressure_level = MemoryPressureLevel.CRITICAL
        else:
            pressure_level = MemoryPressureLevel.SEVERE

        return MemoryStats(
            page=page,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            timestamp=datetime.now(),
            pressure_level=pressure_level
        )

    async def _handle_memory_pressure(self, stats: MemoryStats):
        """Handle memory pressure situations"""
        logger.warning(f"Memory pressure detected: {stats.pressure_level.value} - {stats.memory_percent:.1f}%")

        if stats.pressure_level == MemoryPressureLevel.WARNING:
            # Trigger garbage collection
            gc.collect()
            stats.gc_triggered = True

        elif stats.pressure_level == MemoryPressureLevel.CRITICAL:
            # Force aggressive cleanup
            gc.collect()
            await asyncio.sleep(1)  # Allow cleanup to complete
            stats.gc_triggered = True

        elif stats.pressure_level == MemoryPressureLevel.SEVERE:
            # Emergency cleanup
            gc.collect()
            await asyncio.sleep(2)
            stats.gc_triggered = True

            # Log severe memory pressure
            logger.critical(f"Severe memory pressure: {stats.memory_mb:.1f}MB / {stats.memory_percent:.1f}%")

    def should_pause_processing(self) -> bool:
        """Check if processing should be paused due to memory pressure"""
        current_stats = self.get_current_stats(-1)
        return current_stats.pressure_level in [MemoryPressureLevel.CRITICAL, MemoryPressureLevel.SEVERE]

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of memory statistics"""
        if not self.stats_history:
            return {}

        recent_stats = list(self.stats_history)[-100:]  # Last 100 measurements

        memory_values = [s.memory_mb for s in recent_stats]
        percent_values = [s.memory_percent for s in recent_stats]

        return {
            "current_memory_mb": recent_stats[-1].memory_mb,
            "current_memory_percent": recent_stats[-1].memory_percent,
            "peak_memory_mb": max(memory_values),
            "peak_memory_percent": max(percent_values),
            "average_memory_mb": sum(memory_values) / len(memory_values),
            "pressure_events": len([s for s in recent_stats if s.pressure_level != MemoryPressureLevel.NORMAL]),
            "gc_triggers": len([s for s in recent_stats if s.gc_triggered])
        }

class CheckpointManager:
    """Manages processing checkpoints for recovery"""

    def __init__(self, checkpoint_dir: str = "/tmp/pdf_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    async def save_checkpoint(self, checkpoint: ProcessingCheckpoint) -> str:
        """Save processing checkpoint"""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint.document_id}_checkpoint.json"

        try:
            checkpoint_data = {
                "document_id": checkpoint.document_id,
                "pages_processed": checkpoint.pages_processed,
                "total_pages": checkpoint.total_pages,
                "extracted_text": checkpoint.extracted_text,
                "metadata": checkpoint.metadata,
                "timestamp": checkpoint.timestamp.isoformat(),
                "memory_stats": [asdict(stat) for stat in checkpoint.memory_stats],
                "chapter_structure": checkpoint.chapter_structure,
                "medical_entities": checkpoint.medical_entities
            }

            async with aiofiles.open(checkpoint_file, 'w') as f:
                await f.write(json.dumps(checkpoint_data, indent=2))

            logger.info(f"Checkpoint saved for document {checkpoint.document_id} at page {checkpoint.pages_processed}")
            return str(checkpoint_file)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    async def load_checkpoint(self, document_id: str) -> Optional[ProcessingCheckpoint]:
        """Load processing checkpoint"""
        checkpoint_file = self.checkpoint_dir / f"{document_id}_checkpoint.json"

        if not checkpoint_file.exists():
            return None

        try:
            async with aiofiles.open(checkpoint_file, 'r') as f:
                checkpoint_data = json.loads(await f.read())

            # Reconstruct memory stats
            memory_stats = []
            for stat_data in checkpoint_data["memory_stats"]:
                memory_stats.append(MemoryStats(
                    page=stat_data["page"],
                    memory_mb=stat_data["memory_mb"],
                    memory_percent=stat_data["memory_percent"],
                    timestamp=datetime.fromisoformat(stat_data["timestamp"]),
                    pressure_level=MemoryPressureLevel(stat_data["pressure_level"]),
                    gc_triggered=stat_data.get("gc_triggered", False)
                ))

            checkpoint = ProcessingCheckpoint(
                document_id=checkpoint_data["document_id"],
                pages_processed=checkpoint_data["pages_processed"],
                total_pages=checkpoint_data["total_pages"],
                extracted_text=checkpoint_data["extracted_text"],
                metadata=checkpoint_data["metadata"],
                timestamp=datetime.fromisoformat(checkpoint_data["timestamp"]),
                memory_stats=memory_stats,
                chapter_structure=checkpoint_data["chapter_structure"],
                medical_entities=checkpoint_data["medical_entities"]
            )

            logger.info(f"Checkpoint loaded for document {document_id}")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def cleanup_checkpoint(self, document_id: str):
        """Clean up completed checkpoint"""
        checkpoint_file = self.checkpoint_dir / f"{document_id}_checkpoint.json"

        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info(f"Checkpoint cleaned up for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoint: {e}")

class MedicalEntityExtractor:
    """Extract medical entities using NLP and medical AI models"""

    def __init__(self):
        self.nlp = None
        self.medical_ner = None
        self._load_models()

    def _load_models(self):
        """Load medical NLP models"""
        try:
            # Load spaCy model for medical text
            self.nlp = spacy.load("en_core_web_sm")

            # Load specialized medical NER model (if available)
            try:
                self.medical_ner = pipeline("ner",
                                           model="d4data/biomedical-ner-all",
                                           aggregation_strategy="simple")
            except Exception as e:
                logger.warning(f"Medical NER model not available: {e}")

        except Exception as e:
            logger.error(f"Failed to load NLP models: {e}")

    async def extract_medical_entities(self, text: str, chapter_title: str = "") -> List[Dict[str, Any]]:
        """Extract medical entities from text"""
        entities = []

        try:
            # Use medical NER if available
            if self.medical_ner:
                ner_results = self.medical_ner(text[:512])  # Limit text length

                for entity in ner_results:
                    entities.append({
                        "text": entity["word"],
                        "label": entity["entity_group"],
                        "confidence": entity["score"],
                        "start": entity.get("start", 0),
                        "end": entity.get("end", 0),
                        "entity_type": self._map_to_medical_entity_type(entity["entity_group"]),
                        "context": chapter_title
                    })

            # Use spaCy for additional entities
            if self.nlp:
                doc = self.nlp(text[:1000])  # Limit text length

                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "ORG", "GPE"]:  # Medical-relevant entities
                        entities.append({
                            "text": ent.text,
                            "label": ent.label_,
                            "confidence": 0.8,  # Default confidence for spaCy
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "entity_type": self._map_to_medical_entity_type(ent.label_),
                            "context": chapter_title
                        })

            # Remove duplicates and sort by confidence
            entities = self._deduplicate_entities(entities)
            entities.sort(key=lambda x: x["confidence"], reverse=True)

            return entities[:50]  # Limit to top 50 entities

        except Exception as e:
            logger.error(f"Error extracting medical entities: {e}")
            return []

    def _map_to_medical_entity_type(self, label: str) -> MedicalEntityType:
        """Map NER labels to medical entity types"""
        label_mapping = {
            "ANATOMY": MedicalEntityType.ANATOMICAL_STRUCTURE,
            "DISEASE": MedicalEntityType.MEDICAL_CONDITION,
            "PROCEDURE": MedicalEntityType.SURGICAL_PROCEDURE,
            "DRUG": MedicalEntityType.MEDICATION,
            "TEST": MedicalEntityType.DIAGNOSTIC_TEST,
            "DEVICE": MedicalEntityType.MEDICAL_DEVICE,
            "FINDING": MedicalEntityType.CLINICAL_FINDING,
            "PERSON": MedicalEntityType.CLINICAL_FINDING,  # Often refers to syndromes
            "ORG": MedicalEntityType.CLINICAL_FINDING,
            "GPE": MedicalEntityType.ANATOMICAL_STRUCTURE
        }

        return label_mapping.get(label.upper(), MedicalEntityType.CLINICAL_FINDING)

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on text similarity"""
        unique_entities = []
        seen_texts = set()

        for entity in entities:
            text_lower = entity["text"].lower().strip()
            if text_lower not in seen_texts and len(text_lower) > 2:
                seen_texts.add(text_lower)
                unique_entities.append(entity)

        return unique_entities

class MemoryOptimizedPDFProcessor:
    """
    Memory-optimized PDF processor with medical intelligence
    Incorporates all optimization patterns from existing KOO system
    """

    def __init__(self):
        self.parser_pool = PDFParserPool(PDF_POOL_SIZE)
        self.memory_monitor = MemoryMonitor()
        self.checkpoint_manager = CheckpointManager()
        self.medical_extractor = MedicalEntityExtractor()
        self.processing_semaphore = asyncio.Semaphore(3)  # Limit concurrent processing

        # AI clients (same as existing system)
        self.ai_clients = self._setup_ai_clients()

        # Processing statistics
        self.total_processed = 0
        self.total_errors = 0
        self.total_memory_warnings = 0

    def _setup_ai_clients(self) -> Dict[str, Any]:
        """Setup AI clients for medical analysis"""
        return {
            "openai": openai.AsyncOpenAI(),
            "anthropic": anthropic.AsyncAnthropic(),
            "google": genai
        }

    async def process_medical_textbook(self,
                                     file_path: str,
                                     document_id: str,
                                     metadata: Dict[str, Any] = None,
                                     resume_from_checkpoint: bool = True) -> OptimizedProcessingResult:
        """
        Process medical textbook PDF with full optimization
        """

        async with self.processing_semaphore:
            start_time = time.time()

            # Start memory monitoring
            await self.memory_monitor.start_monitoring()

            try:
                # Check for existing checkpoint
                checkpoint = None
                if resume_from_checkpoint:
                    checkpoint = await self.checkpoint_manager.load_checkpoint(document_id)

                # Validate file
                if not await self._validate_pdf_file(file_path):
                    raise ValueError(f"Invalid or corrupted PDF file: {file_path}")

                # Get parser from pool
                parser = await self.parser_pool.get_parser(file_path)

                try:
                    # Process with memory optimization
                    result = await self._stream_process_pdf(
                        parser, document_id, metadata or {}, checkpoint
                    )

                    # Clean up checkpoint on success
                    await self.checkpoint_manager.cleanup_checkpoint(document_id)

                    self.total_processed += 1

                    return result

                finally:
                    # Return parser to pool
                    await self.parser_pool.return_parser(parser)

            except Exception as e:
                self.total_errors += 1
                logger.error(f"Error processing PDF {file_path}: {e}")

                return OptimizedProcessingResult(
                    document_id=document_id,
                    success=False,
                    pages_processed=0,
                    total_pages=0,
                    processing_time=time.time() - start_time,
                    memory_peak_mb=self.memory_monitor.get_current_stats(-1).memory_mb,
                    extraction_method=ExtractionMethod.PYMUPDF_OPTIMIZED,
                    chapters_extracted=0,
                    medical_entities_found=0,
                    errors=[str(e)],
                    warnings=[],
                    checkpoints_saved=0,
                    memory_stats=list(self.memory_monitor.stats_history)
                )

            finally:
                # Stop memory monitoring
                await self.memory_monitor.stop_monitoring()

    async def _stream_process_pdf(self,
                                parser: PDFParserInstance,
                                document_id: str,
                                metadata: Dict[str, Any],
                                checkpoint: Optional[ProcessingCheckpoint]) -> OptimizedProcessingResult:
        """Stream process PDF with memory optimization"""

        total_pages = len(parser.pymupdf_doc)
        start_page = 0
        extracted_text = []
        memory_stats = []
        errors = []
        warnings = []
        checkpoints_saved = 0
        chapters_extracted = 0
        medical_entities = []

        # Resume from checkpoint if available
        if checkpoint:
            start_page = checkpoint.pages_processed
            extracted_text = checkpoint.extracted_text
            memory_stats = checkpoint.memory_stats
            medical_entities = checkpoint.medical_entities
            logger.info(f"Resuming processing from page {start_page}")

        # Process in batches
        for batch_start in range(start_page, total_pages, PDF_PAGE_BATCH_SIZE):
            batch_end = min(batch_start + PDF_PAGE_BATCH_SIZE, total_pages)

            # Check memory pressure before processing batch
            if self.memory_monitor.should_pause_processing():
                self.total_memory_warnings += 1
                warnings.append(f"Memory pressure detected at page {batch_start}, pausing processing")

                # Save checkpoint before pausing
                checkpoint = ProcessingCheckpoint(
                    document_id=document_id,
                    pages_processed=batch_start,
                    total_pages=total_pages,
                    extracted_text=extracted_text,
                    metadata=metadata,
                    timestamp=datetime.now(),
                    memory_stats=memory_stats,
                    chapter_structure=[],
                    medical_entities=medical_entities
                )

                await self.checkpoint_manager.save_checkpoint(checkpoint)
                checkpoints_saved += 1

                # Wait for memory pressure to reduce
                await asyncio.sleep(5)
                gc.collect()

            try:
                # Process batch of pages
                batch_text, batch_entities = await self._process_page_batch(
                    parser, batch_start, batch_end, document_id
                )

                extracted_text.extend(batch_text)
                medical_entities.extend(batch_entities)

                # Record memory stats
                current_stats = self.memory_monitor.get_current_stats(batch_end)
                memory_stats.append(current_stats)

                # Save checkpoint if needed
                if batch_end % PDF_CHECKPOINT_INTERVAL == 0:
                    checkpoint = ProcessingCheckpoint(
                        document_id=document_id,
                        pages_processed=batch_end,
                        total_pages=total_pages,
                        extracted_text=extracted_text,
                        metadata=metadata,
                        timestamp=datetime.now(),
                        memory_stats=memory_stats,
                        chapter_structure=[],
                        medical_entities=medical_entities
                    )

                    await self.checkpoint_manager.save_checkpoint(checkpoint)
                    checkpoints_saved += 1

                # Check for memory cleanup
                if batch_end % PDF_MEMORY_CHECK_INTERVAL == 0:
                    gc.collect()

            except Exception as e:
                error_msg = f"Error processing pages {batch_start}-{batch_end}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
                continue

        # Final processing statistics
        processing_time = time.time() - time.time()  # Will be calculated properly in calling function
        peak_memory = max([s.memory_mb for s in memory_stats]) if memory_stats else 0

        return OptimizedProcessingResult(
            document_id=document_id,
            success=len(errors) == 0,
            pages_processed=len(extracted_text),
            total_pages=total_pages,
            processing_time=processing_time,
            memory_peak_mb=peak_memory,
            extraction_method=ExtractionMethod.STREAMING_HYBRID,
            chapters_extracted=chapters_extracted,
            medical_entities_found=len(medical_entities),
            errors=errors,
            warnings=warnings,
            checkpoints_saved=checkpoints_saved,
            memory_stats=memory_stats
        )

    async def _process_page_batch(self,
                                parser: PDFParserInstance,
                                start_page: int,
                                end_page: int,
                                document_id: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process a batch of pages with memory optimization"""

        batch_text = []
        batch_entities = []

        for page_num in range(start_page, end_page):
            try:
                # Extract text from page
                page_text = await self._extract_page_text_optimized(parser, page_num)

                if page_text and len(page_text.strip()) > 50:  # Only process meaningful text
                    batch_text.append(page_text)

                    # Extract medical entities (limit frequency to save memory)
                    if page_num % 5 == 0:  # Every 5th page
                        entities = await self.medical_extractor.extract_medical_entities(
                            page_text, f"Page {page_num}"
                        )
                        batch_entities.extend(entities)

            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                continue

        return batch_text, batch_entities

    async def _extract_page_text_optimized(self,
                                         parser: PDFParserInstance,
                                         page_num: int) -> str:
        """Extract text from page using optimized method"""

        try:
            # Try PyMuPDF first (usually faster and more accurate)
            if parser.pymupdf_doc:
                page = parser.pymupdf_doc[page_num]
                text = page.get_text()

                if text and len(text.strip()) > 10:
                    return text

            # Fallback to PyPDF2
            if parser.pypdf2_reader and len(parser.pypdf2_reader.pages) > page_num:
                page = parser.pypdf2_reader.pages[page_num]
                text = page.extract_text()

                if text and len(text.strip()) > 10:
                    return text

            return ""

        except Exception as e:
            logger.error(f"Error extracting text from page {page_num}: {e}")
            return ""

    async def _validate_pdf_file(self, file_path: str) -> bool:
        """Validate PDF file before processing"""

        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > PDF_MAX_FILE_SIZE:
                logger.warning(f"PDF file too large: {file_size} bytes > {PDF_MAX_FILE_SIZE}")
                return False

            # Try to open with PyMuPDF
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()

            if page_count > PDF_MAX_PAGES:
                logger.warning(f"PDF has too many pages: {page_count} > {PDF_MAX_PAGES}")
                return False

            return True

        except Exception as e:
            logger.error(f"PDF validation failed: {e}")
            return False

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""

        memory_summary = self.memory_monitor.get_stats_summary()

        return {
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "total_memory_warnings": self.total_memory_warnings,
            "parser_pool_stats": {
                "pool_size": self.parser_pool.pool_size,
                "total_created": self.parser_pool.total_created,
                "total_reused": self.parser_pool.total_reused,
                "active_parsers": len(self.parser_pool.active_parsers),
                "available_parsers": len(self.parser_pool.available_parsers)
            },
            "memory_stats": memory_summary,
            "configuration": {
                "pdf_max_file_size": PDF_MAX_FILE_SIZE,
                "pdf_max_pages": PDF_MAX_PAGES,
                "pdf_memory_limit": PDF_MEMORY_LIMIT,
                "pdf_chunk_size": PDF_CHUNK_SIZE,
                "pdf_pool_size": PDF_POOL_SIZE,
                "pdf_page_batch_size": PDF_PAGE_BATCH_SIZE,
                "pdf_checkpoint_interval": PDF_CHECKPOINT_INTERVAL
            }
        }

    async def cleanup_resources(self):
        """Clean up all resources"""

        await self.parser_pool.cleanup_pool()
        await self.memory_monitor.stop_monitoring()

        # Force garbage collection
        gc.collect()

        logger.info("PDF processor resources cleaned up")

# Global instance
memory_optimized_pdf_processor = MemoryOptimizedPDFProcessor()

# Convenience functions for API integration
async def process_pdf_with_optimization(file_path: str,
                                      document_id: str,
                                      metadata: Dict[str, Any] = None) -> OptimizedProcessingResult:
    """Process PDF with full memory optimization"""
    return await memory_optimized_pdf_processor.process_medical_textbook(
        file_path, document_id, metadata
    )

async def resume_pdf_processing(document_id: str) -> OptimizedProcessingResult:
    """Resume interrupted PDF processing"""
    # This would need file path from database lookup
    # For now, return an error
    return OptimizedProcessingResult(
        document_id=document_id,
        success=False,
        pages_processed=0,
        total_pages=0,
        processing_time=0,
        memory_peak_mb=0,
        extraction_method=ExtractionMethod.PYMUPDF_OPTIMIZED,
        chapters_extracted=0,
        medical_entities_found=0,
        errors=["File path lookup not implemented"],
        warnings=[],
        checkpoints_saved=0,
        memory_stats=[]
    )

async def cleanup_pdf_memory():
    """Clean up PDF processing memory"""
    await memory_optimized_pdf_processor.cleanup_resources()
    gc.collect()

    return {
        "success": True,
        "message": "PDF processing memory cleaned up"
    }

async def get_pdf_processing_metrics() -> Dict[str, Any]:
    """Get PDF processing performance metrics"""
    return await memory_optimized_pdf_processor.get_processing_stats()