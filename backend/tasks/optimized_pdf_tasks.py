# backend/tasks/optimized_pdf_tasks.py
"""
Optimized PDF processing tasks for Celery integration
Based on existing KOO Platform task architecture with memory optimization
"""

import asyncio
import os
import gc
import psutil
import tempfile
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from celery import shared_task
from celery.utils.log import get_task_logger

from ..core.memory_optimized_pdf_processor import (
    memory_optimized_pdf_processor,
    process_pdf_with_optimization,
    resume_pdf_processing,
    cleanup_pdf_memory,
    get_pdf_processing_metrics,
    OptimizedProcessingResult
)

from ..core.ultra_enhanced_pdf_processor import ultra_enhanced_pdf_processor
from ..core.advanced_research_engine import advanced_research_engine
from ..core.advanced_synthesis_engine import advanced_synthesis_engine

logger = get_task_logger(__name__)

@shared_task(bind=True, name="koo.tasks.pdf_processing.process_document")
def process_document(self, file_path: str, document_id: str, metadata: Dict[str, Any] = None):
    """
    Process a PDF document with memory optimization
    Compatible with existing task management system
    """

    try:
        logger.info(f"Starting optimized processing for document {document_id}")

        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Starting PDF processing', 'progress': 0}
        )

        # Run async processing in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                process_pdf_with_optimization(file_path, document_id, metadata or {})
            )

            # Convert result to serializable format
            result_dict = {
                "document_id": result.document_id,
                "success": result.success,
                "pages_processed": result.pages_processed,
                "total_pages": result.total_pages,
                "processing_time": result.processing_time,
                "memory_peak_mb": result.memory_peak_mb,
                "extraction_method": result.extraction_method.value,
                "chapters_extracted": result.chapters_extracted,
                "medical_entities_found": result.medical_entities_found,
                "errors": result.errors,
                "warnings": result.warnings,
                "checkpoints_saved": result.checkpoints_saved,
                "timestamp": datetime.now().isoformat()
            }

            if result.success:
                logger.info(f"Successfully processed document {document_id}: {result.pages_processed} pages")
                return result_dict
            else:
                logger.error(f"Failed to process document {document_id}: {result.errors}")
                raise Exception(f"Processing failed: {result.errors}")

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=3)

@shared_task(bind=True, name="koo.tasks.pdf_processing.extract_text")
def extract_text(self, file_path: str):
    """
    Extract text only from PDF (lightweight operation)
    """

    try:
        logger.info(f"Starting text extraction for {file_path}")

        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Extracting text', 'progress': 0}
        )

        # Generate temporary document ID
        document_id = f"text_extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Use minimal metadata for text-only extraction
            minimal_metadata = {
                "extraction_mode": "text_only",
                "ai_analysis_enabled": False,
                "medical_entity_extraction": False
            }

            result = loop.run_until_complete(
                process_pdf_with_optimization(file_path, document_id, minimal_metadata)
            )

            # Return just the extracted text
            return {
                "success": result.success,
                "pages_processed": result.pages_processed,
                "text_extracted": True,
                "processing_time": result.processing_time,
                "errors": result.errors
            }

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        raise self.retry(exc=e, countdown=30, max_retries=2)

@shared_task(bind=True, name="koo.tasks.pdf_processing.batch_processing")
def batch_processing(self, file_paths: List[str], batch_size: int = 5):
    """
    Process multiple PDF files in batches with memory optimization
    """

    try:
        logger.info(f"Starting batch processing for {len(file_paths)} files")

        results = []
        total_files = len(file_paths)
        processed_files = 0

        # Process in batches
        for i in range(0, total_files, batch_size):
            batch = file_paths[i:i + batch_size]

            # Update progress
            progress = int((processed_files / total_files) * 100)
            self.update_state(
                state='PROGRESS',
                meta={
                    'status': f'Processing batch {i//batch_size + 1}',
                    'progress': progress,
                    'processed_files': processed_files,
                    'total_files': total_files
                }
            )

            # Process batch
            batch_results = []
            for file_path in batch:
                try:
                    document_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{processed_files}"

                    # Run async processing
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    try:
                        result = loop.run_until_complete(
                            process_pdf_with_optimization(file_path, document_id, {})
                        )

                        batch_results.append({
                            "file_path": file_path,
                            "document_id": document_id,
                            "success": result.success,
                            "pages_processed": result.pages_processed,
                            "errors": result.errors
                        })

                    finally:
                        loop.close()

                    processed_files += 1

                except Exception as e:
                    logger.error(f"Error processing {file_path} in batch: {str(e)}")
                    batch_results.append({
                        "file_path": file_path,
                        "success": False,
                        "error": str(e)
                    })
                    processed_files += 1

            results.extend(batch_results)

            # Memory cleanup between batches
            gc.collect()

            # Check memory pressure
            memory_percent = psutil.Process().memory_percent()
            if memory_percent > 80:
                logger.warning(f"High memory usage: {memory_percent:.1f}% - pausing between batches")
                import time
                time.sleep(5)
                gc.collect()

        # Final results
        successful = len([r for r in results if r.get("success", False)])
        failed = len(results) - successful

        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")

        return {
            "total_files": total_files,
            "successful": successful,
            "failed": failed,
            "results": results,
            "processing_completed": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise self.retry(exc=e, countdown=120, max_retries=2)

@shared_task(bind=True, name="koo.tasks.pdf_processing.resume_processing")
def resume_processing(self, document_id: str):
    """
    Resume interrupted PDF processing from checkpoint
    """

    try:
        logger.info(f"Resuming processing for document {document_id}")

        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Resuming from checkpoint', 'progress': 0}
        )

        # Run async resume
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                resume_pdf_processing(document_id)
            )

            if result.success:
                logger.info(f"Successfully resumed processing for document {document_id}")
                return {
                    "document_id": document_id,
                    "success": True,
                    "pages_processed": result.pages_processed,
                    "resumed_at": datetime.now().isoformat()
                }
            else:
                logger.error(f"Failed to resume processing for document {document_id}: {result.errors}")
                raise Exception(f"Resume failed: {result.errors}")

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error resuming processing for {document_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=2)

@shared_task(bind=True, name="koo.tasks.pdf_processing.cleanup_memory")
def cleanup_memory(self):
    """
    Clean up PDF processing memory and resources
    """

    try:
        logger.info("Starting PDF processing memory cleanup")

        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Cleaning up memory', 'progress': 50}
        )

        # Get memory stats before cleanup
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB

        # Run async cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(cleanup_pdf_memory())
        finally:
            loop.close()

        # Force additional garbage collection
        gc.collect()

        # Get memory stats after cleanup
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_freed = memory_before - memory_after

        logger.info(f"Memory cleanup completed: {memory_freed:.1f}MB freed")

        return {
            "success": result["success"],
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_freed_mb": memory_freed,
            "cleanup_completed": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error during memory cleanup: {str(e)}")
        raise self.retry(exc=e, countdown=30, max_retries=1)

@shared_task(bind=True, name="koo.tasks.pdf_processing.get_processing_metrics")
def get_processing_metrics(self):
    """
    Get comprehensive PDF processing metrics
    """

    try:
        logger.info("Retrieving PDF processing metrics")

        # Run async metrics collection
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            metrics = loop.run_until_complete(get_pdf_processing_metrics())
        finally:
            loop.close()

        # Add system metrics
        process = psutil.Process()
        system_metrics = {
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_cpu_percent": psutil.cpu_percent(),
            "process_memory_mb": process.memory_info().rss / (1024 * 1024),
            "process_cpu_percent": process.cpu_percent(),
            "open_files": len(process.open_files()),
            "threads": process.num_threads()
        }

        metrics["system_metrics"] = system_metrics
        metrics["metrics_collected_at"] = datetime.now().isoformat()

        return metrics

    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise

@shared_task(bind=True, name="koo.tasks.pdf_processing.ultra_process_textbook")
def ultra_process_textbook(self, file_path: str, document_id: str, metadata: Dict[str, Any] = None):
    """
    Ultra-comprehensive processing with full AI analysis
    Uses the ultra-enhanced processor for maximum intelligence
    """

    try:
        logger.info(f"Starting ultra processing for textbook {document_id}")

        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Starting ultra-enhanced processing', 'progress': 0}
        )

        # Run async ultra processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Prepare enhanced metadata
            enhanced_metadata = {
                "processing_mode": "ultra_enhanced",
                "ai_analysis_enabled": True,
                "medical_entity_extraction": True,
                "research_integration": True,
                "synthesis_enabled": True,
                **(metadata or {})
            }

            # Use ultra-enhanced processor
            result = loop.run_until_complete(
                ultra_enhanced_pdf_processor.ultra_process_textbook(
                    file_path, enhanced_metadata, {"enable_all_features": True}
                )
            )

            logger.info(f"Ultra processing completed for {document_id}: {result.total_chapters} chapters")

            return {
                "document_id": document_id,
                "success": True,
                "total_pages": result.total_pages,
                "total_chapters": result.total_chapters,
                "medical_entities": len(result.medical_entities),
                "ai_analysis_summary": result.ai_analysis_summary,
                "processing_completed": datetime.now().isoformat()
            }

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Error in ultra processing for {document_id}: {str(e)}")
        raise self.retry(exc=e, countdown=180, max_retries=2)

@shared_task(bind=True, name="koo.tasks.pdf_processing.health_check")
def health_check(self):
    """
    Health check task for PDF processing system
    """

    try:
        # Check system resources
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage('/').percent

        # Check PDF processor status
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            metrics = loop.run_until_complete(get_pdf_processing_metrics())
        finally:
            loop.close()

        # Determine health status
        issues = []
        if memory_percent > 90:
            issues.append(f"High memory usage: {memory_percent:.1f}%")
        if cpu_percent > 90:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        if disk_usage > 90:
            issues.append(f"High disk usage: {disk_usage:.1f}%")

        error_rate = metrics.get("total_errors", 0) / max(metrics.get("total_processed", 1), 1)
        if error_rate > 0.1:
            issues.append(f"High error rate: {error_rate:.1%}")

        status = "healthy" if not issues else "warning" if len(issues) < 3 else "critical"

        return {
            "status": status,
            "issues": issues,
            "system_metrics": {
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent,
                "disk_usage_percent": disk_usage
            },
            "processing_metrics": {
                "total_processed": metrics.get("total_processed", 0),
                "total_errors": metrics.get("total_errors", 0),
                "error_rate": error_rate
            },
            "health_check_time": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "health_check_time": datetime.now().isoformat()
        }

# Task routing for existing system compatibility
TASK_ROUTES = {
    "process_document": process_document,
    "extract_text": extract_text,
    "batch_processing": batch_processing,
    "resume_processing": resume_processing,
    "cleanup_memory": cleanup_memory,
    "get_processing_metrics": get_processing_metrics,
    "ultra_process_textbook": ultra_process_textbook,
    "health_check": health_check
}

def get_task_by_name(task_name: str):
    """Get task function by name for dynamic routing"""
    return TASK_ROUTES.get(task_name)