# backend/api/v1/endpoints/optimized_pdf_processing.py
"""
API endpoints for memory-optimized PDF processing
Integrates with the advanced PDF processor and task management system
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import aiofiles
import tempfile
import os
from pathlib import Path
import uuid

from ....core.memory_optimized_pdf_processor import (
    memory_optimized_pdf_processor,
    process_pdf_with_optimization,
    resume_pdf_processing,
    cleanup_pdf_memory,
    get_pdf_processing_metrics,
    OptimizedProcessingResult
)

from ....core.task_manager import submit_pdf_processing_task
from ....database.models import ProcessedDocument, User
from ....core.auth import get_current_user

router = APIRouter(prefix="/pdf-processing", tags=["PDF Processing"])

@router.post("/upload-and-process")
async def upload_and_process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    specialty: Optional[str] = None,
    priority_level: Optional[int] = 5,
    processing_mode: Optional[str] = "medical_enhanced",
    enable_ai_analysis: Optional[bool] = True,
    current_user: User = Depends(get_current_user)
):
    """
    Upload and process a medical textbook PDF with memory optimization
    """

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate unique document ID
    document_id = f"doc_{uuid.uuid4().hex}"

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_path = temp_file.name

            # Save uploaded file
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await file.read()
                await f.write(content)

        # Prepare metadata
        metadata = {
            "title": title or file.filename,
            "original_filename": file.filename,
            "file_size": len(content),
            "specialty": specialty or "neurosurgery",
            "priority_level": priority_level,
            "processing_mode": processing_mode,
            "enable_ai_analysis": enable_ai_analysis,
            "uploaded_by": current_user.id,
            "upload_timestamp": "now()"
        }

        # Submit processing task
        task_id = await submit_pdf_processing_task(
            "process_pdf_with_optimization",
            temp_path,
            document_id,
            metadata
        )

        # Clean up temp file in background
        background_tasks.add_task(os.unlink, temp_path)

        return {
            "success": True,
            "document_id": document_id,
            "task_id": task_id,
            "message": "PDF processing started",
            "estimated_completion": "5-15 minutes depending on file size"
        }

    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@router.post("/process-file")
async def process_existing_file(
    file_path: str,
    document_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Process an existing PDF file with memory optimization
    """

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    if not document_id:
        document_id = f"doc_{uuid.uuid4().hex}"

    try:
        # Submit processing task
        task_id = await submit_pdf_processing_task(
            "process_pdf_with_optimization",
            file_path,
            document_id,
            metadata or {}
        )

        return {
            "success": True,
            "document_id": document_id,
            "task_id": task_id,
            "message": "PDF processing started"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@router.post("/batch-process")
async def batch_process_pdfs(
    file_paths: List[str],
    batch_size: Optional[int] = 5,
    current_user: User = Depends(get_current_user)
):
    """
    Process multiple PDF files in batches with memory optimization
    """

    # Validate all files exist
    missing_files = [f for f in file_paths if not os.path.exists(f)]
    if missing_files:
        raise HTTPException(status_code=404, detail=f"Files not found: {missing_files}")

    try:
        # Submit batch processing task
        task_id = await submit_pdf_processing_task(
            "batch_processing",
            file_paths,
            batch_size=batch_size
        )

        return {
            "success": True,
            "task_id": task_id,
            "files_count": len(file_paths),
            "batch_size": batch_size,
            "message": "Batch PDF processing started"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start batch processing: {str(e)}")

@router.post("/resume/{document_id}")
async def resume_processing(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Resume interrupted PDF processing from checkpoint
    """

    try:
        # Submit resume task
        task_id = await submit_pdf_processing_task(
            "resume_pdf_processing",
            document_id
        )

        return {
            "success": True,
            "document_id": document_id,
            "task_id": task_id,
            "message": "PDF processing resumed from checkpoint"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resume processing: {str(e)}")

@router.get("/status/{document_id}")
async def get_processing_status(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get processing status for a document
    """

    try:
        # This would integrate with your task status system
        # For now, return a placeholder response

        return {
            "document_id": document_id,
            "status": "processing",
            "progress": {
                "pages_processed": 0,
                "total_pages": 0,
                "percentage": 0
            },
            "memory_stats": {
                "current_memory_mb": 0,
                "peak_memory_mb": 0
            },
            "errors": [],
            "warnings": []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/metrics")
async def get_processing_metrics(
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive PDF processing metrics
    """

    try:
        metrics = await get_pdf_processing_metrics()

        return {
            "success": True,
            "metrics": metrics,
            "timestamp": "now()"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.post("/cleanup-memory")
async def cleanup_processing_memory(
    current_user: User = Depends(get_current_user)
):
    """
    Clean up PDF processing memory and resources
    """

    try:
        # Submit cleanup task
        task_id = await submit_pdf_processing_task(
            "cleanup_pdf_memory"
        )

        return {
            "success": True,
            "task_id": task_id,
            "message": "Memory cleanup initiated"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup memory: {str(e)}")

@router.get("/configuration")
async def get_processing_configuration(
    current_user: User = Depends(get_current_user)
):
    """
    Get current PDF processing configuration
    """

    from ....core.memory_optimized_pdf_processor import (
        PDF_MAX_FILE_SIZE, PDF_MAX_PAGES, PDF_MEMORY_LIMIT,
        PDF_CHUNK_SIZE, PDF_POOL_SIZE, PDF_PAGE_BATCH_SIZE,
        PDF_CHECKPOINT_INTERVAL, PDF_MEMORY_CHECK_INTERVAL, PDF_CACHE_TTL
    )

    return {
        "configuration": {
            "max_file_size_mb": PDF_MAX_FILE_SIZE / (1024 * 1024),
            "max_pages": PDF_MAX_PAGES,
            "memory_limit_mb": PDF_MEMORY_LIMIT / (1024 * 1024),
            "chunk_size_mb": PDF_CHUNK_SIZE / (1024 * 1024),
            "parser_pool_size": PDF_POOL_SIZE,
            "page_batch_size": PDF_PAGE_BATCH_SIZE,
            "checkpoint_interval": PDF_CHECKPOINT_INTERVAL,
            "memory_check_interval": PDF_MEMORY_CHECK_INTERVAL,
            "cache_ttl_hours": PDF_CACHE_TTL / 3600
        },
        "optimization_features": [
            "Memory-efficient streaming processing",
            "Object pool management for PDF parsers",
            "Real-time memory monitoring",
            "Checkpoint recovery system",
            "Medical entity extraction",
            "Batch processing optimization",
            "Automatic memory pressure handling"
        ]
    }

@router.post("/extract-text-only")
async def extract_text_only(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Extract text only from PDF without full processing
    """

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_path = temp_file.name

            # Save uploaded file
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await file.read()
                await f.write(content)

        # Submit text extraction task
        task_id = await submit_pdf_processing_task(
            "extract_text",
            temp_path
        )

        return {
            "success": True,
            "task_id": task_id,
            "message": "Text extraction started"
        }

    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

@router.get("/health")
async def processing_health_check():
    """
    Health check for PDF processing system
    """

    try:
        metrics = await get_pdf_processing_metrics()

        # Determine health status based on metrics
        memory_usage = metrics.get("memory_stats", {}).get("current_memory_percent", 0)
        error_rate = metrics.get("total_errors", 0) / max(metrics.get("total_processed", 1), 1)

        if memory_usage > 90 or error_rate > 0.1:
            status = "unhealthy"
        elif memory_usage > 80 or error_rate > 0.05:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "memory_usage_percent": memory_usage,
            "error_rate": error_rate,
            "total_processed": metrics.get("total_processed", 0),
            "active_parsers": metrics.get("parser_pool_stats", {}).get("active_parsers", 0),
            "timestamp": "now()"
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": "now()"
        }