#!/usr/bin/env python3
"""
KOO Platform - Isolated Demo Server
Completely standalone server for showcasing medical chapter editing capabilities
Runs on port 9000 - completely separate from main application
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import random
import time
from typing import Dict, Any
from datetime import datetime

# Create isolated FastAPI app
app = FastAPI(
    title="KOO Platform Demo",
    description="Isolated demonstration of medical chapter editing with AI intelligence",
    version="1.0.0-demo"
)

# CORS for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data for demonstrations
SAMPLE_CHAPTER = {
    "title": "Glioblastoma Multiforme: Advanced Surgical Management Protocols",
    "content": """
# Glioblastoma Multiforme: Advanced Surgical Management Protocols

## Introduction

Glioblastoma multiforme (GBM) represents the most aggressive primary brain tumor, requiring immediate surgical intervention and comprehensive multidisciplinary management. The evolution of neurosurgical techniques and intraoperative technologies has significantly enhanced our ability to achieve maximal safe resection while preserving neurological function.

## Preoperative Assessment and Planning

### 1. Advanced Neuroimaging

Multimodal imaging provides crucial information for surgical planning and tumor characterization.

**Essential Imaging Protocols:**
- High-resolution MRI with gadolinium enhancement
- Diffusion tensor imaging (DTI) for white matter tract visualization
- Functional MRI (fMRI) for eloquent area mapping
- Magnetic resonance spectroscopy (MRS) for metabolic assessment

![Glioblastoma MRI](assets/images/glioblastoma-mri.svg)
*Figure 1: T1-weighted MRI with gadolinium showing enhancing glioblastoma in right frontal lobe*

### 2. Functional Localization

Precise identification of eloquent brain regions is critical for optimal surgical outcomes.

**Mapping Techniques:**
- Awake craniotomy with cortical stimulation mapping
- Intraoperative neurophysiological monitoring
- Direct electrical stimulation of language areas
- Motor evoked potential monitoring

### 3. Surgical Planning Considerations

Contemporary surgical approach emphasizes maximal safe resection while minimizing postoperative morbidity.

**Key Planning Elements:**
- Tumor location relative to eloquent cortex
- Proximity to critical white matter tracts
- Vascular considerations and major vessel preservation
- Patient functional status and neurological baseline

## Intraoperative Management

### 1. Awake Craniotomy Protocols

Awake procedures allow real-time functional assessment during tumor resection.

**Anesthetic Considerations:**
- Asleep-awake-asleep technique
- Local anesthesia with conscious sedation
- Airway management protocols
- Patient positioning and comfort measures

### 2. Intraoperative Imaging

Real-time imaging guidance enhances surgical precision and extent of resection.

**Imaging Modalities:**
- Intraoperative MRI for real-time tumor visualization
- Ultrasound for tumor boundary identification
- Fluorescence-guided surgery with 5-ALA
- Neuronavigation system integration

![Intraoperative Navigation](assets/images/neuronavigation.svg)
*Figure 2: Intraoperative neuronavigation system showing real-time tumor localization*

![5-ALA Fluorescence](assets/images/5ala-fluorescence.svg)
*Figure 3: 5-ALA fluorescence-guided surgery showing tumor boundaries under blue light*

### 3. Electrocorticography and Mapping

Direct cortical stimulation provides functional guidance during resection.

**Mapping Protocols:**
- Language mapping in dominant hemisphere lesions
- Motor mapping for precentral gyrus tumors
- Sensory mapping for postcentral involvement
- Subcortical stimulation for white matter preservation

![Awake Craniotomy](assets/images/awake-craniotomy.svg)
*Figure 4: Awake craniotomy with patient performing language tasks during cortical mapping*

## Extent of Resection Optimization

### 1. Gross Total Resection Goals

Maximal safe resection correlates with improved survival outcomes.

**Resection Strategies:**
- Tumor boundary identification using multiple modalities
- Preservation of eloquent cortex and white matter
- Vascular preservation techniques
- Hemostasis and brain protection protocols

### 2. Residual Tumor Assessment

Intraoperative evaluation ensures optimal resection extent.

**Assessment Methods:**
- Real-time MRI evaluation
- Fluorescence visualization of residual tumor
- Ultrasound confirmation of resection cavity
- Frozen section histopathological analysis

## Postoperative Management

### 1. Immediate Postoperative Care

Comprehensive monitoring ensures optimal neurological outcomes.

**Monitoring Protocols:**
- Neurological examination every 2 hours initially
- Imaging surveillance for complications
- Intracranial pressure monitoring when indicated
- Early mobilization and rehabilitation

### 2. Adjuvant Therapy Considerations

Multimodal treatment improves long-term outcomes.

**Treatment Planning:**
- Radiation therapy planning and timing
- Temozolomide chemotherapy protocols
- Tumor treating fields (TTFields) consideration
- Clinical trial eligibility assessment

## Complications and Management

### 1. Surgical Complications

Prompt recognition and management of complications is essential.

**Common Complications:**
- Postoperative hematoma requiring evacuation
- Cerebral edema and mass effect
- Seizure activity and anticonvulsant management
- Infection prevention and treatment protocols

### 2. Functional Preservation

Maintaining neurological function is paramount in GBM surgery.

**Preservation Strategies:**
- Language function monitoring and protection
- Motor pathway preservation techniques
- Cognitive function assessment and optimization
- Quality of life considerations in surgical planning

## Contemporary Advances

### 1. Molecular Profiling Integration

Genetic and molecular characteristics guide treatment decisions.

**Key Biomarkers:**
- IDH mutation status and prognostic implications
- MGMT promoter methylation status
- 1p/19q codeletion analysis
- EGFR amplification and therapeutic targets

### 2. Emerging Technologies

Novel approaches continue to improve surgical outcomes.

**Innovative Techniques:**
- Laser interstitial thermal therapy (LITT)
- Convection-enhanced delivery systems
- Immunotherapy integration with surgery
- Artificial intelligence in surgical planning

## Conclusion

The comprehensive management of glioblastoma multiforme requires sophisticated surgical techniques, advanced imaging guidance, and meticulous attention to functional preservation. The integration of molecular profiling with surgical planning continues to evolve, offering new opportunities for personalized treatment approaches.

Ongoing research in neurosurgical techniques, imaging technologies, and adjuvant therapies continues to improve outcomes for patients with this challenging disease.
""",
    "specialty": "neurosurgery",
    "tags": ["glioblastoma", "brain tumor", "neurosurgery", "craniotomy", "awake surgery"]
}

MOCK_INTELLIGENCE_DATA = {
    "qualityAssessment": {
        "overallScore": 0.87,
        "dimensionScores": {
            "clarity": 0.92,
            "medical_accuracy": 0.89,
            "completeness": 0.85,
            "readability": 0.84,
            "clinical_relevance": 0.91
        },
        "improvementSuggestions": [
            "Consider adding more specific dosage information for medications",
            "Include recent clinical trial data to support recommendations",
            "Add differential diagnosis section"
        ]
    },
    "conflictAnalysis": {
        "conflicts": [
            {
                "type": "terminology_inconsistency",
                "description": "Troponin threshold values vary between sections",
                "severity": "medium",
                "suggestions": ["Standardize troponin reference values throughout"]
            }
        ]
    },
    "researchRecommendations": [
        {
            "title": "High-Sensitivity Troponin in Emergency Diagnosis",
            "abstract": "Recent advances in high-sensitivity troponin assays have improved early detection...",
            "relevanceScore": 0.94,
            "url": "https://pubmed.ncbi.nlm.nih.gov/example1",
            "authors": ["Smith, J.", "Johnson, K."],
            "year": 2024
        },
        {
            "title": "AI-Enhanced ECG Interpretation in Acute MI",
            "abstract": "Machine learning algorithms demonstrate superior performance in ECG analysis...",
            "relevanceScore": 0.88,
            "url": "https://pubmed.ncbi.nlm.nih.gov/example2",
            "authors": ["Chen, L.", "Williams, R."],
            "year": 2024
        }
    ],
    "workflowSuggestions": {
        "predictedProductivity": 0.82,
        "optimizations": [
            "Focus on biomarker section during peak concentration hours",
            "Schedule research validation for tomorrow morning"
        ]
    },
    "intelligenceSummary": {
        "nextLikelyActions": ["research_validation", "content_expansion"],
        "contentGaps": ["differential_diagnosis", "contraindications"],
        "expertiseRecommendations": ["consult_interventional_cardiologist"]
    }
}

@app.get("/", response_class=HTMLResponse)
async def demo_home():
    """Serve the demo HTML page"""
    return FileResponse("index.html")

@app.get("/assets/{file_path:path}")
async def serve_assets(file_path: str):
    """Serve static assets"""
    return FileResponse(f"assets/{file_path}")

@app.get("/api/health")
async def demo_health():
    """Demo health check"""
    return {
        "status": "healthy",
        "service": "KOO Platform Demo",
        "port": 9000,
        "timestamp": datetime.utcnow().isoformat(),
        "isolation": "complete"
    }

@app.get("/api/sample-chapter")
async def get_sample_chapter():
    """Get sample chapter data"""
    return SAMPLE_CHAPTER

@app.post("/api/analyze-content")
async def analyze_content(data: Dict[str, Any]):
    """Mock content analysis with realistic response time"""
    # Simulate processing time
    await asyncio.sleep(random.uniform(0.5, 1.5))

    # Add some variation to make it feel real
    mock_data = MOCK_INTELLIGENCE_DATA.copy()
    mock_data["qualityAssessment"]["overallScore"] += random.uniform(-0.05, 0.05)
    mock_data["timestamp"] = datetime.utcnow().isoformat()
    mock_data["contentLength"] = len(data.get("content", ""))

    return {
        "success": True,
        "data": mock_data,
        "processingTime": random.randint(800, 1500)
    }

@app.post("/api/test-nuance-merge")
async def test_nuance_merge():
    """Mock nuance merge demonstration"""
    await asyncio.sleep(1.0)  # Simulate processing

    return {
        "status": "success",
        "nuance_detected": True,
        "nuance_type": "enhancement",
        "confidence_score": 0.87,
        "information_loss_risk": "minimal",
        "auto_apply_eligible": False,
        "manual_review_required": True,
        "original": "The patient shows symptoms of cardiac distress.",
        "enhanced": "The patient demonstrates clinical manifestations consistent with acute coronary syndrome requiring immediate evaluation.",
        "changes": {
            "added_terms": ["clinical manifestations", "acute coronary syndrome", "immediate evaluation"],
            "improved_specificity": True,
            "medical_accuracy_enhanced": True
        },
        "processing_time_ms": 950
    }

@app.post("/api/ai/gemini-analysis")
async def gemini_content_analysis(data: Dict[str, Any]):
    """Enhanced content analysis using Google Gemini AI"""
    content = data.get("content", "")
    analysis_type = data.get("type", "comprehensive")  # comprehensive, medical_accuracy, clarity, completeness

    await asyncio.sleep(random.uniform(1.2, 2.5))  # Simulate Gemini processing time

    # Mock Gemini-style responses
    if analysis_type == "medical_accuracy":
        return {
            "ai_model": "Gemini Pro",
            "analysis_type": "Medical Accuracy Assessment",
            "overall_score": 0.91,
            "findings": [
                {
                    "category": "Clinical Terminology",
                    "score": 0.94,
                    "notes": "Precise use of neurosurgical terminology throughout"
                },
                {
                    "category": "Evidence-Based Content",
                    "score": 0.88,
                    "notes": "Well-supported by current literature"
                },
                {
                    "category": "Diagnostic Accuracy",
                    "score": 0.92,
                    "notes": "Accurate diagnostic criteria and protocols"
                }
            ],
            "recommendations": [
                "Consider adding latest WHO classification updates",
                "Include molecular subtyping information",
                "Reference recent clinical trial outcomes"
            ],
            "confidence": 0.89,
            "processing_time": random.randint(1200, 2500)
        }
    elif analysis_type == "clarity":
        return {
            "ai_model": "Gemini Pro",
            "analysis_type": "Content Clarity Analysis",
            "readability_score": 0.85,
            "clarity_metrics": {
                "sentence_complexity": "Appropriate for medical professionals",
                "terminology_consistency": 0.93,
                "logical_flow": 0.87,
                "section_organization": 0.91
            },
            "suggestions": [
                "Consider shorter sentences in complex procedural sections",
                "Add more transitional phrases between major topics",
                "Include visual aids for complex anatomical descriptions"
            ],
            "target_audience": "Medical professionals and residents",
            "confidence": 0.86
        }
    else:  # comprehensive
        return {
            "ai_model": "Gemini Pro",
            "analysis_type": "Comprehensive Content Analysis",
            "overall_assessment": {
                "quality_score": 0.89,
                "medical_accuracy": 0.91,
                "completeness": 0.87,
                "clarity": 0.85,
                "evidence_support": 0.88
            },
            "content_insights": {
                "word_count": len(content.split()) if content else 0,
                "reading_time": f"{len(content.split()) // 200 + 1} minutes" if content else "0 minutes",
                "complexity_level": "Advanced medical professional",
                "key_topics": ["Glioblastoma", "Surgical techniques", "Intraoperative monitoring", "Patient outcomes"]
            },
            "ai_recommendations": [
                "Excellent coverage of current surgical techniques",
                "Strong integration of imaging and surgical planning",
                "Could benefit from more outcome statistics",
                "Consider adding patient selection criteria"
            ],
            "next_actions": [
                "Add recent meta-analysis references",
                "Include institutional protocol variations",
                "Expand on postoperative management"
            ],
            "gemini_confidence": 0.91,
            "processing_time": random.randint(1500, 3000)
        }

@app.post("/api/ai/gemini-enhancement")
async def gemini_content_enhancement(data: Dict[str, Any]):
    """Content enhancement suggestions using Gemini AI"""
    content = data.get("content", "")
    enhancement_type = data.get("type", "medical_precision")

    await asyncio.sleep(random.uniform(1.5, 2.8))

    return {
        "ai_model": "Gemini Pro",
        "enhancement_type": enhancement_type,
        "suggestions": [
            {
                "section": "Introduction",
                "current": "Glioblastoma is a malignant brain tumor.",
                "enhanced": "Glioblastoma multiforme (WHO Grade IV astrocytoma) represents the most aggressive primary central nervous system malignancy with a median survival of 12-15 months despite multimodal therapy.",
                "improvement_type": "Medical precision and specificity",
                "confidence": 0.94
            },
            {
                "section": "Surgical Technique",
                "current": "Complete resection is important.",
                "enhanced": "Gross total resection (GTR), defined as >98% tumor removal with minimal residual enhancement on postoperative MRI, significantly correlates with improved progression-free survival.",
                "improvement_type": "Quantitative precision",
                "confidence": 0.91
            },
            {
                "section": "Outcomes",
                "current": "Surgery improves outcomes.",
                "enhanced": "Maximal safe resection combined with adjuvant temozolomide and radiation therapy (Stupp protocol) demonstrates a statistically significant survival benefit with median overall survival extending from 12.1 to 14.6 months (p<0.001).",
                "improvement_type": "Evidence-based specificity",
                "confidence": 0.88
            }
        ],
        "overall_enhancement_score": 0.91,
        "medical_accuracy_improvement": "+15%",
        "evidence_integration": "+22%",
        "processing_time": random.randint(1800, 3200)
    }

@app.get("/api/live-metrics")
async def get_live_metrics():
    """Generate live-updating metrics for demonstration"""
    return {
        "quality_score": round(random.uniform(0.75, 0.95), 2),
        "conflicts_detected": random.randint(0, 3),
        "research_opportunities": random.randint(1, 5),
        "workflow_efficiency": round(random.uniform(0.70, 0.90), 2),
        "ai_confidence": round(random.uniform(0.80, 0.98), 2),
        "last_updated": datetime.utcnow().isoformat()
    }

@app.get("/api/typing-simulation")
async def simulate_typing():
    """Simulate real-time typing analysis"""
    responses = [
        {"suggestion": "Consider adding diagnostic criteria", "confidence": 0.85},
        {"suggestion": "Include contraindications section", "confidence": 0.79},
        {"suggestion": "Add recent clinical evidence", "confidence": 0.92},
        {"suggestion": "Specify dosage recommendations", "confidence": 0.88}
    ]
    return random.choice(responses)

@app.get("/api/config/api-keys")
async def get_api_keys():
    """Get configured API keys (masked for security)"""
    return {
        "openai": {"configured": True, "key": "sk-...J3kL", "status": "active"},
        "pubmed": {"configured": True, "key": "pm_...9fA2", "status": "active"},
        "google_scholar": {"configured": True, "key": "gs_...4bN8", "status": "active"},
        "perplexity": {"configured": True, "key": "pplx_...2mL5", "status": "active"},
        "medline": {"configured": False, "key": None, "status": "inactive"},
        "claude": {"configured": True, "key": "cl-...7nM9", "status": "active"},
        "gemini": {"configured": True, "key": "AIza...8kL4", "status": "active"},
        "palm": {"configured": False, "key": None, "status": "inactive"},
        "together": {"configured": True, "key": "tog_...k8P1", "status": "active"}
    }

@app.post("/api/config/api-keys")
async def update_api_keys(data: Dict[str, Any]):
    """Update API key configuration"""
    await asyncio.sleep(0.5)  # Simulate validation
    return {
        "success": True,
        "message": f"API key for {data.get('service')} updated successfully",
        "validated": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/pdf-library")
async def get_pdf_library():
    """Get PDF reference library"""
    return {
        "total_papers": 15847,
        "recent_uploads": [
            {
                "id": "pdf_001",
                "title": "Advances in Glioblastoma Treatment 2024",
                "authors": ["Chen, L.", "Rodriguez, M.", "Kim, S."],
                "journal": "Nature Neuroscience",
                "year": 2024,
                "doi": "10.1038/nn.2024.123",
                "relevance": 0.95,
                "uploaded": "2024-01-15T10:30:00Z",
                "abstract": "Recent advances in molecular profiling and targeted therapies have significantly improved outcomes for glioblastoma patients...",
                "key_findings": [
                    "IDH mutation status correlates with treatment response",
                    "Combination therapy shows 30% improvement in survival",
                    "Novel biomarkers identified for treatment selection"
                ],
                "referenceable": True
            },
            {
                "id": "pdf_002",
                "title": "5-ALA Fluorescence in Brain Tumor Surgery",
                "authors": ["Mueller, T.", "Wang, K."],
                "journal": "Journal of Neurosurgery",
                "year": 2024,
                "doi": "10.3171/jns.2024.456",
                "relevance": 0.92,
                "uploaded": "2024-01-10T14:20:00Z",
                "abstract": "5-aminolevulinic acid (5-ALA) fluorescence-guided surgery enhances gross total resection rates in glioblastoma...",
                "key_findings": [
                    "98% sensitivity for tumor tissue identification",
                    "Improved extent of resection by 15%",
                    "Reduced recurrence rates at 6 months"
                ],
                "referenceable": True
            },
            {
                "id": "pdf_003",
                "title": "Awake Craniotomy Protocols for Language Mapping",
                "authors": ["Wilson, J.", "Brown, A.", "Davis, M."],
                "journal": "Neurosurgery",
                "year": 2024,
                "doi": "10.1227/neu.2024.789",
                "relevance": 0.89,
                "uploaded": "2024-01-08T09:15:00Z",
                "abstract": "Standardized protocols for awake craniotomy with language mapping improve functional outcomes...",
                "key_findings": [
                    "95% language preservation rate",
                    "Reduced operative time by 20%",
                    "Improved patient satisfaction scores"
                ],
                "referenceable": True
            }
        ],
        "categories": {
            "neurosurgery": 2847,
            "oncology": 1923,
            "radiology": 1456,
            "pathology": 987,
            "pharmacology": 734
        }
    }

@app.post("/api/pdf-library/search")
async def search_pdf_library(data: Dict[str, Any]):
    """Search PDF library with enhanced results"""
    query = data.get("query", "")
    source = data.get("source", "local")  # local, pubmed, google_scholar, perplexity

    await asyncio.sleep(random.uniform(0.8, 2.0))  # Simulate search time

    if source == "pubmed":
        results = [
            {
                "id": "pubmed_001",
                "title": f"PubMed: {query} in Neurosurgical Practice",
                "authors": ["Smith, R.", "Johnson, K."],
                "journal": "Neurosurgery",
                "year": 2024,
                "pmid": "38745621",
                "relevance": 0.94,
                "abstract": f"Comprehensive review of {query} applications in neurosurgical practice...",
                "source": "PubMed"
            },
            {
                "id": "pubmed_002",
                "title": f"Clinical Outcomes of {query} in Brain Surgery",
                "authors": ["Lee, S.", "Wang, H."],
                "journal": "Journal of Neurosurgery",
                "year": 2024,
                "pmid": "38756893",
                "relevance": 0.91,
                "abstract": f"Multi-center study evaluating {query} effectiveness in brain tumor surgery...",
                "source": "PubMed"
            }
        ]
    elif source == "google_scholar":
        results = [
            {
                "id": "scholar_001",
                "title": f"Google Scholar: Advanced {query} Techniques",
                "authors": ["Chen, M.", "Rodriguez, P."],
                "journal": "Nature Medicine",
                "year": 2024,
                "citations": 47,
                "relevance": 0.96,
                "abstract": f"Novel approaches to {query} demonstrate improved patient outcomes...",
                "source": "Google Scholar"
            }
        ]
    elif source == "perplexity":
        results = [
            {
                "id": "perplexity_001",
                "title": f"Perplexity AI: Latest Research on {query}",
                "authors": ["AI Generated Summary"],
                "summary": f"Recent developments in {query} show promising results across multiple studies...",
                "confidence": 0.93,
                "sources_count": 15,
                "relevance": 0.88,
                "source": "Perplexity"
            }
        ]
    else:  # local library
        results = [
            {
                "id": "local_001",
                "title": f"Local Library: {query} Reference Materials",
                "authors": ["Various Authors"],
                "relevance": 0.87,
                "matches": 23,
                "source": "Local Library"
            }
        ]

    return {
        "query": query,
        "source": source,
        "total_results": len(results),
        "search_time": random.randint(850, 2500),
        "results": results
    }

@app.post("/api/pdf-library/reference")
async def add_reference_to_chapter(data: Dict[str, Any]):
    """Add a reference from PDF library to current chapter"""
    reference_id = data.get("reference_id")
    citation_style = data.get("citation_style", "ama")  # ama, mla, apa, vancouver
    position = data.get("position", "end")  # end, cursor, specific_line

    await asyncio.sleep(0.5)

    # Mock reference formatting based on style
    if citation_style == "ama":
        formatted_citation = "Chen L, Rodriguez M, Kim S. Advances in Glioblastoma Treatment 2024. Nature Neuroscience. 2024;15(3):123-135."
    elif citation_style == "vancouver":
        formatted_citation = "Chen L, Rodriguez M, Kim S. Advances in Glioblastoma Treatment 2024. Nat Neurosci. 2024;15(3):123-35."
    else:
        formatted_citation = "Chen, L., Rodriguez, M., & Kim, S. (2024). Advances in Glioblastoma Treatment 2024. Nature Neuroscience, 15(3), 123-135."

    return {
        "success": True,
        "reference_id": reference_id,
        "formatted_citation": formatted_citation,
        "citation_number": random.randint(1, 50),
        "insertion_position": position,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/live-chapters")
async def get_live_chapters():
    """Get active collaborative chapters"""
    return {
        "active_sessions": 3,
        "chapters": [
            {
                "id": "ch_001",
                "title": "Glioblastoma Multiforme: Advanced Surgical Management",
                "collaborators": ["Dr. Smith", "Dr. Johnson", "Dr. Lee"],
                "last_edit": "2024-01-15T15:45:00Z",
                "status": "active",
                "changes_pending": 7
            },
            {
                "id": "ch_002",
                "title": "Meningioma Resection Protocols",
                "collaborators": ["Dr. Wilson", "Dr. Brown"],
                "last_edit": "2024-01-15T14:30:00Z",
                "status": "review",
                "changes_pending": 2
            }
        ],
        "recent_activity": [
            {"user": "Dr. Smith", "action": "edited section", "timestamp": "2 minutes ago"},
            {"user": "Dr. Johnson", "action": "added reference", "timestamp": "5 minutes ago"},
            {"user": "Dr. Lee", "action": "commented", "timestamp": "8 minutes ago"}
        ]
    }

@app.get("/api/auth/user")
async def get_current_user():
    """Get current user information"""
    return {
        "user_id": "usr_12345",
        "name": "Dr. Sarah Chen",
        "email": "s.chen@hospital.edu",
        "role": "Senior Neurosurgeon",
        "institution": "Johns Hopkins Medical Center",
        "specialties": ["neurosurgery", "brain tumors", "skull base"],
        "permissions": {
            "create_chapters": True,
            "edit_all": True,
            "manage_users": False,
            "access_ai": True,
            "export_data": True
        },
        "subscription": {
            "plan": "Professional",
            "expires": "2024-12-31",
            "ai_credits": 1250,
            "storage_gb": 500
        }
    }

@app.get("/api/integrations")
async def get_integrations():
    """Get external system integrations"""
    return {
        "emr_systems": {
            "epic": {"connected": True, "last_sync": "2024-01-15T16:00:00Z"},
            "cerner": {"connected": False, "last_sync": None},
            "allscripts": {"connected": True, "last_sync": "2024-01-15T15:30:00Z"}
        },
        "databases": {
            "pubmed": {"status": "active", "queries_today": 47},
            "google_scholar": {"status": "active", "queries_today": 23},
            "perplexity": {"status": "active", "queries_today": 18},
            "cochrane": {"status": "active", "queries_today": 12},
            "uptodate": {"status": "inactive", "queries_today": 0}
        },
        "ai_services": {
            "gpt4": {"status": "active", "calls_today": 156},
            "claude": {"status": "active", "calls_today": 89},
            "gemini": {"status": "active", "calls_today": 73},
            "palm": {"status": "inactive", "calls_today": 0}
        }
    }

import asyncio

if __name__ == "__main__":
    print("[HOSPITAL] Starting KOO Platform Demo Server...")
    print("[LOCATION] Completely isolated from main application")
    print("[WEB] Access demo at: http://localhost:9000")
    print("[WARNING] This demo can be deleted without affecting the main app")
    print()

    uvicorn.run(
        "demo_server:app",
        host="0.0.0.0",
        port=9000,
        reload=True,
        log_level="info"
    )