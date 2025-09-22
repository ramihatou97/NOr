# backend/core/medical_image_intelligence.py
"""
Medical Image Intelligence System
Advanced image analysis for anatomy and radiology with automatic chapter integration
"""

import asyncio
import base64
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
from datetime import datetime, timedelta
import json

class ImageType(Enum):
    ANATOMY = "anatomy"
    RADIOLOGY_CT = "radiology_ct"
    RADIOLOGY_MRI = "radiology_mri"
    RADIOLOGY_XRAY = "radiology_xray"
    HISTOLOGY = "histology"
    SURGICAL = "surgical"
    DIAGRAM = "diagram"
    CHART = "chart"

class AnatomicalRegion(Enum):
    BRAIN = "brain"
    SPINE = "spine"
    HEAD_NECK = "head_neck"
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    GASTROINTESTINAL = "gastrointestinal"
    MUSCULOSKELETAL = "musculoskeletal"
    NEUROLOGICAL = "neurological"

@dataclass
class ImageAnalysisResult:
    image_id: str
    image_type: ImageType
    anatomical_region: Optional[AnatomicalRegion]
    pathology_detected: List[str]
    anatomical_structures: List[str]
    image_quality_score: float
    clinical_relevance_score: float
    diagnostic_confidence: float
    caption: str
    detailed_description: str
    medical_terminology: List[str]
    differential_diagnoses: List[str]
    teaching_points: List[str]
    related_conditions: List[str]
    metadata: Dict[str, Any]

@dataclass
class ImageRecommendation:
    image_id: str
    relevance_score: float
    placement_suggestion: str  # "introduction", "diagnosis", "treatment", "conclusion"
    caption_suggestion: str
    context_explanation: str
    educational_value: float

class MedicalImageIntelligence:
    def __init__(self):
        self.vision_models = {
            "gpt_vision": "gpt-4-vision-preview",
            "claude_vision": "claude-3-opus-20240229",
            "gemini_vision": "gemini-pro-vision"
        }

        # Initialize local models for offline processing
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

        # Medical image classification models (would be trained specifically)
        self.anatomical_classifier = None  # Custom trained model
        self.pathology_detector = None     # Custom trained model

        # Image enhancement tools
        self.enhancement_pipeline = self._setup_enhancement_pipeline()

        # Knowledge base for medical images
        self.medical_image_database = {}
        self.anatomical_atlas = {}

    async def analyze_medical_image(self, image_path: str, context: Dict[str, Any]) -> ImageAnalysisResult:
        """Comprehensive analysis of medical images using multiple AI models"""

        # Load and preprocess image
        image = await self._load_and_preprocess_image(image_path)

        # Run parallel analysis
        analysis_tasks = [
            self._analyze_with_vision_ai(image, context),
            self._classify_image_type(image),
            self._detect_anatomical_structures(image),
            self._assess_image_quality(image),
            self._generate_medical_caption(image, context),
            self._identify_pathology(image, context),
            self._extract_teaching_points(image, context)
        ]

        results = await asyncio.gather(*analysis_tasks)

        # Combine results into comprehensive analysis
        return ImageAnalysisResult(
            image_id=f"img_{datetime.now().timestamp()}",
            image_type=results[1],
            anatomical_region=results[2].get("primary_region"),
            pathology_detected=results[5],
            anatomical_structures=results[2].get("structures", []),
            image_quality_score=results[3],
            clinical_relevance_score=await self._calculate_clinical_relevance(results, context),
            diagnostic_confidence=results[0].get("confidence", 0.8),
            caption=results[4],
            detailed_description=results[0].get("description", ""),
            medical_terminology=results[0].get("terminology", []),
            differential_diagnoses=results[0].get("differential_diagnoses", []),
            teaching_points=results[6],
            related_conditions=results[0].get("related_conditions", []),
            metadata={
                "analysis_timestamp": datetime.now(),
                "models_used": list(self.vision_models.keys()),
                "processing_time": "calculated",
                "image_dimensions": image.size if hasattr(image, 'size') else None
            }
        )

    async def find_relevant_images_for_chapter(self, chapter_content: str,
                                             chapter_title: str,
                                             specialty: str) -> List[ImageRecommendation]:
        """Find and recommend most appropriate images for chapter content"""

        # Extract medical concepts from chapter
        medical_concepts = await self._extract_medical_concepts(chapter_content)

        # Search image database
        candidate_images = await self._search_medical_image_database(
            concepts=medical_concepts,
            specialty=specialty,
            content_type="chapter"
        )

        # Score and rank images
        recommendations = []
        for image in candidate_images:
            relevance_score = await self._calculate_image_relevance(
                image, medical_concepts, chapter_content
            )

            if relevance_score > 0.7:  # High relevance threshold
                placement = await self._suggest_image_placement(
                    image, chapter_content, chapter_title
                )

                recommendations.append(ImageRecommendation(
                    image_id=image["id"],
                    relevance_score=relevance_score,
                    placement_suggestion=placement["section"],
                    caption_suggestion=await self._generate_contextual_caption(
                        image, chapter_content, placement["context"]
                    ),
                    context_explanation=placement["reasoning"],
                    educational_value=await self._assess_educational_value(image, medical_concepts)
                ))

        # Sort by relevance and educational value
        recommendations.sort(
            key=lambda x: x.relevance_score * x.educational_value,
            reverse=True
        )

        return recommendations[:5]  # Top 5 recommendations

    async def _analyze_with_vision_ai(self, image: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use advanced vision AI models for medical image analysis"""

        # Convert image to base64 for API calls
        image_b64 = await self._image_to_base64(image)

        # Parallel calls to multiple vision AI services
        analysis_tasks = []

        # OpenAI GPT-4 Vision
        if "OPENAI_API_KEY" in context:
            analysis_tasks.append(self._analyze_with_gpt_vision(image_b64, context))

        # Claude Vision
        if "ANTHROPIC_API_KEY" in context:
            analysis_tasks.append(self._analyze_with_claude_vision(image_b64, context))

        # Gemini Vision
        if "GOOGLE_API_KEY" in context:
            analysis_tasks.append(self._analyze_with_gemini_vision(image_b64, context))

        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        # Combine and synthesize results
        return await self._synthesize_vision_analysis(results)

    async def _analyze_with_gpt_vision(self, image_b64: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze medical image using GPT-4 Vision"""

        prompt = f"""
        As a medical AI assistant specializing in {context.get('specialty', 'general medicine')},
        analyze this medical image in detail:

        1. Identify the image type (anatomy, radiology, histology, etc.)
        2. Describe anatomical structures visible
        3. Identify any pathological findings
        4. Provide differential diagnoses if applicable
        5. List key teaching points
        6. Suggest clinical relevance
        7. Rate diagnostic confidence (0-1)

        Focus on accuracy and clinical utility for medical education.
        """

        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            return {
                "source": "gpt_vision",
                "analysis": response.choices[0].message.content,
                "confidence": 0.9,
                "timestamp": datetime.now()
            }
        except Exception as e:
            return {"source": "gpt_vision", "error": str(e)}

    async def _analyze_with_claude_vision(self, image_b64: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze medical image using Claude Vision"""

        # Implementation for Claude Vision API
        # Similar structure to GPT vision but with Claude-specific API calls
        pass

    async def _analyze_with_gemini_vision(self, image_b64: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze medical image using Gemini Vision"""

        # Implementation for Gemini Vision API
        pass

    async def enhance_image_for_medical_use(self, image_path: str,
                                          enhancement_type: str = "auto") -> str:
        """Enhance medical images for better visibility and analysis"""

        image = cv2.imread(image_path)

        if enhancement_type == "auto":
            # Automatic enhancement based on image type detection
            image_type = await self._detect_medical_image_type(image)
            enhancement_type = self._get_optimal_enhancement(image_type)

        enhanced_image = image.copy()

        if enhancement_type == "contrast":
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            if len(enhanced_image.shape) == 3:
                lab = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                enhanced_image = clahe.apply(enhanced_image)

        elif enhancement_type == "brightness":
            # Gamma correction for brightness
            gamma = 1.5
            enhanced_image = np.power(enhanced_image / 255.0, gamma) * 255
            enhanced_image = enhanced_image.astype(np.uint8)

        elif enhancement_type == "sharpening":
            # Unsharp masking for sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

        # Save enhanced image
        enhanced_path = image_path.replace(".jpg", "_enhanced.jpg")
        cv2.imwrite(enhanced_path, enhanced_image)

        return enhanced_path

    async def create_medical_image_atlas(self, specialty: str) -> Dict[str, Any]:
        """Create a comprehensive atlas of medical images for a specialty"""

        atlas = {
            "specialty": specialty,
            "created_at": datetime.now(),
            "categories": {},
            "anatomical_regions": {},
            "pathologies": {},
            "teaching_collections": {}
        }

        # Organize images by medical relevance
        all_images = await self._get_specialty_images(specialty)

        for image in all_images:
            analysis = await self.analyze_medical_image(image["path"], {"specialty": specialty})

            # Categorize by anatomical region
            if analysis.anatomical_region:
                region = analysis.anatomical_region.value
                if region not in atlas["anatomical_regions"]:
                    atlas["anatomical_regions"][region] = []
                atlas["anatomical_regions"][region].append({
                    "image_id": analysis.image_id,
                    "quality_score": analysis.image_quality_score,
                    "teaching_points": analysis.teaching_points
                })

            # Categorize by pathology
            for pathology in analysis.pathology_detected:
                if pathology not in atlas["pathologies"]:
                    atlas["pathologies"][pathology] = []
                atlas["pathologies"][pathology].append(analysis.image_id)

        return atlas

    async def _setup_enhancement_pipeline(self):
        """Setup image enhancement pipeline for medical images"""
        return {
            "radiology": ["contrast", "noise_reduction", "edge_enhancement"],
            "anatomy": ["brightness", "color_balance", "sharpening"],
            "histology": ["color_enhancement", "contrast", "detail_preservation"],
            "surgical": ["brightness", "color_correction", "clarity"]
        }

    async def _extract_medical_concepts(self, text: str) -> List[str]:
        """Extract medical concepts from text using NLP"""
        # Implementation for medical concept extraction
        pass

    async def _search_medical_image_database(self, concepts: List[str],
                                           specialty: str,
                                           content_type: str) -> List[Dict[str, Any]]:
        """Search medical image database using semantic similarity"""
        # Implementation for semantic image search
        pass

# Global medical image intelligence instance
medical_image_intelligence = MedicalImageIntelligence()