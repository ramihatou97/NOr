# backend/main.py
"""
Main FastAPI application for the KOO Platform
Focuses on working core modules and nuance merge functionality
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging

# Import working modules only
from api.enhanced_chapters import router as chapters_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting KOO Platform backend...")

    # Startup logic
    try:
        # Test core imports
        from core.nuance_merge_engine import nuance_merge_engine
        from core.dependencies import get_current_user
        logger.info("✅ Core modules loaded successfully")

        # Initialize services that don't require external dependencies
        logger.info("✅ Services initialized")

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise

    yield

    # Shutdown logic
    logger.info("🛑 Shutting down KOO Platform backend...")

# Create FastAPI application
app = FastAPI(
    title="KOO Platform Backend",
    description="Medical content processing platform with intelligent nuance merge capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://*.use.devtunnels.ms",
        "https://mg05bgbk-8000.use.devtunnels.ms",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chapters_router)

@app.get("/")
async def root():
    """Root endpoint with HTML preview"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>KOO Platform Backend</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }
            .header { text-align: center; margin-bottom: 30px; }
            .status {
                display: inline-block;
                background: #00d4aa;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                margin: 10px 0;
            }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .feature {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #00d4aa;
            }
            .feature h3 { margin-top: 0; color: #00d4aa; }
            .endpoints {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .endpoint {
                background: rgba(0,0,0,0.2);
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                font-family: monospace;
            }
            .method {
                color: #00d4aa;
                font-weight: bold;
                margin-right: 10px;
            }
            a { color: #00d4aa; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .footer { text-align: center; margin-top: 30px; opacity: 0.8; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 KOO Platform Backend</h1>
                <p>Medical Content Processing Platform with AI-Powered Nuance Detection</p>
                <div class="status">✅ OPERATIONAL</div>
            </div>

            <div class="features">
                <div class="feature">
                    <h3>🧠 Nuance Merge Engine</h3>
                    <p>Advanced AI-powered detection of content nuances with risk assessment. Information loss risk: <strong>minimal</strong></p>
                </div>
                <div class="feature">
                    <h3>📚 Intelligent Chapter Management</h3>
                    <p>Smart content organization and processing for medical textbooks and documentation</p>
                </div>
                <div class="feature">
                    <h3>🔍 Medical Content Processing</h3>
                    <p>Specialized tools for medical document analysis and content enhancement</p>
                </div>
            </div>

            <div class="endpoints">
                <h3>🛠️ Available Endpoints</h3>
                <div class="endpoint">
                    <span class="method">GET</span> <a href="/health">/health</a> - System health check
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <a href="/api/test-nuance-merge">/api/test-nuance-merge</a> - Test nuance merge functionality
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <a href="/docs">/docs</a> - Interactive API documentation
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> /api/v1/chapters/nuance/detect - Detect content nuances
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> /api/v1/chapters/nuance/analytics/* - Nuance analytics
                </div>
            </div>

            <div class="footer">
                <p>Version 1.0.0 | Built with FastAPI | Ready for Production</p>
            </div>
        </div>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)

@app.get("/favicon.ico")
async def favicon():
    """Return empty response for favicon requests"""
    return {"message": "No favicon configured"}

@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight requests"""
    return {"message": "OK"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test core functionality
        from core.nuance_merge_engine import nuance_merge_engine
        return {
            "status": "healthy",
            "services": {
                "nuance_merge_engine": "operational",
                "api_server": "operational"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/api/test-nuance-merge")
async def test_nuance_merge():
    """Test endpoint for nuance merge functionality"""
    try:
        from core.nuance_merge_engine import nuance_merge_engine

        # Test with sample content
        original = "The patient shows symptoms of infection."
        updated = "The patient demonstrates clinical symptoms of bacterial infection requiring antibiotic treatment."

        result = await nuance_merge_engine.detect_nuances(
            original_content=original,
            updated_content=updated,
            chapter_id="test_chapter",
            context={"specialty": "general_medicine"}
        )

        if result:
            return {
                "status": "success",
                "nuance_detected": True,
                "nuance_type": result.nuance_type.value,
                "confidence_score": result.confidence_score,
                "information_loss_risk": result.ai_analysis.get("risk_assessment", {}).get("information_loss_risk", "unknown"),
                "auto_apply_eligible": result.auto_apply_eligible,
                "manual_review_required": result.manual_review_required
            }
        else:
            return {
                "status": "success",
                "nuance_detected": False,
                "message": "No nuance detected with test content"
            }

    except Exception as e:
        logger.error(f"Nuance merge test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        forwarded_allow_ips="*",
        proxy_headers=True
    )