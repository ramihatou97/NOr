/**
 * KOO Platform Demo - Interactive JavaScript
 * Comprehensive medical AI demonstration with full feature showcase
 */

class KOODemo {
  constructor() {
    this.baseURL = 'http://localhost:9000';
    this.isAnalyzing = false;
    this.typingTimer = null;
    this.metricsUpdateInterval = null;
    this.currentSuggestionIndex = 0;

    this.suggestions = [
      {
        type: 'quality',
        icon: 'fas fa-chart-line',
        title: 'Improve Medical Accuracy',
        description: 'Consider adding specific troponin threshold values (>99th percentile)',
        confidence: 0.89
      },
      {
        type: 'research',
        icon: 'fas fa-microscope',
        title: 'Latest Research Integration',
        description: 'Recent studies show high-sensitivity troponin reduces diagnostic time',
        confidence: 0.94
      },
      {
        type: 'clarity',
        icon: 'fas fa-lightbulb',
        title: 'Enhance Readability',
        description: 'Break down complex diagnostic criteria into bullet points',
        confidence: 0.76
      },
      {
        type: 'completeness',
        icon: 'fas fa-clipboard-check',
        title: 'Add Clinical Context',
        description: 'Include differential diagnosis considerations for chest pain',
        confidence: 0.88
      }
    ];

    this.init();
  }

  async init() {
    this.showLoadingOverlay();
    await this.loadSampleContent();
    this.bindEvents();
    this.startLiveMetrics();
    this.hideLoadingOverlay();
    this.animateOnLoad();
  }

  showLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
      overlay.style.display = 'flex';
    }
  }

  hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
      setTimeout(() => {
        overlay.classList.add('hidden');
        setTimeout(() => {
          overlay.style.display = 'none';
        }, 500);
      }, 1500);
    }
  }

  animateOnLoad() {
    // Add fade-in animations to elements
    const elements = document.querySelectorAll('.editor-main, .intelligence-sidebar, .feature-card');
    elements.forEach((el, index) => {
      setTimeout(() => {
        el.classList.add('fade-in');
      }, index * 200);
    });
  }

  async loadSampleContent() {
    try {
      const response = await fetch(`${this.baseURL}/api/sample-chapter`);
      const chapter = await response.json();

      document.getElementById('chapter-title').value = chapter.title;
      document.getElementById('content-editor').value = chapter.content;

      this.showNotification('Sample medical chapter loaded successfully', 'success');
    } catch (error) {
      console.error('Failed to load sample content:', error);
      this.showNotification('Using offline sample content', 'info');
      this.loadOfflineSample();
    }
  }

  loadOfflineSample() {
    const offlineContent = `# Acute Myocardial Infarction: Advanced Diagnostic Protocols

## Introduction

Acute myocardial infarction (AMI) represents a critical cardiovascular emergency requiring immediate recognition and intervention. The rapid evolution of diagnostic technologies has significantly enhanced our ability to identify and stratify patients presenting with suspected AMI.

## Primary Diagnostic Approaches

### 1. Electrocardiographic Assessment

The 12-lead electrocardiogram remains the cornerstone of initial AMI diagnosis. ST-elevation patterns in anatomically contiguous leads provide crucial localization information for the affected coronary territory.

**Key ECG Findings:**
- ST elevation ≥1mm in limb leads or ≥2mm in precordial leads
- Reciprocal changes in opposing leads
- Q-wave development indicating transmural infarction
- T-wave inversions suggesting ischemic memory

### 2. Cardiac Biomarker Analysis

High-sensitivity troponin assays have revolutionized the biochemical diagnosis of myocardial injury. The temporal profile of biomarker release provides insights into infarct timing and size.

### 3. Advanced Imaging Modalities

Contemporary imaging techniques offer real-time assessment of myocardial perfusion and wall motion abnormalities.`;

    document.getElementById('content-editor').value = offlineContent;
  }

  bindEvents() {
    // Content analysis
    const analyzeBtn = document.getElementById('analyze-btn');
    const nuanceBtn = document.getElementById('nuance-test-btn');
    const contentEditor = document.getElementById('content-editor');

    if (analyzeBtn) {
      analyzeBtn.addEventListener('click', () => this.analyzeContent());
    }

    if (nuanceBtn) {
      nuanceBtn.addEventListener('click', () => this.testNuanceMerge());
    }

    if (contentEditor) {
      contentEditor.addEventListener('input', () => this.onContentChange());
      contentEditor.addEventListener('focus', () => this.startTypingSuggestions());
      contentEditor.addEventListener('blur', () => this.stopTypingSuggestions());
    }

    // Demo feature buttons
    window.demonstrateNuance = () => this.demonstrateNuance();
    window.demonstrateQuality = () => this.demonstrateQuality();
    window.demonstrateResearch = () => this.demonstrateResearch();
    window.resetDemo = () => this.resetDemo();
    window.toggleFullscreen = () => this.toggleFullscreen();

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'Enter':
            e.preventDefault();
            this.analyzeContent();
            break;
          case 's':
            e.preventDefault();
            this.showNotification('Auto-save enabled in full version', 'info');
            break;
        }
      }
    });
  }

  onContentChange() {
    clearTimeout(this.typingTimer);
    this.typingTimer = setTimeout(() => {
      this.updateWordCount();
      this.simulateRealTimeAnalysis();
    }, 1000);
  }

  updateWordCount() {
    const content = document.getElementById('content-editor').value;
    const wordCount = content.trim().split(/\s+/).length;
    const charCount = content.length;

    // Update stats (if elements exist)
    const wordCountEl = document.querySelector('.word-count');
    if (wordCountEl) {
      wordCountEl.textContent = `${wordCount} words, ${charCount} characters`;
    }
  }

  async analyzeContent() {
    if (this.isAnalyzing) return;

    this.isAnalyzing = true;
    this.showAnalysisOverlay();

    const content = document.getElementById('content-editor').value;
    const title = document.getElementById('chapter-title').value;

    try {
      const response = await fetch(`${this.baseURL}/api/analyze-content`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, title })
      });

      const result = await response.json();

      if (result.success) {
        this.updateIntelligenceDisplay(result.data);
        this.showNotification('Content analysis completed successfully', 'success');
      } else {
        throw new Error('Analysis failed');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      this.simulateOfflineAnalysis();
      this.showNotification('Using offline analysis mode', 'warning');
    }

    this.hideAnalysisOverlay();
    this.isAnalyzing = false;
  }

  simulateOfflineAnalysis() {
    const mockData = {
      qualityAssessment: {
        overallScore: 0.87 + (Math.random() - 0.5) * 0.1,
        dimensionScores: {
          clarity: 0.92,
          medical_accuracy: 0.89,
          completeness: 0.85,
          readability: 0.84,
          clinical_relevance: 0.91
        }
      },
      conflictAnalysis: { conflicts: [{ type: 'minor', severity: 'low' }] },
      researchRecommendations: [
        { title: 'High-Sensitivity Troponin', relevanceScore: 0.94 },
        { title: 'AI-Enhanced ECG', relevanceScore: 0.88 }
      ]
    };

    this.updateIntelligenceDisplay(mockData);
  }

  updateIntelligenceDisplay(data) {
    // Update quality score
    const qualityScore = Math.round(data.qualityAssessment.overallScore * 100);
    document.getElementById('quality-score').textContent = `${qualityScore}%`;
    document.getElementById('live-quality').textContent = `${qualityScore}%`;

    // Update progress bar
    const qualityBar = document.querySelector('.quality-score');
    if (qualityBar) {
      qualityBar.style.width = `${qualityScore}%`;
    }

    // Update conflicts
    const conflictCount = data.conflictAnalysis?.conflicts?.length || 1;
    document.getElementById('live-conflicts').textContent = conflictCount;
    document.getElementById('conflicts-pill').innerHTML =
      `<i class="fas fa-check"></i> ${conflictCount} Minor Conflict${conflictCount > 1 ? 's' : ''}`;

    // Update research
    const researchCount = data.researchRecommendations?.length || 3;
    document.getElementById('live-research').textContent = researchCount;
    document.getElementById('research-pill').innerHTML =
      `<i class="fas fa-book"></i> ${researchCount} Research Suggestion${researchCount > 1 ? 's' : ''}`;

    // Animate updates
    this.animateUpdates();
  }

  async testNuanceMerge() {
    this.showNotification('Testing nuance merge detection...', 'info');

    try {
      const response = await fetch(`${this.baseURL}/api/test-nuance-merge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      const result = await response.json();

      if (result.status === 'success') {
        this.displayNuanceResult(result);
        this.showNotification('Nuance merge test completed successfully', 'success');
      }
    } catch (error) {
      console.error('Nuance test error:', error);
      this.displayOfflineNuanceResult();
      this.showNotification('Using offline nuance demonstration', 'warning');
    }
  }

  displayNuanceResult(result) {
    const nuanceDemo = document.getElementById('nuance-demo');
    if (!nuanceDemo) return;

    nuanceDemo.innerHTML = `
      <div class="nuance-example">
        <div class="nuance-before">
          <label>Original Text:</label>
          <p>"${result.original}"</p>
        </div>
        <div class="nuance-after">
          <label>AI-Enhanced Text:</label>
          <p>"${result.enhanced}"</p>
        </div>
        <div class="nuance-stats">
          <span class="nuance-stat">
            <i class="fas fa-shield-check"></i>
            Information Loss: <strong>${result.information_loss_risk}</strong>
          </span>
          <span class="nuance-stat">
            <i class="fas fa-percentage"></i>
            Confidence: <strong>${Math.round(result.confidence_score * 100)}%</strong>
          </span>
          <span class="nuance-stat">
            <i class="fas fa-brain"></i>
            Type: <strong>${result.nuance_type}</strong>
          </span>
        </div>
      </div>
    `;

    // Animate the update
    nuanceDemo.classList.add('slide-up');
  }

  displayOfflineNuanceResult() {
    const result = {
      original: "The patient shows symptoms of cardiac distress.",
      enhanced: "The patient demonstrates clinical manifestations consistent with acute coronary syndrome requiring immediate evaluation.",
      information_loss_risk: "minimal",
      confidence_score: 0.87,
      nuance_type: "enhancement"
    };

    this.displayNuanceResult(result);
  }

  startLiveMetrics() {
    this.metricsUpdateInterval = setInterval(() => {
      this.updateLiveMetrics();
    }, 3000);
  }

  async updateLiveMetrics() {
    try {
      const response = await fetch(`${this.baseURL}/api/live-metrics`);
      const metrics = await response.json();

      this.animateMetricUpdates(metrics);
    } catch (error) {
      // Fallback to simulated metrics
      const metrics = {
        quality_score: 0.75 + Math.random() * 0.2,
        conflicts_detected: Math.floor(Math.random() * 3),
        research_opportunities: 2 + Math.floor(Math.random() * 4),
        workflow_efficiency: 0.70 + Math.random() * 0.25
      };

      this.animateMetricUpdates(metrics);
    }
  }

  animateMetricUpdates(metrics) {
    const updates = [
      { id: 'live-quality', value: `${Math.round(metrics.quality_score * 100)}%` },
      { id: 'live-conflicts', value: metrics.conflicts_detected },
      { id: 'live-research', value: metrics.research_opportunities },
      { id: 'live-workflow', value: `${Math.round(metrics.workflow_efficiency * 100)}%` }
    ];

    updates.forEach(update => {
      const element = document.getElementById(update.id);
      if (element) {
        element.style.transform = 'scale(1.1)';
        element.style.color = 'var(--primary-blue)';

        setTimeout(() => {
          element.textContent = update.value;
          element.style.transform = 'scale(1)';
          element.style.color = '';
        }, 150);
      }
    });
  }

  startTypingSuggestions() {
    this.showSuggestions();
    this.cycleSuggestions();
  }

  stopTypingSuggestions() {
    clearInterval(this.suggestionInterval);
  }

  showSuggestions() {
    const panel = document.getElementById('suggestions-panel');
    if (panel) {
      panel.style.display = 'block';
      this.updateSuggestionDisplay();
    }
  }

  cycleSuggestions() {
    this.suggestionInterval = setInterval(() => {
      this.currentSuggestionIndex = (this.currentSuggestionIndex + 1) % this.suggestions.length;
      this.updateSuggestionDisplay();
    }, 4000);
  }

  updateSuggestionDisplay() {
    const suggestionList = document.getElementById('suggestion-list');
    if (!suggestionList) return;

    const suggestion = this.suggestions[this.currentSuggestionIndex];

    suggestionList.innerHTML = `
      <div class="suggestion-item">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
          <i class="${suggestion.icon}" style="color: var(--primary-blue);"></i>
          <h5 style="margin: 0; font-weight: 600;">${suggestion.title}</h5>
          <span style="margin-left: auto; background: var(--medical-green); color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">
            ${Math.round(suggestion.confidence * 100)}%
          </span>
        </div>
        <p style="margin: 0; color: var(--gray-600); line-height: 1.5;">${suggestion.description}</p>
      </div>
    `;
  }

  simulateRealTimeAnalysis() {
    // Show brief analysis indicator
    const overlay = document.getElementById('editor-overlay');
    if (overlay) {
      overlay.classList.add('active');
      setTimeout(() => {
        overlay.classList.remove('active');
      }, 800);
    }
  }

  showAnalysisOverlay() {
    const overlay = document.getElementById('editor-overlay');
    if (overlay) {
      overlay.classList.add('active');
    }
  }

  hideAnalysisOverlay() {
    const overlay = document.getElementById('editor-overlay');
    if (overlay) {
      setTimeout(() => {
        overlay.classList.remove('active');
      }, 1500);
    }
  }

  demonstrateNuance() {
    this.testNuanceMerge();
    this.scrollToElement('.intelligence-sidebar');
    this.highlightElement(document.querySelector('[data-feature="nuance"]'));
  }

  demonstrateQuality() {
    this.analyzeContent();
    this.scrollToElement('.quality-bar');
    this.highlightElement(document.querySelector('[data-feature="quality"]'));
  }

  demonstrateResearch() {
    this.showNotification('Research recommendations updated based on content analysis', 'info');
    this.scrollToElement('.research-list');
    this.highlightElement(document.querySelector('[data-feature="research"]'));
  }

  highlightElement(element) {
    if (!element) return;

    element.style.transform = 'scale(1.05)';
    element.style.boxShadow = '0 0 20px rgba(59, 130, 246, 0.3)';
    element.style.transition = 'all 0.3s ease';

    setTimeout(() => {
      element.style.transform = '';
      element.style.boxShadow = '';
    }, 2000);
  }

  scrollToElement(selector) {
    const element = document.querySelector(selector);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  animateUpdates() {
    const elements = document.querySelectorAll('.stat-value, .quality-value, .progress-fill');
    elements.forEach(el => {
      el.style.transform = 'scale(1.1)';
      setTimeout(() => {
        el.style.transform = 'scale(1)';
      }, 200);
    });
  }

  resetDemo() {
    this.showNotification('Resetting demo to initial state...', 'info');

    setTimeout(() => {
      location.reload();
    }, 1000);
  }

  toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => {
        this.showNotification('Fullscreen not supported in this browser', 'warning');
      });
    } else {
      document.exitFullscreen();
    }
  }

  showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
      <div style="display: flex; align-items: center; gap: 12px;">
        <i class="fas fa-${this.getNotificationIcon(type)}"></i>
        <span>${message}</span>
      </div>
    `;

    // Style the notification
    Object.assign(notification.style, {
      position: 'fixed',
      top: '24px',
      right: '24px',
      background: this.getNotificationColor(type),
      color: 'white',
      padding: '16px 24px',
      borderRadius: '12px',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
      zIndex: '1001',
      transform: 'translateX(400px)',
      transition: 'all 0.3s ease',
      maxWidth: '400px',
      fontSize: '14px',
      fontWeight: '500'
    });

    document.body.appendChild(notification);

    // Animate in
    setTimeout(() => {
      notification.style.transform = 'translateX(0)';
    }, 100);

    // Auto remove
    setTimeout(() => {
      notification.style.transform = 'translateX(400px)';
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      }, 300);
    }, 4000);
  }

  getNotificationIcon(type) {
    const icons = {
      success: 'check-circle',
      error: 'exclamation-circle',
      warning: 'exclamation-triangle',
      info: 'info-circle'
    };
    return icons[type] || 'info-circle';
  }

  getNotificationColor(type) {
    const colors = {
      success: 'var(--medical-green)',
      error: 'var(--error)',
      warning: 'var(--warning)',
      info: 'var(--primary-blue)'
    };
    return colors[type] || 'var(--primary-blue)';
  }

  // API Keys Management
  async loadApiKeys() {
    try {
      const response = await fetch(`${this.baseURL}/api/config/api-keys`);
      const data = await response.json();

      // Update UI with current API key status
      Object.keys(data).forEach(service => {
        const serviceElement = document.querySelector(`[data-service="${service}"]`);
        if (serviceElement) {
          const statusBadge = serviceElement.parentElement.querySelector('.status-badge');
          if (data[service].configured) {
            statusBadge.textContent = 'Active';
            statusBadge.className = 'status-badge active';
            serviceElement.placeholder = data[service].key;
          } else {
            statusBadge.textContent = 'Inactive';
            statusBadge.className = 'status-badge inactive';
          }
        }
      });
    } catch (error) {
      console.error('Failed to load API keys:', error);
    }
  }

  async updateApiKey(service, key) {
    try {
      const response = await fetch(`${this.baseURL}/api/config/api-keys`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ service, key })
      });

      const result = await response.json();
      if (result.success) {
        this.showNotification(`${service} API key updated successfully`, 'success');
        this.loadApiKeys(); // Refresh status
      }
    } catch (error) {
      this.showNotification(`Failed to update ${service} API key`, 'error');
    }
  }

  // PDF Library Management
  async loadPDFLibrary() {
    try {
      const response = await fetch(`${this.baseURL}/api/pdf-library`);
      const data = await response.json();

      // Update library count
      const libraryTitle = document.querySelector('.panel h4');
      if (libraryTitle && libraryTitle.textContent.includes('PDF Library')) {
        libraryTitle.innerHTML = `<i class="fas fa-file-pdf"></i> PDF Library (${data.total_papers.toLocaleString()} papers)`;
      }

      // Update recent papers
      const papersContainer = document.querySelector('.recent-papers');
      if (papersContainer && data.recent_uploads) {
        papersContainer.innerHTML = data.recent_uploads.map(paper => `
          <div class="paper-item">
            <h5>${paper.title}</h5>
            <p>${paper.authors.join(', ')} - ${paper.journal}</p>
            <span class="relevance">${Math.round(paper.relevance * 100)}% relevant</span>
          </div>
        `).join('');
      }
    } catch (error) {
      console.error('Failed to load PDF library:', error);
    }
  }

  async searchPDFLibrary(query) {
    // Simulate search delay
    await new Promise(resolve => setTimeout(resolve, 500));

    this.showNotification(`Searching ${query} in 15,847 papers...`, 'info');

    // In a real implementation, this would search the actual library
    setTimeout(() => {
      this.showNotification(`Found 23 papers matching "${query}"`, 'success');
    }, 1500);
  }

  // Live Chapters Management
  async loadLiveChapters() {
    try {
      const response = await fetch(`${this.baseURL}/api/live-chapters`);
      const data = await response.json();

      // Update active sessions count
      const chaptersTitle = document.querySelector('.panel h4');
      if (chaptersTitle && chaptersTitle.textContent.includes('Live Chapters')) {
        chaptersTitle.innerHTML = `<i class="fas fa-users"></i> Live Chapters (${data.active_sessions} active)`;
      }

      // Update recent activity
      const activityContainer = document.querySelector('.recent-activity');
      if (activityContainer && data.recent_activity) {
        activityContainer.innerHTML = data.recent_activity.map(activity => `
          <p><strong>${activity.user}</strong> ${activity.action} <span>${activity.timestamp}</span></p>
        `).join('');
      }
    } catch (error) {
      console.error('Failed to load live chapters:', error);
    }
  }

  // System Integrations
  async loadIntegrations() {
    try {
      const response = await fetch(`${this.baseURL}/api/integrations`);
      const data = await response.json();

      // Update integration status indicators
      const integrationsContainer = document.querySelector('.integrations');
      if (integrationsContainer) {
        // Update EMR systems status
        if (data.emr_systems) {
          Object.keys(data.emr_systems).forEach(system => {
            const systemElement = integrationsContainer.querySelector(`[data-system="${system}"]`);
            if (systemElement) {
              const badge = systemElement.querySelector('.status-badge');
              if (data.emr_systems[system].connected) {
                badge.textContent = 'Connected';
                badge.className = 'status-badge active';
              } else {
                badge.textContent = 'Disconnected';
                badge.className = 'status-badge inactive';
              }
            }
          });
        }

        // Update AI services usage
        if (data.ai_services) {
          Object.keys(data.ai_services).forEach(service => {
            const serviceElement = integrationsContainer.querySelector(`[data-ai-service="${service}"]`);
            if (serviceElement) {
              const badge = serviceElement.querySelector('.usage-badge');
              if (badge) {
                badge.textContent = `${data.ai_services[service].calls_today} calls today`;
              }
            }
          });
        }
      }
    } catch (error) {
      console.error('Failed to load integrations:', error);
    }
  }

  // Initialize all new features
  initializeNewFeatures() {
    // Load all data
    this.loadApiKeys();
    this.loadPDFLibrary();
    this.loadLiveChapters();
    this.loadIntegrations();

    // Set up periodic updates
    setInterval(() => {
      this.loadLiveChapters();
      this.loadIntegrations();
    }, 30000); // Update every 30 seconds

    // Set up search functionality
    const librarySearch = document.querySelector('.library-search');
    if (librarySearch) {
      librarySearch.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          this.searchPDFLibrary(e.target.value);
        }
      });
    }
  }
}

// Initialize demo when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.kooDemo = new KOODemo();
  // Initialize new features after a short delay
  setTimeout(() => {
    window.kooDemo.initializeNewFeatures();
  }, 1000);
});

// Global functions for HTML onclick handlers
function updateApiKeys() {
  const inputs = document.querySelectorAll('.api-input');
  inputs.forEach(input => {
    if (input.value.trim()) {
      const service = input.getAttribute('data-service');
      window.kooDemo.updateApiKey(service, input.value.trim());
      input.value = ''; // Clear after update
    }
  });
}

function uploadPDF() {
  window.kooDemo.showNotification('PDF upload feature would open file dialog here', 'info');
  // In real implementation: file input dialog
}

function searchPDFLibrary() {
  const searchInput = document.querySelector('.library-search');
  if (searchInput && searchInput.value.trim()) {
    window.kooDemo.searchPDFLibrary(searchInput.value.trim());
  }
}

function joinLiveChapter(chapterId) {
  window.kooDemo.showNotification(`Joining live chapter session: ${chapterId}`, 'info');
  // In real implementation: WebRTC connection setup
}

function connectIntegration(system) {
  window.kooDemo.showNotification(`Connecting to ${system}...`, 'info');
  // In real implementation: OAuth or API setup
}

function exportChapter(format) {
  const title = document.getElementById('chapter-title').value;
  const content = document.getElementById('content-editor').value;

  window.kooDemo.showNotification(`Exporting "${title}" as ${format.toUpperCase()}...`, 'info');

  // Simulate export process
  setTimeout(() => {
    window.kooDemo.showNotification(`Chapter exported successfully as ${format.toUpperCase()}`, 'success');
  }, 2000);

  // In real implementation: generate actual PDF/Word document
}

function shareChapter() {
  const title = document.getElementById('chapter-title').value;

  window.kooDemo.showNotification(`Generating share link for "${title}"...`, 'info');

  setTimeout(() => {
    const shareUrl = `https://koo-platform.com/share/ch_${Math.random().toString(36).substr(2, 9)}`;
    navigator.clipboard.writeText(shareUrl).then(() => {
      window.kooDemo.showNotification(`Share link copied to clipboard!`, 'success');
    }).catch(() => {
      window.kooDemo.showNotification(`Share link: ${shareUrl}`, 'info');
    });
  }, 1500);
}

// Enhanced PDF Library Functions
async function searchLibrary() {
  const query = document.getElementById('library-search').value.trim();
  const source = document.getElementById('search-source').value;

  if (!query) {
    window.kooDemo.showNotification('Please enter a search term', 'warning');
    return;
  }

  window.kooDemo.showNotification(`Searching ${source} for "${query}"...`, 'info');

  try {
    const response = await fetch(`${window.kooDemo.baseURL}/api/pdf-library/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, source })
    });

    const data = await response.json();

    // Update search results
    const resultsContainer = document.getElementById('search-results');
    if (data.results && data.results.length > 0) {
      resultsContainer.innerHTML = `
        <div class="search-info">
          <p><strong>${data.total_results} results</strong> from ${source} in ${data.search_time}ms</p>
        </div>
        <div class="recent-papers">
          ${data.results.map(paper => `
            <div class="paper-item referenceable" data-id="${paper.id}">
              <h5>${paper.title}</h5>
              <p>${paper.authors ? paper.authors.join(', ') : 'Various Authors'} - ${paper.journal || paper.source}</p>
              ${paper.pmid ? `<p><strong>PMID:</strong> ${paper.pmid}</p>` : ''}
              ${paper.citations ? `<p><strong>Citations:</strong> ${paper.citations}</p>` : ''}
              <span class="relevance">${Math.round((paper.relevance || 0.8) * 100)}% relevant</span>
              <div class="paper-actions">
                <button class="btn btn-xs" onclick="addReference('${paper.id}')">
                  <i class="fas fa-plus"></i> Add Ref
                </button>
                <button class="btn btn-xs" onclick="viewAbstract('${paper.id}')">
                  <i class="fas fa-eye"></i> Abstract
                </button>
              </div>
            </div>
          `).join('')}
        </div>
      `;

      window.kooDemo.showNotification(`Found ${data.total_results} papers in ${source}`, 'success');
    } else {
      resultsContainer.innerHTML = '<p>No results found. Try different keywords or source.</p>';
      window.kooDemo.showNotification('No results found', 'warning');
    }
  } catch (error) {
    window.kooDemo.showNotification('Search failed. Please try again.', 'error');
  }
}

async function addReference(referenceId) {
  const citationStyle = document.getElementById('citation-style').value;

  window.kooDemo.showNotification('Adding reference to chapter...', 'info');

  try {
    const response = await fetch(`${window.kooDemo.baseURL}/api/pdf-library/reference`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        reference_id: referenceId,
        citation_style: citationStyle,
        position: 'end'
      })
    });

    const data = await response.json();

    if (data.success) {
      // Add reference to content editor
      const editor = document.getElementById('content-editor');
      const currentContent = editor.value;
      const referenceText = `\n\n**Reference ${data.citation_number}:** ${data.formatted_citation}`;
      editor.value = currentContent + referenceText;

      // Visual feedback
      const paperElement = document.querySelector(`[data-id="${referenceId}"]`);
      if (paperElement) {
        paperElement.classList.add('reference-inserted');
        setTimeout(() => {
          paperElement.classList.remove('reference-inserted');
        }, 2000);
      }

      window.kooDemo.showNotification(`Reference added successfully (Citation ${data.citation_number})`, 'success');
    }
  } catch (error) {
    window.kooDemo.showNotification('Failed to add reference', 'error');
  }
}

function viewAbstract(referenceId) {
  // Create modal or expand view for abstract
  window.kooDemo.showNotification('Abstract view would open here', 'info');

  // In real implementation: show modal with full abstract, key findings, etc.
  console.log(`Viewing abstract for ${referenceId}`);
}

// Google Gemini AI Analysis Functions
async function runGeminiAnalysis() {
  const content = document.getElementById('content-editor').value;

  if (!content.trim()) {
    window.kooDemo.showNotification('Please add content to analyze', 'warning');
    return;
  }

  window.kooDemo.showNotification('Running Gemini Pro analysis...', 'info');

  try {
    const response = await fetch(`${window.kooDemo.baseURL}/api/ai/gemini-analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: content,
        type: 'comprehensive'
      })
    });

    const data = await response.json();

    if (data.ai_model === 'Gemini Pro') {
      // Show comprehensive analysis results
      showGeminiResults(data);
      window.kooDemo.showNotification(
        `Gemini analysis complete! Quality: ${Math.round(data.overall_assessment.quality_score * 100)}%`,
        'success'
      );
    }
  } catch (error) {
    window.kooDemo.showNotification('Gemini analysis failed. Please try again.', 'error');
  }
}

function showGeminiResults(data) {
  // Create or update results panel
  let resultsPanel = document.getElementById('gemini-results');

  if (!resultsPanel) {
    resultsPanel = document.createElement('div');
    resultsPanel.id = 'gemini-results';
    resultsPanel.className = 'panel gemini-panel';

    // Insert after suggestions panel
    const suggestionsPanel = document.getElementById('suggestions-panel');
    suggestionsPanel.parentNode.insertBefore(resultsPanel, suggestionsPanel.nextSibling);
  }

  resultsPanel.innerHTML = `
    <h4><i class="fas fa-google"></i> Gemini Pro Analysis</h4>
    <div class="gemini-results-content">
      <div class="analysis-overview">
        <h5>Overall Assessment</h5>
        <div class="score-grid">
          <div class="score-item">
            <span class="score-label">Quality</span>
            <span class="score-value">${Math.round(data.overall_assessment.quality_score * 100)}%</span>
          </div>
          <div class="score-item">
            <span class="score-label">Medical Accuracy</span>
            <span class="score-value">${Math.round(data.overall_assessment.medical_accuracy * 100)}%</span>
          </div>
          <div class="score-item">
            <span class="score-label">Clarity</span>
            <span class="score-value">${Math.round(data.overall_assessment.clarity * 100)}%</span>
          </div>
          <div class="score-item">
            <span class="score-label">Completeness</span>
            <span class="score-value">${Math.round(data.overall_assessment.completeness * 100)}%</span>
          </div>
        </div>
      </div>

      <div class="content-insights">
        <h5>Content Insights</h5>
        <p><strong>Word Count:</strong> ${data.content_insights.word_count}</p>
        <p><strong>Reading Time:</strong> ${data.content_insights.reading_time}</p>
        <p><strong>Complexity:</strong> ${data.content_insights.complexity_level}</p>
      </div>

      <div class="ai-recommendations">
        <h5>AI Recommendations</h5>
        <ul>
          ${data.ai_recommendations.map(rec => `<li>${rec}</li>`).join('')}
        </ul>
      </div>

      <div class="next-actions">
        <h5>Suggested Next Actions</h5>
        <ul>
          ${data.next_actions.map(action => `<li>${action}</li>`).join('')}
        </ul>
      </div>

      <div class="analysis-meta">
        <p><strong>Gemini Confidence:</strong> ${Math.round(data.gemini_confidence * 100)}%</p>
        <p><strong>Processing Time:</strong> ${data.processing_time}ms</p>
        <button class="btn btn-xs btn-accent" onclick="runGeminiEnhancement()">
          <i class="fas fa-magic"></i> Get Enhancement Suggestions
        </button>
      </div>
    </div>
  `;

  // Scroll to results
  resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function runGeminiEnhancement() {
  const content = document.getElementById('content-editor').value;

  window.kooDemo.showNotification('Generating Gemini enhancement suggestions...', 'info');

  try {
    const response = await fetch(`${window.kooDemo.baseURL}/api/ai/gemini-enhancement`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: content,
        type: 'medical_precision'
      })
    });

    const data = await response.json();

    if (data.ai_model === 'Gemini Pro') {
      showGeminiEnhancements(data);
      window.kooDemo.showNotification(
        `Enhancement suggestions ready! Potential improvement: ${data.medical_accuracy_improvement}`,
        'success'
      );
    }
  } catch (error) {
    window.kooDemo.showNotification('Enhancement generation failed', 'error');
  }
}

function showGeminiEnhancements(data) {
  // Update existing panel or create new one
  const resultsPanel = document.getElementById('gemini-results');

  const enhancementsHtml = `
    <div class="gemini-enhancements">
      <h5><i class="fas fa-lightbulb"></i> Enhancement Suggestions</h5>
      <div class="enhancement-overview">
        <p><strong>Overall Enhancement Score:</strong> ${Math.round(data.overall_enhancement_score * 100)}%</p>
        <p><strong>Medical Accuracy Improvement:</strong> ${data.medical_accuracy_improvement}</p>
        <p><strong>Evidence Integration:</strong> ${data.evidence_integration}</p>
      </div>

      <div class="suggestions-list">
        ${data.suggestions.map((suggestion, index) => `
          <div class="enhancement-suggestion">
            <h6>${suggestion.section}</h6>
            <div class="suggestion-comparison">
              <div class="current-text">
                <label>Current:</label>
                <p>"${suggestion.current}"</p>
              </div>
              <div class="enhanced-text">
                <label>Enhanced:</label>
                <p>"${suggestion.enhanced}"</p>
              </div>
            </div>
            <div class="suggestion-meta">
              <span class="improvement-type">${suggestion.improvement_type}</span>
              <span class="confidence">Confidence: ${Math.round(suggestion.confidence * 100)}%</span>
              <button class="btn btn-xs btn-primary" onclick="applyEnhancement(${index})">
                <i class="fas fa-check"></i> Apply
              </button>
            </div>
          </div>
        `).join('')}
      </div>
    </div>
  `;

  resultsPanel.innerHTML += enhancementsHtml;
}

function applyEnhancement(index) {
  window.kooDemo.showNotification(`Enhancement ${index + 1} would be applied to content`, 'info');
  // In real implementation: actually modify the content editor with the enhancement
}

// Service Worker for offline functionality (optional)
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(err => {
    console.log('Service worker registration failed');
  });
}