/**
 * Enhanced API Service
 * Comprehensive API client for all intelligence modules and platform features
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

// Types
interface ApiResponse<T> {
  data: T;
  message?: string;
  success: boolean;
  timestamp: Date;
}

interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  totalPages: number;
}

interface ErrorResponse {
  error: string;
  message: string;
  statusCode: number;
  timestamp: Date;
}

// Enhanced API Configuration
class EnhancedApiService {
  private api: AxiosInstance;
  private baseURL: string;
  private retryAttempts: number = 3;
  private retryDelay: number = 1000;

  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

    this.api = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor with retry logic
    this.api.interceptors.response.use(
      (response) => response,
      async (error) => {
        const config = error.config;

        if (!config || !config.retry) {
          config.retry = 0;
        }

        if (config.retry < this.retryAttempts && this.shouldRetry(error)) {
          config.retry += 1;
          await this.delay(this.retryDelay * config.retry);
          return this.api.request(config);
        }

        return Promise.reject(error);
      }
    );
  }

  private shouldRetry(error: any): boolean {
    return (
      error.code === 'NETWORK_ERROR' ||
      error.code === 'TIMEOUT' ||
      (error.response && error.response.status >= 500)
    );
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // ============================================================================
  // AUTHENTICATION
  // ============================================================================

  async login(credentials: { email: string; password: string }) {
    const response = await this.api.post('/auth/login', credentials);
    return response.data;
  }

  async logout() {
    const response = await this.api.post('/auth/logout');
    localStorage.removeItem('auth_token');
    return response.data;
  }

  async refreshToken() {
    const response = await this.api.post('/auth/refresh');
    return response.data;
  }

  // ============================================================================
  // INTELLIGENT CHAPTERS
  // ============================================================================

  async createIntelligentChapter(data: {
    title: string;
    content: string;
    summary?: string;
    tags?: string[];
    specialty?: string;
    context?: Record<string, any>;
  }) {
    const response = await this.api.post('/api/v1/chapters/intelligent-create', data);
    return response.data;
  }

  async updateIntelligentChapter(chapterId: string, data: any) {
    const response = await this.api.put(`/api/v1/chapters/${chapterId}`, data);
    return response.data;
  }

  async analyzeChapterIntelligence(data: {
    content: string;
    title: string;
    specialty?: string;
    context?: Record<string, any>;
  }) {
    const response = await this.api.post('/api/v1/chapters/analyze', data);
    return response.data;
  }

  async getChapterIntelligenceDashboard(chapterId: string) {
    const response = await this.api.get(`/api/v1/chapters/intelligence-dashboard/${chapterId}`);
    return response.data;
  }

  async synthesizeChapters(data: {
    source_chapters: string[];
    synthesis_type: string;
    target_topic: string;
    context?: Record<string, any>;
  }) {
    const response = await this.api.post('/api/v1/chapters/synthesize', data);
    return response.data;
  }

  // ============================================================================
  // QUALITY ASSESSMENT
  // ============================================================================

  async assessContentQuality(data: {
    content: string;
    contentType: string;
    realTime?: boolean;
    context?: Record<string, any>;
  }) {
    const response = await this.api.post('/api/v1/quality/assess', data);
    return response.data;
  }

  async getContentQualityAssessment(contentId: string) {
    const response = await this.api.get(`/api/v1/quality/assessment/${contentId}`);
    return response.data;
  }

  async getQualityTrends(contentId?: string) {
    const url = contentId ? `/api/v1/quality/trends/${contentId}` : '/api/v1/quality/trends';
    const response = await this.api.get(url);
    return response.data;
  }

  async compareQuality(contentId1: string, contentId2: string) {
    const response = await this.api.get(`/api/v1/quality/compare/${contentId1}/${contentId2}`);
    return response.data;
  }

  async getAssessmentHistory(contentId: string) {
    const response = await this.api.get(`/api/v1/quality/history/${contentId}`);
    return response.data;
  }

  async trackQualityImprovement(contentId: string, improvement: { action: string; result: string }) {
    const response = await this.api.post(`/api/v1/quality/track-improvement/${contentId}`, improvement);
    return response.data;
  }

  async getQualityInsights(contentId: string) {
    const response = await this.api.get(`/api/v1/quality/insights/${contentId}`);
    return response.data;
  }

  // ============================================================================
  // RESEARCH ENGINE
  // ============================================================================

  async intelligentSearch(query: {
    query: string;
    domain: string;
    urgency: number;
    qualityThreshold: number;
    maxResults: number;
    sourcePreferences: string[];
    timeRange: string;
    includeGrayLiterature: boolean;
    contextualExpansion: boolean;
  }) {
    const response = await this.api.post('/api/v1/research/intelligent-search', query);
    return response.data;
  }

  async getSmartResearchSuggestions(query: string, filters: any) {
    const response = await this.api.post('/api/v1/research/smart-suggestions', { query, filters });
    return response.data;
  }

  async saveResearchResult(resultId: string) {
    const response = await this.api.post(`/api/v1/research/save/${resultId}`);
    return response.data;
  }

  async getResearchHistory() {
    const response = await this.api.get('/api/v1/research/history');
    return response.data;
  }

  async getDomainSuggestions() {
    const response = await this.api.get('/api/v1/research/domains');
    return response.data;
  }

  async getResearchTrends() {
    const response = await this.api.get('/api/v1/research/trends');
    return response.data;
  }

  async getNodeSuggestions(query: string) {
    const response = await this.api.get(`/api/v1/research/node-suggestions?q=${encodeURIComponent(query)}`);
    return response.data;
  }

  // ============================================================================
  // WORKFLOW OPTIMIZATION
  // ============================================================================

  async getWorkflowTasks() {
    const response = await this.api.get('/api/v1/workflow/tasks');
    return response.data;
  }

  async getOptimizedSchedule(params: {
    date: Date;
    optimizationLevel: number;
    preferences: Record<string, any>;
  }) {
    const response = await this.api.post('/api/v1/workflow/optimize-schedule', params);
    return response.data;
  }

  async startWorkSession(taskId: string) {
    const response = await this.api.post('/api/v1/workflow/start-session', { taskId });
    return response.data;
  }

  async endWorkSession(sessionData: {
    taskId: string;
    duration: number;
    completed: boolean;
  }) {
    const response = await this.api.post('/api/v1/workflow/end-session', sessionData);
    return response.data;
  }

  async getProductivityMetrics() {
    const response = await this.api.get('/api/v1/workflow/productivity-metrics');
    return response.data;
  }

  async getWorkflowRecommendations() {
    const response = await this.api.get('/api/v1/workflow/recommendations');
    return response.data;
  }

  async getSessionAnalytics() {
    const response = await this.api.get('/api/v1/workflow/session-analytics');
    return response.data;
  }

  async optimizeWorkflow(params: {
    optimizationLevel: number;
    smartBreaks: boolean;
    adaptiveScheduling: boolean;
  }) {
    const response = await this.api.post('/api/v1/workflow/optimize', params);
    return response.data;
  }

  // ============================================================================
  // KNOWLEDGE GRAPH
  // ============================================================================

  async getKnowledgeGraph(params: {
    search?: string;
    filters?: Record<string, any>;
  }) {
    const response = await this.api.post('/api/v1/knowledge-graph/query', params);
    return response.data;
  }

  async getGraphAnalytics() {
    const response = await this.api.get('/api/v1/knowledge-graph/analytics');
    return response.data;
  }

  async getGraphSchema() {
    const response = await this.api.get('/api/v1/knowledge-graph/schema');
    return response.data;
  }

  async getKnowledgeGraphInsights() {
    const response = await this.api.get('/api/v1/knowledge-graph/insights');
    return response.data;
  }

  async updateKnowledgeGraph(data: {
    nodes?: any[];
    edges?: any[];
    operation: 'add' | 'update' | 'delete';
  }) {
    const response = await this.api.post('/api/v1/knowledge-graph/update', data);
    return response.data;
  }

  // ============================================================================
  // DASHBOARD & ANALYTICS
  // ============================================================================

  async getDashboardMetrics() {
    const response = await this.api.get('/api/v1/dashboard/metrics');
    return response.data;
  }

  async getIntelligenceInsights() {
    const response = await this.api.get('/api/v1/dashboard/intelligence-insights');
    return response.data;
  }

  async getPredictiveAlerts() {
    const response = await this.api.get('/api/v1/dashboard/predictive-alerts');
    return response.data;
  }

  async getRecentActivity() {
    const response = await this.api.get('/api/v1/dashboard/recent-activity');
    return response.data;
  }

  // ============================================================================
  // CONFLICT DETECTION
  // ============================================================================

  async detectConflicts(content: {
    content_pieces: Array<{
      content: string;
      source_id: string;
      title?: string;
    }>;
    context?: Record<string, any>;
  }) {
    const response = await this.api.post('/api/v1/conflicts/detect', content);
    return response.data;
  }

  async getConflictTrends(specialty: string, timeRange: string) {
    const response = await this.api.get(`/api/v1/conflicts/trends?specialty=${specialty}&timeRange=${timeRange}`);
    return response.data;
  }

  async resolveConflict(conflictId: string, resolution: {
    resolution_strategy: string;
    preferred_source?: string;
    manual_resolution?: string;
  }) {
    const response = await this.api.post(`/api/v1/conflicts/resolve/${conflictId}`, resolution);
    return response.data;
  }

  // ============================================================================
  // PREDICTIVE INTELLIGENCE
  // ============================================================================

  async getPredictiveAnalysis(context: Record<string, any>) {
    const response = await this.api.post('/api/v1/predictive/analyze', context);
    return response.data;
  }

  async getNextQueryPredictions(currentContext: Record<string, any>) {
    const response = await this.api.post('/api/v1/predictive/next-queries', currentContext);
    return response.data;
  }

  async getWorkflowPredictions(userId: string, context: Record<string, any>) {
    const response = await this.api.post('/api/v1/predictive/workflow', { userId, context });
    return response.data;
  }

  // ============================================================================
  // PERFORMANCE & MONITORING
  // ============================================================================

  async getSystemPerformance() {
    const response = await this.api.get('/api/v1/performance/system');
    return response.data;
  }

  async getPerformanceTrends(timeRange: string) {
    const response = await this.api.get(`/api/v1/performance/trends?timeRange=${timeRange}`);
    return response.data;
  }

  async optimizePerformance(context: Record<string, any>) {
    const response = await this.api.post('/api/v1/performance/optimize', context);
    return response.data;
  }

  // ============================================================================
  // USER MANAGEMENT
  // ============================================================================

  async getUserProfile() {
    const response = await this.api.get('/api/v1/users/profile');
    return response.data;
  }

  async updateUserProfile(data: Record<string, any>) {
    const response = await this.api.put('/api/v1/users/profile', data);
    return response.data;
  }

  async getUserPreferences() {
    const response = await this.api.get('/api/v1/users/preferences');
    return response.data;
  }

  async updateUserPreferences(preferences: Record<string, any>) {
    const response = await this.api.put('/api/v1/users/preferences', preferences);
    return response.data;
  }

  // ============================================================================
  // CONTEXTUAL INTELLIGENCE
  // ============================================================================

  async updateContext(context: Record<string, any>) {
    const response = await this.api.post('/api/v1/context/update', context);
    return response.data;
  }

  async getContextualSuggestions(query: string, context: Record<string, any>) {
    const response = await this.api.post('/api/v1/context/suggestions', { query, context });
    return response.data;
  }

  async getContextHistory() {
    const response = await this.api.get('/api/v1/context/history');
    return response.data;
  }

  // ============================================================================
  // SYNTHESIS ENGINE
  // ============================================================================

  async createSynthesis(data: {
    sources: Array<{
      source_id: string;
      content: string;
      evidence_level: string;
      confidence_score: number;
    }>;
    synthesis_type: string;
    target_topic: string;
    context?: Record<string, any>;
  }) {
    const response = await this.api.post('/api/v1/synthesis/create', data);
    return response.data;
  }

  async getSynthesisHistory() {
    const response = await this.api.get('/api/v1/synthesis/history');
    return response.data;
  }

  async updateSynthesis(synthesisId: string, newSources: any[]) {
    const response = await this.api.post(`/api/v1/synthesis/update/${synthesisId}`, { newSources });
    return response.data;
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  async uploadFile(file: File, purpose: string) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('purpose', purpose);

    const response = await this.api.post('/api/v1/files/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async exportData(type: string, filters?: Record<string, any>) {
    const response = await this.api.post('/api/v1/export', { type, filters }, {
      responseType: 'blob',
    });
    return response.data;
  }

  async getSystemHealth() {
    const response = await this.api.get('/api/v1/health');
    return response.data;
  }

  async getVersion() {
    const response = await this.api.get('/api/v1/version');
    return response.data;
  }

  // ============================================================================
  // BATCH OPERATIONS
  // ============================================================================

  async batchRequest(requests: Array<{
    method: 'GET' | 'POST' | 'PUT' | 'DELETE';
    url: string;
    data?: any;
  }>) {
    const response = await this.api.post('/api/v1/batch', { requests });
    return response.data;
  }

  // ============================================================================
  // WEBSOCKET CONNECTIONS (for real-time features)
  // ============================================================================

  createWebSocketConnection(endpoint: string, handlers: {
    onMessage?: (data: any) => void;
    onError?: (error: any) => void;
    onClose?: () => void;
  }) {
    const wsUrl = this.baseURL.replace('http', 'ws') + endpoint;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handlers.onMessage?.(data);
    };

    ws.onerror = (error) => {
      handlers.onError?.(error);
    };

    ws.onclose = () => {
      handlers.onClose?.();
    };

    return ws;
  }

  // ============================================================================
  // CACHING UTILITIES
  // ============================================================================

  private cache = new Map<string, { data: any; timestamp: number; ttl: number }>();

  private getCacheKey(url: string, params?: any): string {
    return `${url}_${JSON.stringify(params || {})}`;
  }

  private isValidCache(cacheEntry: any): boolean {
    return Date.now() - cacheEntry.timestamp < cacheEntry.ttl;
  }

  async getCachedRequest<T>(
    url: string,
    params?: any,
    ttl: number = 300000 // 5 minutes default
  ): Promise<T> {
    const cacheKey = this.getCacheKey(url, params);
    const cached = this.cache.get(cacheKey);

    if (cached && this.isValidCache(cached)) {
      return cached.data;
    }

    const response = await this.api.get(url, { params });
    this.cache.set(cacheKey, {
      data: response.data,
      timestamp: Date.now(),
      ttl
    });

    return response.data;
  }

  clearCache() {
    this.cache.clear();
  }
}

// Create and export singleton instance
const apiService = new EnhancedApiService();
export default apiService;