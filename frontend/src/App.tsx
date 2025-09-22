/**
 * Main Application Component for KOO Platform
 * Enhanced medical knowledge management with full AI intelligence integration
 */

import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box, CssBaseline } from '@mui/material';

// Layout components
import AppLayout from './components/layout/AppLayout';
import LoadingScreen from './components/common/LoadingScreen';

// Lazy-loaded components for code splitting
const IntelligentDashboard = lazy(() => import('./components/intelligent/IntelligentDashboard'));
const IntelligentChapterEditor = lazy(() => import('./components/intelligent/IntelligentChapterEditor'));
const IntelligentResearchAssistant = lazy(() => import('./components/intelligent/IntelligentResearchAssistant'));
const AdaptiveQualityAssessment = lazy(() => import('./components/intelligent/AdaptiveQualityAssessment'));
const WorkflowOptimizer = lazy(() => import('./components/intelligent/WorkflowOptimizer'));
const KnowledgeGraphVisualizer = lazy(() => import('./components/intelligent/KnowledgeGraphVisualizer'));
const OptimizedPDFProcessor = lazy(() => import('./components/intelligent/OptimizedPDFProcessor'));
const TextbookUploadManager = lazy(() => import('./components/intelligent/TextbookUploadManager'));

// Error boundary for route-level error handling
class RouteErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Route Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box
          display="flex"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
          minHeight="400px"
          p={4}
          textAlign="center"
        >
          <h2 style={{ color: '#d32f2f', marginBottom: '1rem' }}>
            Something went wrong
          </h2>
          <p style={{ color: '#666', marginBottom: '2rem' }}>
            This section of the application encountered an error.
          </p>
          <button
            onClick={() => this.setState({ hasError: false })}
            style={{
              background: '#1976d2',
              color: 'white',
              border: 'none',
              padding: '12px 24px',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: '600',
            }}
          >
            Try Again
          </button>
        </Box>
      );
    }

    return this.props.children;
  }
}

// Suspense wrapper for lazy-loaded components
const SuspenseWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <Suspense fallback={<LoadingScreen />}>
    <RouteErrorBoundary>
      {children}
    </RouteErrorBoundary>
  </Suspense>
);

/**
 * Main Application Component
 * Provides routing and layout structure for the entire KOO Platform
 */
const App: React.FC = () => {
  return (
    <Box className="App" sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <CssBaseline />

      <Routes>
        {/* Main application routes with layout */}
        <Route path="/" element={<AppLayout />}>
          {/* Dashboard - Main landing page */}
          <Route index element={
            <SuspenseWrapper>
              <IntelligentDashboard />
            </SuspenseWrapper>
          } />

          {/* Chapter Management */}
          <Route path="chapters" element={
            <SuspenseWrapper>
              <IntelligentChapterEditor />
            </SuspenseWrapper>
          } />
          <Route path="chapters/:chapterId" element={
            <SuspenseWrapper>
              <IntelligentChapterEditor />
            </SuspenseWrapper>
          } />

          {/* Research Tools */}
          <Route path="research" element={
            <SuspenseWrapper>
              <IntelligentResearchAssistant />
            </SuspenseWrapper>
          } />

          {/* Quality Assessment */}
          <Route path="quality" element={
            <SuspenseWrapper>
              <AdaptiveQualityAssessment />
            </SuspenseWrapper>
          } />

          {/* Workflow Optimization */}
          <Route path="workflow" element={
            <SuspenseWrapper>
              <WorkflowOptimizer />
            </SuspenseWrapper>
          } />

          {/* Knowledge Graph */}
          <Route path="knowledge-graph" element={
            <SuspenseWrapper>
              <KnowledgeGraphVisualizer />
            </SuspenseWrapper>
          } />

          {/* PDF Processing */}
          <Route path="pdf-processor" element={
            <SuspenseWrapper>
              <OptimizedPDFProcessor />
            </SuspenseWrapper>
          } />

          {/* Upload Management */}
          <Route path="upload" element={
            <SuspenseWrapper>
              <TextbookUploadManager />
            </SuspenseWrapper>
          } />

          {/* Dashboard alias routes */}
          <Route path="dashboard" element={<Navigate to="/" replace />} />
          <Route path="home" element={<Navigate to="/" replace />} />
        </Route>

        {/* Catch-all route for 404 */}
        <Route path="*" element={
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            minHeight="100vh"
            p={4}
            textAlign="center"
            bgcolor="#fafafa"
          >
            <h1 style={{ color: '#1976d2', marginBottom: '1rem', fontSize: '3rem' }}>
              404
            </h1>
            <h2 style={{ color: '#333', marginBottom: '1rem' }}>
              Page Not Found
            </h2>
            <p style={{ color: '#666', marginBottom: '2rem', maxWidth: '500px' }}>
              The page you're looking for doesn't exist. It might have been moved,
              deleted, or you entered the wrong URL.
            </p>
            <button
              onClick={() => window.location.href = '/'}
              style={{
                background: '#1976d2',
                color: 'white',
                border: 'none',
                padding: '12px 24px',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '16px',
                fontWeight: '600',
                textDecoration: 'none',
              }}
            >
              Go to Dashboard
            </button>
          </Box>
        } />
      </Routes>
    </Box>
  );
};

export default App;