/**
 * Adaptive Quality Assessment Component
 * Real-time content quality analysis with learning capabilities
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box, Paper, Typography, Grid, Card, CardContent,
  LinearProgress, Chip, IconButton, Tabs, Tab,
  List, ListItem, ListItemText, ListItemIcon,
  Alert, Button, Dialog, DialogTitle, DialogContent,
  Accordion, AccordionSummary, AccordionDetails,
  CircularProgress, Divider, Tooltip, Badge,
  Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, FormControl, InputLabel,
  Select, MenuItem, Switch, FormControlLabel
} from '@mui/material';
import {
  Assessment as QualityIcon,
  TrendingUp as ImprovementIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Psychology as AIIcon,
  Timeline as TrendIcon,
  Lightbulb as SuggestionIcon,
  Speed as PerformanceIcon,
  Star as ExcellenceIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Compare as CompareIcon,
  History as HistoryIcon,
  Analytics as AnalyticsIcon
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useDebounce } from 'use-debounce';
import { format, formatDistanceToNow } from 'date-fns';

// Services
import apiService from '../../services/api';

// Types
interface QualityDimension {
  name: string;
  score: number;
  weight: number;
  description: string;
  suggestions: string[];
  trend: 'improving' | 'stable' | 'declining';
}

interface QualityAssessment {
  overallScore: number;
  confidence: number;
  dimensions: Record<string, QualityDimension>;
  improvementSuggestions: string[];
  strengths: string[];
  weaknesses: string[];
  predictedLongevity: number;
  comparativeRanking: number;
  evidenceGaps: string[];
  bias_indicators: string[];
  factual_accuracy: number;
  clinical_relevance: number;
  currency: number;
  assessment_timestamp: Date;
}

interface QualityComparison {
  baseline: QualityAssessment;
  current: QualityAssessment;
  improvements: string[];
  regressions: string[];
  netChange: number;
}

interface QualityTrend {
  date: Date;
  score: number;
  dimension: string;
  note?: string;
}

interface AdaptiveQualityAssessmentProps {
  contentId?: string;
  content?: string;
  contentType?: 'medical_fact' | 'research_summary' | 'clinical_guideline' | 'case_study';
  realTimeMode?: boolean;
  onQualityUpdate?: (assessment: QualityAssessment) => void;
}

const AdaptiveQualityAssessment: React.FC<AdaptiveQualityAssessmentProps> = ({
  contentId,
  content = '',
  contentType = 'medical_fact',
  realTimeMode = true,
  onQualityUpdate
}) => {
  // State
  const [selectedTab, setSelectedTab] = useState(0);
  const [showComparison, setShowComparison] = useState(false);
  const [comparisonTarget, setComparisonTarget] = useState<string>('');
  const [adaptiveLearning, setAdaptiveLearning] = useState(true);
  const [qualityThreshold, setQualityThreshold] = useState(0.7);
  const [selectedDimension, setSelectedDimension] = useState<string>('all');

  // Debounced content for real-time analysis
  const [debouncedContent] = useDebounce(content, 2000);

  const queryClient = useQueryClient();

  // Quality assessment query
  const {
    data: qualityAssessment,
    isLoading: isAssessing,
    error: assessmentError
  } = useQuery({
    queryKey: ['quality-assessment', contentId, debouncedContent, contentType],
    queryFn: async () => {
      if (contentId) {
        return apiService.getContentQualityAssessment(contentId);
      } else if (debouncedContent && debouncedContent.length > 50) {
        return apiService.assessContentQuality({
          content: debouncedContent,
          contentType,
          realTime: realTimeMode
        });
      }
      return null;
    },
    enabled: (contentId || debouncedContent.length > 50) && realTimeMode,
    refetchInterval: realTimeMode ? 30000 : false, // Refresh every 30 seconds in real-time mode
  });

  // Quality trends query
  const { data: qualityTrends } = useQuery({
    queryKey: ['quality-trends', contentId],
    queryFn: () => apiService.getQualityTrends(contentId),
    enabled: !!contentId,
  });

  // Quality comparison query
  const { data: qualityComparison } = useQuery({
    queryKey: ['quality-comparison', contentId, comparisonTarget],
    queryFn: () => apiService.compareQuality(contentId, comparisonTarget),
    enabled: !!contentId && !!comparisonTarget,
  });

  // Historical assessments query
  const { data: assessmentHistory } = useQuery({
    queryKey: ['assessment-history', contentId],
    queryFn: () => apiService.getAssessmentHistory(contentId),
    enabled: !!contentId,
  });

  // Improvement tracking mutation
  const improvementMutation = useMutation({
    mutationFn: (improvement: { action: string; result: string }) =>
      apiService.trackQualityImprovement(contentId, improvement),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['quality-assessment'] });
    },
  });

  // Update callback effect
  useEffect(() => {
    if (qualityAssessment && onQualityUpdate) {
      onQualityUpdate(qualityAssessment);
    }
  }, [qualityAssessment, onQualityUpdate]);

  // Calculate quality color
  const getQualityColor = useCallback((score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  }, []);

  // Get trend icon
  const getTrendIcon = useCallback((trend: string) => {
    switch (trend) {
      case 'improving': return <ImprovementIcon color="success" />;
      case 'declining': return <ErrorIcon color="error" />;
      default: return <InfoIcon color="info" />;
    }
  }, []);

  // Format quality score
  const formatScore = useCallback((score: number) => {
    return `${(score * 100).toFixed(1)}%`;
  }, []);

  // Render overall quality summary
  const renderQualitySummary = () => {
    if (!qualityAssessment) return null;

    return (
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h6" gutterBottom>
                Overall Quality
              </Typography>
              <Box sx={{ position: 'relative', display: 'inline-flex', mb: 2 }}>
                <CircularProgress
                  variant="determinate"
                  value={qualityAssessment.overallScore * 100}
                  size={80}
                  thickness={6}
                  color={getQualityColor(qualityAssessment.overallScore)}
                />
                <Box
                  sx={{
                    top: 0,
                    left: 0,
                    bottom: 0,
                    right: 0,
                    position: 'absolute',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Typography variant="h6" component="div" color="text.secondary">
                    {formatScore(qualityAssessment.overallScore)}
                  </Typography>
                </Box>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Confidence: {formatScore(qualityAssessment.confidence)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Longevity Prediction
              </Typography>
              <Typography variant="h4" color="primary">
                {Math.round(qualityAssessment.predictedLongevity * 10)} years
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Expected relevance duration
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Comparative Ranking
              </Typography>
              <Typography variant="h4" color="info.main">
                {Math.round(qualityAssessment.comparativeRanking * 100)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Better than similar content
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Clinical Relevance
              </Typography>
              <Typography variant="h4" color="secondary.main">
                {formatScore(qualityAssessment.clinical_relevance)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Clinical applicability
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  // Render quality dimensions
  const renderQualityDimensions = () => {
    if (!qualityAssessment?.dimensions) return null;

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Quality Dimensions Analysis
        </Typography>

        <Grid container spacing={2}>
          {Object.entries(qualityAssessment.dimensions).map(([key, dimension]) => (
            <Grid item xs={12} md={6} key={key}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="subtitle1">
                      {dimension.name}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {getTrendIcon(dimension.trend)}
                      <Chip
                        label={formatScore(dimension.score)}
                        color={getQualityColor(dimension.score)}
                        size="small"
                      />
                    </Box>
                  </Box>

                  <LinearProgress
                    variant="determinate"
                    value={dimension.score * 100}
                    color={getQualityColor(dimension.score)}
                    sx={{ mb: 2, height: 8, borderRadius: 4 }}
                  />

                  <Typography variant="body2" color="text.secondary" paragraph>
                    {dimension.description}
                  </Typography>

                  {dimension.suggestions.length > 0 && (
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Suggestions:
                      </Typography>
                      <List dense>
                        {dimension.suggestions.slice(0, 2).map((suggestion, index) => (
                          <ListItem key={index} sx={{ px: 0 }}>
                            <ListItemIcon sx={{ minWidth: 24 }}>
                              <SuggestionIcon fontSize="small" color="primary" />
                            </ListItemIcon>
                            <ListItemText
                              primary={suggestion}
                              primaryTypographyProps={{ variant: 'body2' }}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>
    );
  };

  // Render improvement suggestions
  const renderImprovementSuggestions = () => {
    if (!qualityAssessment?.improvementSuggestions?.length) return null;

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          AI-Powered Improvement Suggestions
        </Typography>

        <List>
          {qualityAssessment.improvementSuggestions.map((suggestion, index) => (
            <ListItem key={index}>
              <ListItemIcon>
                <SuggestionIcon color="primary" />
              </ListItemIcon>
              <ListItemText
                primary={suggestion}
                secondary="Click to apply this suggestion"
              />
              <Button
                size="small"
                onClick={() => {
                  // Track suggestion application
                  improvementMutation.mutate({
                    action: 'apply_suggestion',
                    result: suggestion
                  });
                }}
              >
                Apply
              </Button>
            </ListItem>
          ))}
        </List>
      </Paper>
    );
  };

  // Render quality trends
  const renderQualityTrends = () => {
    if (!qualityTrends?.length) return null;

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Quality Evolution Over Time
        </Typography>

        <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography color="text.secondary">
            Quality trend visualization would be implemented here
          </Typography>
        </Box>

        <List>
          {qualityTrends.slice(0, 5).map((trend: QualityTrend, index: number) => (
            <ListItem key={index}>
              <ListItemText
                primary={`${trend.dimension}: ${formatScore(trend.score)}`}
                secondary={`${format(new Date(trend.date), 'MMM d, yyyy')} ${trend.note ? `• ${trend.note}` : ''}`}
              />
            </ListItem>
          ))}
        </List>
      </Paper>
    );
  };

  // Render quality issues
  const renderQualityIssues = () => {
    if (!qualityAssessment) return null;

    const issues = [
      ...qualityAssessment.evidenceGaps.map(gap => ({ type: 'evidence_gap', message: gap, severity: 'warning' })),
      ...qualityAssessment.bias_indicators.map(bias => ({ type: 'bias', message: bias, severity: 'error' })),
      ...qualityAssessment.weaknesses.map(weakness => ({ type: 'weakness', message: weakness, severity: 'info' }))
    ];

    if (issues.length === 0) {
      return (
        <Paper sx={{ p: 2 }}>
          <Alert severity="success" icon={<CheckIcon />}>
            No quality issues detected! Your content meets all quality standards.
          </Alert>
        </Paper>
      );
    }

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Quality Issues & Recommendations
        </Typography>

        {issues.map((issue, index) => (
          <Alert
            key={index}
            severity={issue.severity as any}
            sx={{ mb: 1 }}
            action={
              <Button size="small" color="inherit">
                Fix
              </Button>
            }
          >
            <Typography variant="subtitle2">
              {issue.type.replace('_', ' ').toUpperCase()}
            </Typography>
            <Typography variant="body2">
              {issue.message}
            </Typography>
          </Alert>
        ))}
      </Paper>
    );
  };

  // Render assessment history
  const renderAssessmentHistory = () => {
    if (!assessmentHistory?.length) return null;

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Assessment History
        </Typography>

        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Date</TableCell>
                <TableCell>Overall Score</TableCell>
                <TableCell>Key Changes</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {assessmentHistory.slice(0, 10).map((assessment: any, index: number) => (
                <TableRow key={index}>
                  <TableCell>
                    {format(new Date(assessment.timestamp), 'MMM d, yyyy HH:mm')}
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={formatScore(assessment.overallScore)}
                      color={getQualityColor(assessment.overallScore)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {assessment.changes || 'Initial assessment'}
                  </TableCell>
                  <TableCell>
                    <IconButton size="small">
                      <CompareIcon />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    );
  };

  // Render controls
  const renderControls = () => (
    <Paper sx={{ p: 2, mb: 3 }}>
      <Grid container spacing={2} alignItems="center">
        <Grid item xs={12} md={4}>
          <FormControlLabel
            control={
              <Switch
                checked={adaptiveLearning}
                onChange={(e) => setAdaptiveLearning(e.target.checked)}
              />
            }
            label="Adaptive Learning"
          />
        </Grid>

        <Grid item xs={12} md={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Quality Threshold</InputLabel>
            <Select
              value={qualityThreshold}
              onChange={(e) => setQualityThreshold(e.target.value as number)}
            >
              <MenuItem value={0.5}>50% - Basic</MenuItem>
              <MenuItem value={0.7}>70% - Good</MenuItem>
              <MenuItem value={0.8}>80% - High</MenuItem>
              <MenuItem value={0.9}>90% - Excellent</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} md={4}>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              size="small"
              startIcon={<RefreshIcon />}
              onClick={() => queryClient.invalidateQueries({ queryKey: ['quality-assessment'] })}
            >
              Refresh
            </Button>
            <Button
              size="small"
              startIcon={<AnalyticsIcon />}
              onClick={() => setShowComparison(true)}
            >
              Compare
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5">
          Adaptive Quality Assessment
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {isAssessing && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={16} />
              <Typography variant="body2" color="text.secondary">
                Analyzing...
              </Typography>
            </Box>
          )}
          <IconButton>
            <SettingsIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Controls */}
      {renderControls()}

      {/* Quality Summary */}
      {renderQualitySummary()}

      {/* Tabs */}
      <Box sx={{ mb: 3 }}>
        <Tabs value={selectedTab} onChange={(_, newValue) => setSelectedTab(newValue)}>
          <Tab icon={<QualityIcon />} label="Dimensions" />
          <Tab icon={<SuggestionIcon />} label="Improvements" />
          <Tab icon={<WarningIcon />} label="Issues" />
          <Tab icon={<TrendIcon />} label="Trends" />
          <Tab icon={<HistoryIcon />} label="History" />
        </Tabs>
      </Box>

      {/* Tab Content */}
      {selectedTab === 0 && renderQualityDimensions()}
      {selectedTab === 1 && renderImprovementSuggestions()}
      {selectedTab === 2 && renderQualityIssues()}
      {selectedTab === 3 && renderQualityTrends()}
      {selectedTab === 4 && renderAssessmentHistory()}

      {/* Error State */}
      {assessmentError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          Failed to assess content quality. Please try again.
        </Alert>
      )}
    </Box>
  );
};

export default AdaptiveQualityAssessment;