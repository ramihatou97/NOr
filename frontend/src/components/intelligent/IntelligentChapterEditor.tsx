/**
 * Intelligent Chapter Editor with Real-time AI Assistance
 * Integrates all intelligence modules for enhanced writing experience
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box, Paper, Typography, TextField, Button, Grid,
  Card, CardContent, Chip, IconButton, Tabs, Tab,
  LinearProgress, Alert, Dialog, DialogTitle, DialogContent,
  List, ListItem, ListItemText, ListItemIcon, Fab,
  Accordion, AccordionSummary, AccordionDetails, Badge
} from '@mui/material';
import {
  Save as SaveIcon,
  AutoAwesome as AIIcon,
  Science as ResearchIcon,
  Timeline as PredictiveIcon,
  Assessment as QualityIcon,
  Warning as ConflictIcon,
  ExpandMore as ExpandMoreIcon,
  Lightbulb as SuggestionIcon,
  TrendingUp as TrendIcon,
  Speed as PerformanceIcon
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useDebounce } from 'use-debounce';

// Services
import apiService from '../../services/api';

// Types
interface IntelligenceInsight {
  type: 'quality' | 'research' | 'conflict' | 'prediction' | 'workflow';
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  actionable: boolean;
  suggestion?: string;
}

interface ChapterIntelligence {
  qualityScore: number;
  conflictsDetected: number;
  researchGaps: number;
  predictiveInsights: number;
  workflowOptimization: number;
}

interface IntelligentChapterEditorProps {
  chapterId?: string;
  initialContent?: string;
  onSave?: (content: string) => void;
}

const IntelligentChapterEditor: React.FC<IntelligentChapterEditorProps> = ({
  chapterId,
  initialContent = '',
  onSave
}) => {
  // State
  const [content, setContent] = useState(initialContent);
  const [title, setTitle] = useState('');
  const [specialty, setSpecialty] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [selectedTab, setSelectedTab] = useState(0);
  const [showIntelligencePanel, setShowIntelligencePanel] = useState(false);
  const [realTimeAnalysis, setRealTimeAnalysis] = useState(true);

  // Debounced content for analysis
  const [debouncedContent] = useDebounce(content, 2000);
  const [debouncedTitle] = useDebounce(title, 1000);

  const queryClient = useQueryClient();

  // Real-time intelligence analysis
  const {
    data: intelligenceData,
    isLoading: isAnalyzing,
    error: analysisError
  } = useQuery({
    queryKey: ['chapter-intelligence', debouncedContent, debouncedTitle],
    queryFn: async () => {
      if (!debouncedContent || debouncedContent.length < 100) return null;

      const response = await apiService.analyzeChapterIntelligence({
        content: debouncedContent,
        title: debouncedTitle,
        specialty,
        context: {
          realTime: true,
          analysisTypes: ['quality', 'conflicts', 'research_gaps', 'predictions']
        }
      });

      return response;
    },
    enabled: realTimeAnalysis && debouncedContent.length > 100,
    refetchInterval: 30000, // Re-analyze every 30 seconds
  });

  // Chapter creation/update mutation
  const chapterMutation = useMutation({
    mutationFn: async (data: any) => {
      if (chapterId) {
        return apiService.updateIntelligentChapter(chapterId, data);
      } else {
        return apiService.createIntelligentChapter(data);
      }
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['chapters'] });
      if (onSave) {
        onSave(content);
      }
    },
  });

  // Intelligence insights processing
  const intelligenceInsights = useMemo(() => {
    if (!intelligenceData) return [];

    const insights: IntelligenceInsight[] = [];

    // Quality insights
    if (intelligenceData.qualityAssessment) {
      const quality = intelligenceData.qualityAssessment;
      if (quality.overallScore < 0.7) {
        insights.push({
          type: 'quality',
          title: 'Quality Score Below Threshold',
          description: `Current quality score: ${(quality.overallScore * 100).toFixed(0)}%`,
          severity: quality.overallScore < 0.5 ? 'critical' : 'medium',
          actionable: true,
          suggestion: quality.improvementSuggestions?.[0]
        });
      }
    }

    // Conflict insights
    if (intelligenceData.conflictAnalysis?.conflicts?.length > 0) {
      insights.push({
        type: 'conflict',
        title: 'Conflicts Detected',
        description: `${intelligenceData.conflictAnalysis.conflicts.length} potential conflicts found`,
        severity: 'high',
        actionable: true,
        suggestion: 'Review and resolve conflicting information'
      });
    }

    // Research gap insights
    if (intelligenceData.researchRecommendations?.length > 0) {
      insights.push({
        type: 'research',
        title: 'Research Opportunities',
        description: `${intelligenceData.researchRecommendations.length} research recommendations available`,
        severity: 'medium',
        actionable: true,
        suggestion: 'Consider incorporating latest research findings'
      });
    }

    // Workflow insights
    if (intelligenceData.workflowSuggestions?.predictedProductivity) {
      const productivity = intelligenceData.workflowSuggestions.predictedProductivity;
      if (productivity < 0.6) {
        insights.push({
          type: 'workflow',
          title: 'Workflow Optimization Available',
          description: `Predicted productivity: ${(productivity * 100).toFixed(0)}%`,
          severity: 'low',
          actionable: true,
          suggestion: 'Consider workflow optimization suggestions'
        });
      }
    }

    return insights.sort((a, b) => {
      const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      return severityOrder[b.severity] - severityOrder[a.severity];
    });
  }, [intelligenceData]);

  // Intelligence summary
  const intelligenceSummary = useMemo((): ChapterIntelligence => {
    if (!intelligenceData) {
      return {
        qualityScore: 0,
        conflictsDetected: 0,
        researchGaps: 0,
        predictiveInsights: 0,
        workflowOptimization: 0
      };
    }

    return {
      qualityScore: intelligenceData.qualityAssessment?.overallScore || 0,
      conflictsDetected: intelligenceData.conflictAnalysis?.conflicts?.length || 0,
      researchGaps: intelligenceData.researchRecommendations?.length || 0,
      predictiveInsights: Object.keys(intelligenceData.intelligenceSummary || {}).length,
      workflowOptimization: intelligenceData.workflowSuggestions?.predictedProductivity || 0
    };
  }, [intelligenceData]);

  // Event handlers
  const handleSave = useCallback(() => {
    chapterMutation.mutate({
      title,
      content,
      specialty,
      tags,
      context: {
        editorSession: true,
        intelligenceEnabled: realTimeAnalysis
      }
    });
  }, [title, content, specialty, tags, realTimeAnalysis, chapterMutation]);

  const handleContentChange = useCallback((event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContent(event.target.value);
  }, []);

  const handleTitleChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setTitle(event.target.value);
  }, []);

  const handleTabChange = useCallback((_: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  }, []);

  // Render intelligence status chips
  const renderIntelligenceStatus = () => (
    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
      <Chip
        icon={<QualityIcon />}
        label={`Quality: ${(intelligenceSummary.qualityScore * 100).toFixed(0)}%`}
        color={intelligenceSummary.qualityScore > 0.8 ? 'success' : intelligenceSummary.qualityScore > 0.6 ? 'warning' : 'error'}
        size="small"
      />

      {intelligenceSummary.conflictsDetected > 0 && (
        <Chip
          icon={<ConflictIcon />}
          label={`${intelligenceSummary.conflictsDetected} Conflicts`}
          color="error"
          size="small"
        />
      )}

      {intelligenceSummary.researchGaps > 0 && (
        <Chip
          icon={<ResearchIcon />}
          label={`${intelligenceSummary.researchGaps} Research Opportunities`}
          color="info"
          size="small"
        />
      )}

      <Chip
        icon={<TrendIcon />}
        label={`Workflow: ${(intelligenceSummary.workflowOptimization * 100).toFixed(0)}%`}
        color={intelligenceSummary.workflowOptimization > 0.7 ? 'success' : 'default'}
        size="small"
      />
    </Box>
  );

  // Render intelligence insights panel
  const renderIntelligencePanel = () => (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography variant="h6" gutterBottom>
        Intelligence Insights
        {isAnalyzing && <LinearProgress sx={{ mt: 1 }} />}
      </Typography>

      {intelligenceInsights.length === 0 ? (
        <Alert severity="success">
          No issues detected. Your content looks great!
        </Alert>
      ) : (
        <List>
          {intelligenceInsights.map((insight, index) => (
            <ListItem key={index}>
              <ListItemIcon>
                {insight.type === 'quality' && <QualityIcon color={insight.severity === 'critical' ? 'error' : 'warning'} />}
                {insight.type === 'conflict' && <ConflictIcon color="error" />}
                {insight.type === 'research' && <ResearchIcon color="info" />}
                {insight.type === 'prediction' && <PredictiveIcon color="primary" />}
                {insight.type === 'workflow' && <PerformanceIcon color="secondary" />}
              </ListItemIcon>
              <ListItemText
                primary={insight.title}
                secondary={
                  <>
                    <Typography variant="body2" color="text.secondary">
                      {insight.description}
                    </Typography>
                    {insight.suggestion && (
                      <Typography variant="body2" color="primary" sx={{ mt: 0.5 }}>
                        💡 {insight.suggestion}
                      </Typography>
                    )}
                  </>
                }
              />
            </ListItem>
          ))}
        </List>
      )}
    </Paper>
  );

  // Render research recommendations
  const renderResearchRecommendations = () => {
    if (!intelligenceData?.researchRecommendations?.length) return null;

    return (
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Research Recommendations
        </Typography>
        {intelligenceData.researchRecommendations.slice(0, 3).map((rec: any, index: number) => (
          <Accordion key={index}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2">
                {rec.title}
              </Typography>
              <Chip
                label={`${(rec.relevanceScore * 100).toFixed(0)}% relevant`}
                size="small"
                sx={{ ml: 1 }}
              />
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2">
                {rec.abstract}
              </Typography>
              <Button
                size="small"
                startIcon={<ResearchIcon />}
                sx={{ mt: 1 }}
                onClick={() => window.open(rec.url, '_blank')}
              >
                View Paper
              </Button>
            </AccordionDetails>
          </Accordion>
        ))}
      </Paper>
    );
  };

  return (
    <Box>
      {/* Main Editor */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={showIntelligencePanel ? 8 : 12}>
          <Paper sx={{ p: 3 }}>
            {/* Header */}
            <Box sx={{ mb: 3 }}>
              <TextField
                fullWidth
                placeholder="Chapter Title"
                value={title}
                onChange={handleTitleChange}
                variant="outlined"
                sx={{ mb: 2 }}
              />

              {renderIntelligenceStatus()}

              <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={handleSave}
                  disabled={chapterMutation.isPending}
                >
                  {chapterMutation.isPending ? 'Saving...' : 'Save Chapter'}
                </Button>

                <Button
                  variant="outlined"
                  startIcon={<AIIcon />}
                  onClick={() => setShowIntelligencePanel(!showIntelligencePanel)}
                >
                  {showIntelligencePanel ? 'Hide' : 'Show'} Intelligence
                </Button>
              </Box>
            </Box>

            {/* Content Editor */}
            <TextField
              fullWidth
              multiline
              rows={25}
              placeholder="Start writing your chapter content..."
              value={content}
              onChange={handleContentChange}
              variant="outlined"
              sx={{
                '& .MuiOutlinedInput-root': {
                  fontSize: '16px',
                  lineHeight: '1.6'
                }
              }}
            />

            {/* Real-time Analysis Status */}
            {isAnalyzing && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  🧠 AI analyzing your content...
                </Typography>
                <LinearProgress />
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Intelligence Panel */}
        {showIntelligencePanel && (
          <Grid item xs={12} md={4}>
            <Box sx={{ position: 'sticky', top: 24 }}>
              {/* Intelligence Insights */}
              {renderIntelligencePanel()}

              {/* Research Recommendations */}
              {renderResearchRecommendations()}

              {/* Tabs for Additional Intelligence */}
              <Paper sx={{ p: 2 }}>
                <Tabs value={selectedTab} onChange={handleTabChange} variant="fullWidth">
                  <Tab icon={<QualityIcon />} label="Quality" />
                  <Tab icon={<PredictiveIcon />} label="Predictions" />
                  <Tab icon={<PerformanceIcon />} label="Workflow" />
                </Tabs>

                <Box sx={{ mt: 2 }}>
                  {selectedTab === 0 && (
                    <Box>
                      {intelligenceData?.qualityAssessment && (
                        <>
                          <Typography variant="subtitle2" gutterBottom>
                            Quality Dimensions
                          </Typography>
                          {Object.entries(intelligenceData.qualityAssessment.dimensionScores || {}).map(([dim, score]) => (
                            <Box key={dim} sx={{ mb: 1 }}>
                              <Typography variant="caption">
                                {dim.replace('_', ' ')}
                              </Typography>
                              <LinearProgress
                                variant="determinate"
                                value={(score as number) * 100}
                                sx={{ height: 6, borderRadius: 3 }}
                              />
                            </Box>
                          ))}
                        </>
                      )}
                    </Box>
                  )}

                  {selectedTab === 1 && (
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Predictive Insights
                      </Typography>
                      {intelligenceData?.intelligenceSummary && (
                        <List dense>
                          <ListItem>
                            <ListItemText
                              primary="Next Likely Action"
                              secondary="Research validation recommended"
                            />
                          </ListItem>
                          <ListItem>
                            <ListItemText
                              primary="Content Expansion"
                              secondary="Clinical implications section suggested"
                            />
                          </ListItem>
                        </List>
                      )}
                    </Box>
                  )}

                  {selectedTab === 2 && (
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Workflow Optimization
                      </Typography>
                      {intelligenceData?.workflowSuggestions && (
                        <List dense>
                          <ListItem>
                            <ListItemText
                              primary="Optimal Writing Time"
                              secondary="Peak productivity in 15 minutes"
                            />
                          </ListItem>
                          <ListItem>
                            <ListItemText
                              primary="Focus Recommendation"
                              secondary="High-quality research phase suggested"
                            />
                          </ListItem>
                        </List>
                      )}
                    </Box>
                  )}
                </Box>
              </Paper>
            </Box>
          </Grid>
        )}
      </Grid>

      {/* Floating Action Buttons */}
      <Box sx={{ position: 'fixed', bottom: 24, right: 24, display: 'flex', flexDirection: 'column', gap: 1 }}>
        <Badge badgeContent={intelligenceInsights.length} color="error">
          <Fab
            color="primary"
            onClick={() => setShowIntelligencePanel(!showIntelligencePanel)}
          >
            <AIIcon />
          </Fab>
        </Badge>
      </Box>
    </Box>
  );
};

export default IntelligentChapterEditor;