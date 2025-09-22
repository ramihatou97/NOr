/**
 * Intelligent Dashboard Component
 * Provides comprehensive overview of user's writing progress, insights, and AI recommendations
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Box, Grid, Paper, Typography, Card, CardContent,
  LinearProgress, Chip, IconButton, Tabs, Tab,
  List, ListItem, ListItemText, ListItemIcon,
  Avatar, Tooltip, Button, Alert, Divider,
  CircularProgress, Badge, AccordionSummary,
  Accordion, AccordionDetails
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  TrendingUp as TrendIcon,
  Psychology as AIIcon,
  Assessment as QualityIcon,
  Timeline as WorkflowIcon,
  Science as ResearchIcon,
  Warning as ConflictIcon,
  Lightbulb as InsightIcon,
  AutoAwesome as PredictiveIcon,
  Speed as PerformanceIcon,
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { format, formatDistanceToNow } from 'date-fns';

// Services
import apiService from '../../services/api';

// Types
interface DashboardMetrics {
  totalChapters: number;
  weeklyProgress: number;
  averageQualityScore: number;
  researchHours: number;
  conflictsResolved: number;
  predictiveAccuracy: number;
}

interface IntelligenceInsight {
  id: string;
  type: 'quality' | 'research' | 'workflow' | 'predictive' | 'performance';
  title: string;
  description: string;
  severity: 'info' | 'warning' | 'error';
  actionable: boolean;
  action?: string;
  createdAt: Date;
}

interface WorkflowRecommendation {
  id: string;
  title: string;
  description: string;
  estimatedImpact: number;
  timeToImplement: string;
  category: 'productivity' | 'quality' | 'research' | 'writing';
}

interface PredictiveAlert {
  id: string;
  type: 'deadline_risk' | 'quality_concern' | 'research_gap' | 'workflow_optimization';
  message: string;
  confidence: number;
  suggestedAction: string;
  timeframe: string;
}

const IntelligentDashboard: React.FC = () => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const queryClient = useQueryClient();

  // Dashboard metrics query
  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['dashboard-metrics', refreshTrigger],
    queryFn: () => apiService.getDashboardMetrics(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Intelligence insights query
  const { data: insights, isLoading: insightsLoading } = useQuery({
    queryKey: ['intelligence-insights', refreshTrigger],
    queryFn: () => apiService.getIntelligenceInsights(),
    refetchInterval: 60000, // Refresh every minute
  });

  // Workflow recommendations query
  const { data: workflowRecommendations, isLoading: workflowLoading } = useQuery({
    queryKey: ['workflow-recommendations', refreshTrigger],
    queryFn: () => apiService.getWorkflowRecommendations(),
  });

  // Predictive alerts query
  const { data: predictiveAlerts, isLoading: alertsLoading } = useQuery({
    queryKey: ['predictive-alerts', refreshTrigger],
    queryFn: () => apiService.getPredictiveAlerts(),
    refetchInterval: 120000, // Refresh every 2 minutes
  });

  // Recent activity query
  const { data: recentActivity, isLoading: activityLoading } = useQuery({
    queryKey: ['recent-activity'],
    queryFn: () => apiService.getRecentActivity(),
  });

  // Knowledge graph insights
  const { data: knowledgeInsights, isLoading: knowledgeLoading } = useQuery({
    queryKey: ['knowledge-insights'],
    queryFn: () => apiService.getKnowledgeGraphInsights(),
  });

  // Calculate overall intelligence score
  const intelligenceScore = useMemo(() => {
    if (!metrics) return 0;
    const qualityWeight = 0.3;
    const productivityWeight = 0.25;
    const researchWeight = 0.25;
    const conflictWeight = 0.2;

    return Math.round(
      (metrics.averageQualityScore * qualityWeight) +
      (metrics.weeklyProgress * productivityWeight) +
      (Math.min(metrics.researchHours / 10, 1) * researchWeight) +
      (Math.min(metrics.conflictsResolved / 5, 1) * conflictWeight)
    ) * 100;
  }, [metrics]);

  // Get insight severity color
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'error': return 'error';
      case 'warning': return 'warning';
      case 'info': return 'info';
      default: return 'default';
    }
  };

  // Get insight icon
  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'quality': return <QualityIcon />;
      case 'research': return <ResearchIcon />;
      case 'workflow': return <WorkflowIcon />;
      case 'predictive': return <PredictiveIcon />;
      case 'performance': return <PerformanceIcon />;
      default: return <InsightIcon />;
    }
  };

  // Handle manual refresh
  const handleRefresh = () => {
    setRefreshTrigger(prev => prev + 1);
    queryClient.invalidateQueries({ queryKey: ['dashboard'] });
  };

  // Handle tab change
  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  };

  // Render metrics cards
  const renderMetricsCards = () => (
    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <QualityIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h6">Intelligence Score</Typography>
            </Box>
            <Typography variant="h3" color="primary">
              {intelligenceScore}%
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Overall system intelligence
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <TrendIcon color="success" sx={{ mr: 1 }} />
              <Typography variant="h6">Weekly Progress</Typography>
            </Box>
            <Typography variant="h3" color="success.main">
              {metrics?.weeklyProgress.toFixed(1)}%
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Compared to last week
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <ResearchIcon color="info" sx={{ mr: 1 }} />
              <Typography variant="h6">Research Hours</Typography>
            </Box>
            <Typography variant="h3" color="info.main">
              {metrics?.researchHours || 0}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              This week
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <ConflictIcon color="warning" sx={{ mr: 1 }} />
              <Typography variant="h6">Conflicts Resolved</Typography>
            </Box>
            <Typography variant="h3" color="warning.main">
              {metrics?.conflictsResolved || 0}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Automatically detected
            </Typography>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  // Render predictive alerts
  const renderPredictiveAlerts = () => {
    if (!predictiveAlerts?.length) return null;

    return (
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Predictive Alerts
        </Typography>
        {predictiveAlerts.map((alert: PredictiveAlert) => (
          <Alert
            key={alert.id}
            severity={alert.type === 'deadline_risk' ? 'error' : 'warning'}
            sx={{ mb: 1 }}
            action={
              <Button size="small" color="inherit">
                Act Now
              </Button>
            }
          >
            <Typography variant="subtitle2">{alert.message}</Typography>
            <Typography variant="body2">
              Confidence: {(alert.confidence * 100).toFixed(0)}% | {alert.timeframe}
            </Typography>
            <Typography variant="body2" sx={{ fontStyle: 'italic' }}>
              💡 {alert.suggestedAction}
            </Typography>
          </Alert>
        ))}
      </Paper>
    );
  };

  // Render intelligence insights
  const renderIntelligenceInsights = () => (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Intelligence Insights</Typography>
        <IconButton onClick={handleRefresh} size="small">
          <RefreshIcon />
        </IconButton>
      </Box>

      {insightsLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <List>
          {insights?.map((insight: IntelligenceInsight) => (
            <ListItem key={insight.id} sx={{ px: 0 }}>
              <ListItemIcon>
                <Badge
                  color={getSeverityColor(insight.severity)}
                  variant="dot"
                  invisible={insight.severity === 'info'}
                >
                  {getInsightIcon(insight.type)}
                </Badge>
              </ListItemIcon>
              <ListItemText
                primary={insight.title}
                secondary={
                  <>
                    <Typography variant="body2" color="text.secondary">
                      {insight.description}
                    </Typography>
                    {insight.actionable && insight.action && (
                      <Typography variant="body2" color="primary" sx={{ mt: 0.5 }}>
                        💡 {insight.action}
                      </Typography>
                    )}
                    <Typography variant="caption" color="text.secondary">
                      {formatDistanceToNow(new Date(insight.createdAt), { addSuffix: true })}
                    </Typography>
                  </>
                }
              />
            </ListItem>
          ))}
        </List>
      )}
    </Paper>
  );

  // Render workflow recommendations
  const renderWorkflowRecommendations = () => (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Workflow Optimization
      </Typography>
      {workflowRecommendations?.map((rec: WorkflowRecommendation) => (
        <Accordion key={rec.id}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
              <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
                {rec.title}
              </Typography>
              <Chip
                label={`+${(rec.estimatedImpact * 100).toFixed(0)}% productivity`}
                size="small"
                color="success"
                sx={{ ml: 1 }}
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Typography variant="body2" paragraph>
              {rec.description}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Chip
                label={`${rec.timeToImplement} to implement`}
                size="small"
                variant="outlined"
              />
              <Chip
                label={rec.category}
                size="small"
                color="primary"
              />
            </Box>
          </AccordionDetails>
        </Accordion>
      ))}
    </Paper>
  );

  // Render activity timeline
  const renderActivityTimeline = () => (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Recent Activity
      </Typography>
      {activityLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <CircularProgress />
        </Box>
      ) : (
        <List dense>
          {recentActivity?.map((activity: any, index: number) => (
            <ListItem key={index}>
              <ListItemIcon>
                <Avatar sx={{ width: 24, height: 24, fontSize: '0.75rem' }}>
                  {activity.type.charAt(0).toUpperCase()}
                </Avatar>
              </ListItemIcon>
              <ListItemText
                primary={activity.title}
                secondary={`${activity.description} • ${format(new Date(activity.timestamp), 'MMM d, HH:mm')}`}
              />
            </ListItem>
          ))}
        </List>
      )}
    </Paper>
  );

  // Render knowledge connections
  const renderKnowledgeConnections = () => (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Knowledge Connections
      </Typography>
      {knowledgeLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box>
          {knowledgeInsights?.topConnections?.map((connection: any, index: number) => (
            <Box key={index} sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
              <Typography variant="subtitle2">
                {connection.sourceNode} ↔ {connection.targetNode}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Strength: {(connection.strength * 100).toFixed(0)}% |
                {connection.relationship}
              </Typography>
            </Box>
          ))}
        </Box>
      )}
    </Paper>
  );

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Intelligent Dashboard
        </Typography>
        <Box>
          <IconButton onClick={handleRefresh}>
            <RefreshIcon />
          </IconButton>
          <IconButton>
            <SettingsIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Metrics Cards */}
      {renderMetricsCards()}

      {/* Predictive Alerts */}
      {renderPredictiveAlerts()}

      {/* Main Content */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Box sx={{ mb: 3 }}>
            <Tabs value={selectedTab} onChange={handleTabChange}>
              <Tab icon={<InsightIcon />} label="Insights" />
              <Tab icon={<WorkflowIcon />} label="Workflow" />
              <Tab icon={<TrendIcon />} label="Activity" />
              <Tab icon={<ResearchIcon />} label="Knowledge" />
            </Tabs>
          </Box>

          {selectedTab === 0 && renderIntelligenceInsights()}
          {selectedTab === 1 && renderWorkflowRecommendations()}
          {selectedTab === 2 && renderActivityTimeline()}
          {selectedTab === 3 && renderKnowledgeConnections()}
        </Grid>

        <Grid item xs={12} md={4}>
          {/* Real-time Performance Monitor */}
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Performance
            </Typography>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" gutterBottom>
                AI Response Time
              </Typography>
              <LinearProgress
                variant="determinate"
                value={85}
                color="success"
                sx={{ mb: 1 }}
              />
              <Typography variant="caption">
                Average: 1.2s
              </Typography>
            </Box>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" gutterBottom>
                Quality Analysis
              </Typography>
              <LinearProgress
                variant="determinate"
                value={92}
                color="info"
                sx={{ mb: 1 }}
              />
              <Typography variant="caption">
                Running optimally
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2" gutterBottom>
                Predictive Accuracy
              </Typography>
              <LinearProgress
                variant="determinate"
                value={metrics?.predictiveAccuracy || 0}
                color="primary"
                sx={{ mb: 1 }}
              />
              <Typography variant="caption">
                {(metrics?.predictiveAccuracy || 0).toFixed(1)}% this week
              </Typography>
            </Box>
          </Paper>

          {/* Quick Actions */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Button variant="outlined" startIcon={<AIIcon />} fullWidth>
                Run Intelligence Scan
              </Button>
              <Button variant="outlined" startIcon={<ResearchIcon />} fullWidth>
                Update Knowledge Graph
              </Button>
              <Button variant="outlined" startIcon={<PerformanceIcon />} fullWidth>
                Optimize Performance
              </Button>
              <Button variant="outlined" startIcon={<QualityIcon />} fullWidth>
                Quality Assessment
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default IntelligentDashboard;