/**
 * Workflow Optimizer Component
 * AI-powered productivity optimization with adaptive scheduling and task management
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box, Paper, Typography, Grid, Card, CardContent,
  Button, Chip, IconButton, Tabs, Tab, LinearProgress,
  List, ListItem, ListItemText, ListItemIcon, Avatar,
  Alert, Dialog, DialogTitle, DialogContent, DialogActions,
  Accordion, AccordionSummary, AccordionDetails, Divider,
  CircularProgress, Tooltip, Badge, FormControl, InputLabel,
  Select, MenuItem, Switch, FormControlLabel, Slider,
  TextField, Timeline, TimelineItem, TimelineSeparator,
  TimelineDot, TimelineContent, TimelineConnector
} from '@mui/material';
import {
  Psychology as AIIcon,
  Speed as OptimizeIcon,
  Schedule as ScheduleIcon,
  TrendingUp as ProductivityIcon,
  Assessment as AnalyticsIcon,
  Lightbulb as SuggestionIcon,
  Timer as TimerIcon,
  PlayArrow as StartIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Notifications as NotificationIcon,
  Settings as SettingsIcon,
  AutoAwesome as SmartIcon,
  Timeline as TimelineIcon,
  EmojiObjects as InsightIcon,
  BarChart as MetricsIcon,
  Refresh as RefreshIcon,
  Warning as WarningIcon,
  CheckCircle as CompleteIcon,
  RadioButtonUnchecked as PendingIcon,
  AccessTime as TimeIcon,
  ExpandMore as ExpandMoreIcon,
  Celebration as CelebrationIcon
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format, addMinutes, differenceInMinutes, startOfDay, endOfDay } from 'date-fns';

// Services
import apiService from '../../services/api';

// Types
interface WorkflowTask {
  id: string;
  title: string;
  description: string;
  estimatedDuration: number; // in minutes
  priority: 'low' | 'medium' | 'high' | 'critical';
  complexity: number; // 1-10 scale
  energyRequired: 'low' | 'medium' | 'high';
  prerequisites: string[];
  tags: string[];
  dueDate?: Date;
  status: 'pending' | 'in_progress' | 'completed' | 'blocked';
  actualDuration?: number;
  createdAt: Date;
}

interface OptimizedSchedule {
  timeBlocks: TimeBlock[];
  productivityScore: number;
  efficiencyGain: number;
  breakSuggestions: TimeBlock[];
  adaptations: string[];
  riskFactors: string[];
}

interface TimeBlock {
  id: string;
  startTime: Date;
  endTime: Date;
  task: WorkflowTask;
  energyLevel: number;
  focusLevel: number;
  interruptions: number;
  actualProductivity?: number;
}

interface ProductivityMetrics {
  dailyProductivity: number;
  weeklyTrend: number;
  optimalTimeSlots: string[];
  averageTaskDuration: number;
  completionRate: number;
  procrastinationPattern: string;
  energyPattern: number[];
  focusPattern: number[];
}

interface WorkflowRecommendation {
  type: 'task_ordering' | 'break_timing' | 'environment' | 'energy_management';
  title: string;
  description: string;
  impact: number; // Expected productivity increase
  effort: 'low' | 'medium' | 'high';
  timeframe: string;
  actionSteps: string[];
}

const WorkflowOptimizer: React.FC = () => {
  // State
  const [selectedTab, setSelectedTab] = useState(0);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [currentTask, setCurrentTask] = useState<WorkflowTask | null>(null);
  const [workSession, setWorkSession] = useState({
    isActive: false,
    startTime: null as Date | null,
    currentTaskId: null as string | null,
    elapsedTime: 0
  });
  const [optimizationLevel, setOptimizationLevel] = useState(2); // 1-3 scale
  const [smartBreaks, setSmartBreaks] = useState(true);
  const [adaptiveScheduling, setAdaptiveScheduling] = useState(true);
  const [selectedDate, setSelectedDate] = useState(new Date());

  const queryClient = useQueryClient();

  // Timer for work session
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (workSession.isActive && workSession.startTime) {
      interval = setInterval(() => {
        setWorkSession(prev => ({
          ...prev,
          elapsedTime: differenceInMinutes(new Date(), prev.startTime!)
        }));
      }, 60000); // Update every minute
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [workSession.isActive, workSession.startTime]);

  // Workflow tasks query
  const { data: workflowTasks, isLoading: tasksLoading } = useQuery({
    queryKey: ['workflow-tasks'],
    queryFn: () => apiService.getWorkflowTasks(),
  });

  // Optimized schedule query
  const { data: optimizedSchedule, isLoading: scheduleLoading } = useQuery({
    queryKey: ['optimized-schedule', selectedDate, optimizationLevel],
    queryFn: () => apiService.getOptimizedSchedule({
      date: selectedDate,
      optimizationLevel,
      preferences: {
        smartBreaks,
        adaptiveScheduling
      }
    }),
  });

  // Productivity metrics query
  const { data: productivityMetrics } = useQuery({
    queryKey: ['productivity-metrics'],
    queryFn: () => apiService.getProductivityMetrics(),
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  // Workflow recommendations query
  const { data: workflowRecommendations } = useQuery({
    queryKey: ['workflow-recommendations'],
    queryFn: () => apiService.getWorkflowRecommendations(),
  });

  // Current session analytics
  const { data: sessionAnalytics } = useQuery({
    queryKey: ['session-analytics'],
    queryFn: () => apiService.getSessionAnalytics(),
    enabled: workSession.isActive,
    refetchInterval: 60000, // Refresh every minute during active session
  });

  // Start work session mutation
  const startSessionMutation = useMutation({
    mutationFn: (taskId: string) => apiService.startWorkSession(taskId),
    onSuccess: (data) => {
      setWorkSession({
        isActive: true,
        startTime: new Date(),
        currentTaskId: data.taskId,
        elapsedTime: 0
      });
      setCurrentTask(workflowTasks?.find((task: WorkflowTask) => task.id === data.taskId) || null);
    },
  });

  // End work session mutation
  const endSessionMutation = useMutation({
    mutationFn: (sessionData: any) => apiService.endWorkSession(sessionData),
    onSuccess: () => {
      setWorkSession({
        isActive: false,
        startTime: null,
        currentTaskId: null,
        elapsedTime: 0
      });
      setCurrentTask(null);
      queryClient.invalidateQueries({ queryKey: ['productivity-metrics'] });
    },
  });

  // Optimize workflow mutation
  const optimizeWorkflowMutation = useMutation({
    mutationFn: (params: any) => apiService.optimizeWorkflow(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['optimized-schedule'] });
    },
  });

  // Handle task start
  const handleStartTask = useCallback((task: WorkflowTask) => {
    if (workSession.isActive) {
      // End current session first
      endSessionMutation.mutate({
        taskId: workSession.currentTaskId,
        duration: workSession.elapsedTime,
        completed: false
      });
    }
    startSessionMutation.mutate(task.id);
  }, [workSession, startSessionMutation, endSessionMutation]);

  // Handle session pause/resume
  const handlePauseResume = useCallback(() => {
    if (workSession.isActive) {
      // Pause logic
      setWorkSession(prev => ({ ...prev, isActive: false }));
    } else if (workSession.currentTaskId) {
      // Resume logic
      setWorkSession(prev => ({ ...prev, isActive: true }));
    }
  }, [workSession]);

  // Handle session complete
  const handleCompleteTask = useCallback(() => {
    if (workSession.currentTaskId) {
      endSessionMutation.mutate({
        taskId: workSession.currentTaskId,
        duration: workSession.elapsedTime,
        completed: true
      });
    }
  }, [workSession, endSessionMutation]);

  // Get priority color
  const getPriorityColor = useCallback((priority: string) => {
    switch (priority) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      case 'low': return 'success';
      default: return 'default';
    }
  }, []);

  // Get energy color
  const getEnergyColor = useCallback((level: string) => {
    switch (level) {
      case 'high': return '#ff6b6b';
      case 'medium': return '#ffd93d';
      case 'low': return '#6bcf7f';
      default: return '#95a5a6';
    }
  }, []);

  // Format duration
  const formatDuration = useCallback((minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    if (hours > 0) {
      return `${hours}h ${mins}m`;
    }
    return `${mins}m`;
  }, []);

  // Render active work session
  const renderActiveSession = () => {
    if (!workSession.isActive || !currentTask) return null;

    return (
      <Paper sx={{ p: 3, mb: 3, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              🎯 Currently Working On
            </Typography>
            <Typography variant="subtitle1" gutterBottom>
              {currentTask.title}
            </Typography>
            <Typography variant="body2">
              Elapsed: {formatDuration(workSession.elapsedTime)} / {formatDuration(currentTask.estimatedDuration)}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={(workSession.elapsedTime / currentTask.estimatedDuration) * 100}
              sx={{ mt: 1, bgcolor: 'rgba(255,255,255,0.3)', '& .MuiLinearProgress-bar': { bgcolor: 'white' } }}
            />
          </Grid>

          <Grid item xs={12} md={6} sx={{ textAlign: 'right' }}>
            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
              <Button
                variant="contained"
                color="secondary"
                startIcon={workSession.isActive ? <PauseIcon /> : <StartIcon />}
                onClick={handlePauseResume}
              >
                {workSession.isActive ? 'Pause' : 'Resume'}
              </Button>
              <Button
                variant="contained"
                color="success"
                startIcon={<CompleteIcon />}
                onClick={handleCompleteTask}
              >
                Complete
              </Button>
              <Button
                variant="outlined"
                startIcon={<StopIcon />}
                onClick={handleCompleteTask}
                sx={{ color: 'white', borderColor: 'white' }}
              >
                Stop
              </Button>
            </Box>
          </Grid>
        </Grid>
      </Paper>
    );
  };

  // Render productivity metrics
  const renderProductivityMetrics = () => {
    if (!productivityMetrics) return null;

    return (
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <ProductivityIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="primary">
                {(productivityMetrics.dailyProductivity * 100).toFixed(0)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Today's Productivity
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <TrendingUpIcon color="success" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="success.main">
                {productivityMetrics.weeklyTrend > 0 ? '+' : ''}{(productivityMetrics.weeklyTrend * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Weekly Trend
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <CompleteIcon color="info" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="info.main">
                {(productivityMetrics.completionRate * 100).toFixed(0)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Completion Rate
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <TimerIcon color="secondary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="secondary.main">
                {formatDuration(productivityMetrics.averageTaskDuration)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Task Duration
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  // Render optimized schedule
  const renderOptimizedSchedule = () => {
    if (!optimizedSchedule) return null;

    return (
      <Paper sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            AI-Optimized Schedule
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip
              label={`Productivity Score: ${(optimizedSchedule.productivityScore * 100).toFixed(0)}%`}
              color="primary"
              icon={<OptimizeIcon />}
            />
            <Chip
              label={`+${(optimizedSchedule.efficiencyGain * 100).toFixed(0)}% efficiency`}
              color="success"
              icon={<TrendingUpIcon />}
            />
          </Box>
        </Box>

        <Timeline position="right">
          {optimizedSchedule.timeBlocks.map((block: TimeBlock, index: number) => (
            <TimelineItem key={block.id}>
              <TimelineSeparator>
                <TimelineDot
                  color={block.task.status === 'completed' ? 'success' : 'primary'}
                  sx={{ bgcolor: getEnergyColor(block.task.energyRequired) }}
                />
                {index < optimizedSchedule.timeBlocks.length - 1 && <TimelineConnector />}
              </TimelineSeparator>
              <TimelineContent>
                <Card sx={{ mb: 1 }}>
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                      <Typography variant="subtitle2">
                        {block.task.title}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                        <Chip
                          label={getPriorityColor(block.task.priority)}
                          color={getPriorityColor(block.task.priority)}
                          size="small"
                        />
                        {block.task.status === 'in_progress' && (
                          <Button
                            size="small"
                            startIcon={<StartIcon />}
                            onClick={() => handleStartTask(block.task)}
                          >
                            Start
                          </Button>
                        )}
                      </Box>
                    </Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {format(block.startTime, 'HH:mm')} - {format(block.endTime, 'HH:mm')} •
                      {formatDuration(differenceInMinutes(block.endTime, block.startTime))}
                    </Typography>
                    <Typography variant="body2">
                      {block.task.description}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                      <Chip label={`Energy: ${block.energyLevel}/10`} size="small" variant="outlined" />
                      <Chip label={`Focus: ${block.focusLevel}/10`} size="small" variant="outlined" />
                    </Box>
                  </CardContent>
                </Card>
              </TimelineContent>
            </TimelineItem>
          ))}
        </Timeline>

        {/* Break suggestions */}
        {optimizedSchedule.breakSuggestions.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              💡 Smart Break Suggestions
            </Typography>
            <Grid container spacing={1}>
              {optimizedSchedule.breakSuggestions.map((breakTime: TimeBlock, index: number) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <Chip
                    label={`${format(breakTime.startTime, 'HH:mm')} - ${formatDuration(differenceInMinutes(breakTime.endTime, breakTime.startTime))} break`}
                    variant="outlined"
                    color="info"
                    size="small"
                  />
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
      </Paper>
    );
  };

  // Render workflow recommendations
  const renderWorkflowRecommendations = () => {
    if (!workflowRecommendations?.length) return null;

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          🚀 Workflow Optimization Recommendations
        </Typography>

        {workflowRecommendations.map((rec: WorkflowRecommendation, index: number) => (
          <Accordion key={index}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                <Box sx={{ flexGrow: 1 }}>
                  <Typography variant="subtitle2">{rec.title}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {rec.description}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', gap: 1, ml: 2 }}>
                  <Chip
                    label={`+${(rec.impact * 100).toFixed(0)}%`}
                    color="success"
                    size="small"
                  />
                  <Chip
                    label={rec.effort}
                    color={rec.effort === 'low' ? 'success' : rec.effort === 'medium' ? 'warning' : 'error'}
                    size="small"
                  />
                </Box>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" paragraph>
                Expected timeframe: {rec.timeframe}
              </Typography>
              <Typography variant="subtitle2" gutterBottom>
                Action Steps:
              </Typography>
              <List dense>
                {rec.actionSteps.map((step, stepIndex) => (
                  <ListItem key={stepIndex} sx={{ px: 0 }}>
                    <ListItemIcon sx={{ minWidth: 24 }}>
                      <Typography variant="body2" color="primary">
                        {stepIndex + 1}.
                      </Typography>
                    </ListItemIcon>
                    <ListItemText primary={step} primaryTypographyProps={{ variant: 'body2' }} />
                  </ListItem>
                ))}
              </List>
              <Button variant="outlined" size="small" sx={{ mt: 1 }}>
                Apply Recommendation
              </Button>
            </AccordionDetails>
          </Accordion>
        ))}
      </Paper>
    );
  };

  // Render task list
  const renderTaskList = () => {
    if (!workflowTasks?.length) return null;

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Task Queue
        </Typography>

        <List>
          {workflowTasks.map((task: WorkflowTask) => (
            <ListItem
              key={task.id}
              sx={{
                border: 1,
                borderColor: 'divider',
                borderRadius: 1,
                mb: 1,
                bgcolor: task.status === 'completed' ? 'success.light' : 'background.paper'
              }}
            >
              <ListItemIcon>
                {task.status === 'completed' ? (
                  <CompleteIcon color="success" />
                ) : task.status === 'in_progress' ? (
                  <TimerIcon color="primary" />
                ) : (
                  <PendingIcon color="disabled" />
                )}
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle2">{task.title}</Typography>
                    <Chip
                      label={task.priority}
                      color={getPriorityColor(task.priority)}
                      size="small"
                    />
                    <Chip
                      label={formatDuration(task.estimatedDuration)}
                      variant="outlined"
                      size="small"
                    />
                  </Box>
                }
                secondary={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      {task.description}
                    </Typography>
                    {task.dueDate && (
                      <Typography variant="caption" color="warning.main">
                        Due: {format(new Date(task.dueDate), 'MMM d, yyyy')}
                      </Typography>
                    )}
                  </Box>
                }
              />
              {task.status !== 'completed' && (
                <Button
                  size="small"
                  startIcon={<StartIcon />}
                  onClick={() => handleStartTask(task)}
                  disabled={workSession.isActive && workSession.currentTaskId !== task.id}
                >
                  Start
                </Button>
              )}
            </ListItem>
          ))}
        </List>
      </Paper>
    );
  };

  // Render settings panel
  const renderSettings = () => (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Optimization Settings
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography gutterBottom>Optimization Level</Typography>
          <Slider
            value={optimizationLevel}
            onChange={(_, value) => setOptimizationLevel(value as number)}
            min={1}
            max={3}
            step={1}
            marks={[
              { value: 1, label: 'Conservative' },
              { value: 2, label: 'Balanced' },
              { value: 3, label: 'Aggressive' }
            ]}
            valueLabelDisplay="off"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormControlLabel
            control={
              <Switch
                checked={smartBreaks}
                onChange={(e) => setSmartBreaks(e.target.checked)}
              />
            }
            label="Smart Break Timing"
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormControlLabel
            control={
              <Switch
                checked={adaptiveScheduling}
                onChange={(e) => setAdaptiveScheduling(e.target.checked)}
              />
            }
            label="Adaptive Scheduling"
          />
        </Grid>

        <Grid item xs={12}>
          <Button
            variant="contained"
            startIcon={<OptimizeIcon />}
            onClick={() => optimizeWorkflowMutation.mutate({
              optimizationLevel,
              smartBreaks,
              adaptiveScheduling
            })}
            disabled={optimizeWorkflowMutation.isPending}
          >
            {optimizeWorkflowMutation.isPending ? 'Optimizing...' : 'Optimize Workflow'}
          </Button>
        </Grid>
      </Grid>
    </Paper>
  );

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Workflow Optimizer
        </Typography>
        <Box>
          <IconButton onClick={() => queryClient.invalidateQueries()}>
            <RefreshIcon />
          </IconButton>
          <IconButton>
            <SettingsIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Active Session */}
      {renderActiveSession()}

      {/* Productivity Metrics */}
      {renderProductivityMetrics()}

      {/* Tabs */}
      <Box sx={{ mb: 3 }}>
        <Tabs value={selectedTab} onChange={(_, newValue) => setSelectedTab(newValue)}>
          <Tab icon={<ScheduleIcon />} label="Schedule" />
          <Tab icon={<SuggestionIcon />} label="Recommendations" />
          <Tab icon={<TimerIcon />} label="Tasks" />
          <Tab icon={<AnalyticsIcon />} label="Analytics" />
          <Tab icon={<SettingsIcon />} label="Settings" />
        </Tabs>
      </Box>

      {/* Tab Content */}
      {selectedTab === 0 && renderOptimizedSchedule()}
      {selectedTab === 1 && renderWorkflowRecommendations()}
      {selectedTab === 2 && renderTaskList()}
      {selectedTab === 3 && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Analytics coming soon...</Typography>
        </Paper>
      )}
      {selectedTab === 4 && renderSettings()}

      {/* Loading States */}
      {(tasksLoading || scheduleLoading) && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      )}
    </Box>
  );
};

export default WorkflowOptimizer;