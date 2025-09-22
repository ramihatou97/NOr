/**
 * Intelligent Research Assistant Component
 * AI-powered research companion with multi-source integration and context awareness
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box, Paper, Typography, TextField, Button, Grid,
  Card, CardContent, Chip, IconButton, Tabs, Tab,
  LinearProgress, Alert, Dialog, DialogTitle, DialogContent,
  List, ListItem, ListItemText, ListItemIcon, Fab,
  Accordion, AccordionSummary, AccordionDetails, Badge,
  Avatar, Tooltip, Divider, CircularProgress, Autocomplete,
  FormControl, InputLabel, Select, MenuItem, Switch,
  FormControlLabel, Collapse, Slider
} from '@mui/material';
import {
  Search as SearchIcon,
  Science as ResearchIcon,
  AutoAwesome as AIIcon,
  BookmarkAdd as SaveIcon,
  FilterList as FilterIcon,
  TrendingUp as TrendIcon,
  Visibility as ViewIcon,
  Download as DownloadIcon,
  Share as ShareIcon,
  Timer as TimerIcon,
  Star as StarIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Psychology as BrainIcon,
  ExpandMore as ExpandMoreIcon,
  Close as CloseIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useDebounce } from 'use-debounce';
import { format } from 'date-fns';

// Services
import apiService from '../../services/api';

// Types
interface ResearchQuery {
  query: string;
  domain: string;
  urgency: number;
  qualityThreshold: number;
  maxResults: number;
  sourcePreferences: string[];
  timeRange: string;
  includeGrayLiterature: boolean;
  contextualExpansion: boolean;
}

interface ResearchResult {
  id: string;
  title: string;
  authors: string[];
  journal: string;
  publicationDate: Date;
  abstract: string;
  url: string;
  qualityScore: number;
  relevanceScore: number;
  evidenceLevel: string;
  citationCount: number;
  accessType: 'open' | 'subscription' | 'paywall';
  keyFindings: string[];
  methodology: string;
  sampleSize?: number;
  studyType: string;
  conflictsPotential: number;
  synthesisReady: boolean;
}

interface ResearchSession {
  id: string;
  query: string;
  results: ResearchResult[];
  savedResults: string[];
  insights: string[];
  startTime: Date;
  duration: number;
  contextTags: string[];
}

interface SmartSuggestion {
  type: 'query_expansion' | 'source_recommendation' | 'methodology_filter' | 'temporal_focus';
  title: string;
  description: string;
  action: string;
  confidence: number;
}

const IntelligentResearchAssistant: React.FC = () => {
  // State management
  const [query, setQuery] = useState('');
  const [activeSession, setActiveSession] = useState<ResearchSession | null>(null);
  const [selectedFilters, setSelectedFilters] = useState<Partial<ResearchQuery>>({
    domain: 'all',
    urgency: 3,
    qualityThreshold: 0.7,
    maxResults: 20,
    sourcePreferences: [],
    timeRange: 'all',
    includeGrayLiterature: false,
    contextualExpansion: true
  });
  const [selectedTab, setSelectedTab] = useState(0);
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [smartMode, setSmartMode] = useState(true);
  const [savedQueries, setSavedQueries] = useState<string[]>([]);
  const [selectedResults, setSelectedResults] = useState<Set<string>>(new Set());

  // Debounced query for intelligent suggestions
  const [debouncedQuery] = useDebounce(query, 1000);

  const queryClient = useQueryClient();
  const resultsRef = useRef<HTMLDivElement>(null);

  // Research query execution
  const researchMutation = useMutation({
    mutationFn: async (searchQuery: ResearchQuery) => {
      const response = await apiService.intelligentSearch(searchQuery);
      return response;
    },
    onSuccess: (data) => {
      setActiveSession({
        id: `session_${Date.now()}`,
        query: query,
        results: data.results,
        savedResults: [],
        insights: data.insights || [],
        startTime: new Date(),
        duration: 0,
        contextTags: data.contextTags || []
      });
    },
  });

  // Smart suggestions query
  const { data: smartSuggestions, isLoading: suggestionsLoading } = useQuery({
    queryKey: ['smart-suggestions', debouncedQuery, selectedFilters],
    queryFn: () => apiService.getSmartResearchSuggestions(debouncedQuery, selectedFilters),
    enabled: smartMode && debouncedQuery.length > 3,
  });

  // Research history query
  const { data: researchHistory } = useQuery({
    queryKey: ['research-history'],
    queryFn: () => apiService.getResearchHistory(),
  });

  // Domain suggestions query
  const { data: domainSuggestions } = useQuery({
    queryKey: ['domain-suggestions'],
    queryFn: () => apiService.getDomainSuggestions(),
  });

  // Real-time research trends
  const { data: researchTrends } = useQuery({
    queryKey: ['research-trends'],
    queryFn: () => apiService.getResearchTrends(),
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  // Handle search execution
  const handleSearch = useCallback(() => {
    if (!query.trim()) return;

    const searchQuery: ResearchQuery = {
      query: query.trim(),
      domain: selectedFilters.domain || 'all',
      urgency: selectedFilters.urgency || 3,
      qualityThreshold: selectedFilters.qualityThreshold || 0.7,
      maxResults: selectedFilters.maxResults || 20,
      sourcePreferences: selectedFilters.sourcePreferences || [],
      timeRange: selectedFilters.timeRange || 'all',
      includeGrayLiterature: selectedFilters.includeGrayLiterature || false,
      contextualExpansion: selectedFilters.contextualExpansion || true
    };

    researchMutation.mutate(searchQuery);
  }, [query, selectedFilters, researchMutation]);

  // Handle result selection
  const handleResultSelect = useCallback((resultId: string) => {
    setSelectedResults(prev => {
      const newSet = new Set(prev);
      if (newSet.has(resultId)) {
        newSet.delete(resultId);
      } else {
        newSet.add(resultId);
      }
      return newSet;
    });
  }, []);

  // Handle save result
  const handleSaveResult = useCallback(async (resultId: string) => {
    if (!activeSession) return;

    try {
      await apiService.saveResearchResult(resultId);
      setActiveSession(prev => prev ? {
        ...prev,
        savedResults: [...prev.savedResults, resultId]
      } : null);
    } catch (error) {
      console.error('Failed to save result:', error);
    }
  }, [activeSession]);

  // Apply smart suggestion
  const applySuggestion = useCallback((suggestion: SmartSuggestion) => {
    switch (suggestion.type) {
      case 'query_expansion':
        setQuery(prev => `${prev} ${suggestion.action}`);
        break;
      case 'source_recommendation':
        setSelectedFilters(prev => ({
          ...prev,
          sourcePreferences: [...(prev.sourcePreferences || []), suggestion.action]
        }));
        break;
      case 'temporal_focus':
        setSelectedFilters(prev => ({
          ...prev,
          timeRange: suggestion.action
        }));
        break;
      default:
        break;
    }
  }, []);

  // Render search interface
  const renderSearchInterface = () => (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        <TextField
          fullWidth
          placeholder="Enter your research question..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          InputProps={{
            endAdornment: (
              <IconButton onClick={handleSearch} disabled={researchMutation.isPending}>
                {researchMutation.isPending ? <CircularProgress size={20} /> : <SearchIcon />}
              </IconButton>
            )
          }}
        />
        <Button
          variant="outlined"
          startIcon={<FilterIcon />}
          onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
        >
          Filters
        </Button>
      </Box>

      {/* Smart Mode Toggle */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <FormControlLabel
          control={
            <Switch
              checked={smartMode}
              onChange={(e) => setSmartMode(e.target.checked)}
            />
          }
          label="AI-Enhanced Search"
        />
        <Typography variant="body2" color="text.secondary">
          {researchMutation.isPending && "🧠 AI analyzing your query..."}
        </Typography>
      </Box>

      {/* Advanced Filters */}
      <Collapse in={showAdvancedFilters}>
        <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1, mb: 2 }}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Domain</InputLabel>
                <Select
                  value={selectedFilters.domain}
                  onChange={(e) => setSelectedFilters(prev => ({ ...prev, domain: e.target.value }))}
                >
                  <MenuItem value="all">All Domains</MenuItem>
                  <MenuItem value="neurosurgery">Neurosurgery</MenuItem>
                  <MenuItem value="cardiology">Cardiology</MenuItem>
                  <MenuItem value="oncology">Oncology</MenuItem>
                  <MenuItem value="general_medicine">General Medicine</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Time Range</InputLabel>
                <Select
                  value={selectedFilters.timeRange}
                  onChange={(e) => setSelectedFilters(prev => ({ ...prev, timeRange: e.target.value }))}
                >
                  <MenuItem value="all">All Time</MenuItem>
                  <MenuItem value="last_year">Last Year</MenuItem>
                  <MenuItem value="last_5_years">Last 5 Years</MenuItem>
                  <MenuItem value="last_10_years">Last 10 Years</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Quality Threshold</Typography>
              <Slider
                value={selectedFilters.qualityThreshold || 0.7}
                onChange={(_, value) => setSelectedFilters(prev => ({ ...prev, qualityThreshold: value as number }))}
                min={0.1}
                max={1.0}
                step={0.1}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Max Results</Typography>
              <Slider
                value={selectedFilters.maxResults || 20}
                onChange={(_, value) => setSelectedFilters(prev => ({ ...prev, maxResults: value as number }))}
                min={5}
                max={100}
                step={5}
                valueLabelDisplay="auto"
              />
            </Grid>

            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={selectedFilters.includeGrayLiterature || false}
                    onChange={(e) => setSelectedFilters(prev => ({ ...prev, includeGrayLiterature: e.target.checked }))}
                  />
                }
                label="Include Gray Literature (Preprints, Conference Papers)"
              />
            </Grid>
          </Grid>
        </Box>
      </Collapse>

      {/* Smart Suggestions */}
      {smartMode && smartSuggestions && smartSuggestions.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            💡 AI Suggestions
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {smartSuggestions.map((suggestion: SmartSuggestion, index: number) => (
              <Chip
                key={index}
                label={suggestion.title}
                onClick={() => applySuggestion(suggestion)}
                icon={<BrainIcon />}
                variant="outlined"
                size="small"
                sx={{ cursor: 'pointer' }}
              />
            ))}
          </Box>
        </Box>
      )}
    </Paper>
  );

  // Render research results
  const renderResults = () => {
    if (!activeSession) return null;

    return (
      <Box ref={resultsRef}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Research Results ({activeSession.results.length})
          </Typography>
          <Box>
            <Button
              startIcon={<DownloadIcon />}
              size="small"
              onClick={() => {/* Handle export */}}
              disabled={selectedResults.size === 0}
            >
              Export Selected
            </Button>
            <Button
              startIcon={<ShareIcon />}
              size="small"
              onClick={() => {/* Handle share */}}
              sx={{ ml: 1 }}
            >
              Share Session
            </Button>
          </Box>
        </Box>

        <Grid container spacing={2}>
          {activeSession.results.map((result: ResearchResult) => (
            <Grid item xs={12} key={result.id}>
              <Card
                sx={{
                  cursor: 'pointer',
                  border: selectedResults.has(result.id) ? 2 : 1,
                  borderColor: selectedResults.has(result.id) ? 'primary.main' : 'divider',
                  '&:hover': { boxShadow: 3 }
                }}
                onClick={() => handleResultSelect(result.id)}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="h6" gutterBottom>
                        {result.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        {result.authors.join(', ')} • {result.journal} • {format(new Date(result.publicationDate), 'MMM yyyy')}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', gap: 1, ml: 2 }}>
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleSaveResult(result.id);
                        }}
                        color={activeSession.savedResults.includes(result.id) ? 'primary' : 'default'}
                      >
                        <SaveIcon />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          window.open(result.url, '_blank');
                        }}
                      >
                        <ViewIcon />
                      </IconButton>
                    </Box>
                  </Box>

                  {/* Quality indicators */}
                  <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                    <Chip
                      label={`Quality: ${(result.qualityScore * 100).toFixed(0)}%`}
                      color={result.qualityScore > 0.8 ? 'success' : result.qualityScore > 0.6 ? 'warning' : 'error'}
                      size="small"
                    />
                    <Chip
                      label={`Relevance: ${(result.relevanceScore * 100).toFixed(0)}%`}
                      color="info"
                      size="small"
                    />
                    <Chip
                      label={result.evidenceLevel}
                      color="primary"
                      size="small"
                    />
                    <Chip
                      label={`${result.citationCount} citations`}
                      variant="outlined"
                      size="small"
                    />
                    {result.accessType === 'open' && (
                      <Chip
                        label="Open Access"
                        color="success"
                        size="small"
                        icon={<CheckIcon />}
                      />
                    )}
                    {result.conflictsPotential > 0.5 && (
                      <Chip
                        label="Potential Conflicts"
                        color="warning"
                        size="small"
                        icon={<WarningIcon />}
                      />
                    )}
                  </Box>

                  {/* Abstract */}
                  <Typography variant="body2" paragraph>
                    {result.abstract.length > 300
                      ? `${result.abstract.substring(0, 300)}...`
                      : result.abstract
                    }
                  </Typography>

                  {/* Key findings */}
                  {result.keyFindings.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Key Findings:
                      </Typography>
                      <List dense>
                        {result.keyFindings.slice(0, 3).map((finding, index) => (
                          <ListItem key={index} sx={{ py: 0 }}>
                            <Typography variant="body2">
                              • {finding}
                            </Typography>
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}

                  {/* Study details */}
                  <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="caption" color="text.secondary">
                          Study Type: {result.studyType}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption" color="text.secondary">
                          Sample Size: {result.sampleSize || 'Not specified'}
                        </Typography>
                      </Grid>
                    </Grid>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };

  // Render session insights
  const renderSessionInsights = () => {
    if (!activeSession || !activeSession.insights.length) return null;

    return (
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Research Insights
        </Typography>
        <List>
          {activeSession.insights.map((insight, index) => (
            <ListItem key={index}>
              <ListItemIcon>
                <BrainIcon color="primary" />
              </ListItemIcon>
              <ListItemText primary={insight} />
            </ListItem>
          ))}
        </List>
      </Paper>
    );
  };

  // Render research trends
  const renderResearchTrends = () => (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Trending Research Topics
      </Typography>
      {researchTrends?.map((trend: any, index: number) => (
        <Box key={index} sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="subtitle2">{trend.topic}</Typography>
            <Chip
              label={`+${trend.growth}%`}
              color="success"
              size="small"
              icon={<TrendIcon />}
            />
          </Box>
          <LinearProgress
            variant="determinate"
            value={trend.popularity}
            sx={{ mt: 1 }}
          />
        </Box>
      ))}
    </Paper>
  );

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Research Assistant
        </Typography>
        <Box>
          <IconButton>
            <SettingsIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Search Interface */}
      {renderSearchInterface()}

      {/* Main Content */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          {/* Session Insights */}
          {renderSessionInsights()}

          {/* Results */}
          {renderResults()}

          {/* Loading State */}
          {researchMutation.isPending && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <Box sx={{ textAlign: 'center' }}>
                <CircularProgress sx={{ mb: 2 }} />
                <Typography variant="body1">
                  🧠 AI is searching and analyzing research papers...
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  This may take a few moments for comprehensive results
                </Typography>
              </Box>
            </Box>
          )}
        </Grid>

        <Grid item xs={12} md={4}>
          {/* Research Trends */}
          {renderResearchTrends()}

          {/* Quick Access */}
          <Paper sx={{ p: 2, mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                fullWidth
                size="small"
              >
                Update Knowledge Base
              </Button>
              <Button
                variant="outlined"
                startIcon={<StarIcon />}
                fullWidth
                size="small"
              >
                View Saved Research
              </Button>
              <Button
                variant="outlined"
                startIcon={<TimerIcon />}
                fullWidth
                size="small"
              >
                Research History
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default IntelligentResearchAssistant;