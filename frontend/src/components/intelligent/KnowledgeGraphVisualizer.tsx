/**
 * Knowledge Graph Visualizer Component
 * Interactive visualization of medical knowledge relationships with 3D capabilities
 */

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  Box, Paper, Typography, Grid, Card, CardContent,
  Button, Chip, IconButton, Tabs, Tab, Slider,
  TextField, Autocomplete, FormControl, InputLabel,
  Select, MenuItem, Switch, FormControlLabel,
  Dialog, DialogTitle, DialogContent, DialogActions,
  List, ListItem, ListItemText, ListItemIcon,
  Accordion, AccordionSummary, AccordionDetails,
  Tooltip, Badge, CircularProgress, Alert,
  Divider, Avatar, Fab, Zoom
} from '@mui/material';
import {
  AccountTree as GraphIcon,
  Psychology as BrainIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  Fullscreen as FullscreenIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as CenterIcon,
  Settings as SettingsIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Share as ShareIcon,
  Visibility as ViewIcon,
  VisibilityOff as HideIcon,
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  TrendingUp as TrendIcon,
  Timeline as ConnectionIcon,
  Science as ResearchIcon,
  LocalHospital as MedicalIcon,
  Category as CategoryIcon,
  Link as LinkIcon,
  AutoAwesome as AIIcon,
  ThreeDRotation as ThreeDIcon
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useDebounce } from 'use-debounce';

// Services
import apiService from '../../services/api';

// Types
interface KnowledgeNode {
  id: string;
  name: string;
  type: 'concept' | 'disease' | 'treatment' | 'symptom' | 'medication' | 'procedure';
  category: string;
  confidence: number;
  importance: number;
  lastUpdated: Date;
  sources: string[];
  metadata: Record<string, any>;
  position?: { x: number; y: number; z?: number };
  connections: number;
  evidence_level: string;
}

interface KnowledgeEdge {
  id: string;
  source: string;
  target: string;
  relationship: string;
  strength: number;
  bidirectional: boolean;
  confidence: number;
  evidence: string[];
  type: 'causal' | 'correlational' | 'hierarchical' | 'functional' | 'temporal';
  metadata: Record<string, any>;
}

interface GraphData {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
  clusters: Record<string, KnowledgeNode[]>;
  statistics: {
    totalNodes: number;
    totalEdges: number;
    averageConnections: number;
    strongestConnections: KnowledgeEdge[];
    centralNodes: KnowledgeNode[];
  };
}

interface GraphFilter {
  nodeTypes: string[];
  relationshipTypes: string[];
  minimumConfidence: number;
  minimumStrength: number;
  timeRange: string;
  categories: string[];
  evidenceLevels: string[];
}

interface GraphLayout {
  algorithm: 'force-directed' | 'hierarchical' | 'circular' | 'grid' | 'cluster';
  is3D: boolean;
  nodeSize: number;
  edgeThickness: number;
  animation: boolean;
  physics: {
    enabled: boolean;
    gravity: number;
    repulsion: number;
    springLength: number;
  };
}

const KnowledgeGraphVisualizer: React.FC = () => {
  // State
  const [selectedTab, setSelectedTab] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNode, setSelectedNode] = useState<KnowledgeNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<KnowledgeEdge | null>(null);
  const [showNodeDetails, setShowNodeDetails] = useState(false);
  const [showEdgeDetails, setShowEdgeDetails] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [graphFilters, setGraphFilters] = useState<GraphFilter>({
    nodeTypes: [],
    relationshipTypes: [],
    minimumConfidence: 0.5,
    minimumStrength: 0.3,
    timeRange: 'all',
    categories: [],
    evidenceLevels: []
  });
  const [graphLayout, setGraphLayout] = useState<GraphLayout>({
    algorithm: 'force-directed',
    is3D: false,
    nodeSize: 1.0,
    edgeThickness: 1.0,
    animation: true,
    physics: {
      enabled: true,
      gravity: 0.1,
      repulsion: 100,
      springLength: 100
    }
  });

  // Refs
  const graphContainerRef = useRef<HTMLDivElement>(null);
  const graphInstanceRef = useRef<any>(null);

  // Debounced search
  const [debouncedSearch] = useDebounce(searchQuery, 500);

  const queryClient = useQueryClient();

  // Knowledge graph data query
  const {
    data: graphData,
    isLoading: graphLoading,
    error: graphError
  } = useQuery({
    queryKey: ['knowledge-graph', debouncedSearch, graphFilters],
    queryFn: () => apiService.getKnowledgeGraph({
      search: debouncedSearch,
      filters: graphFilters
    }),
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  // Node suggestions query
  const { data: nodeSuggestions } = useQuery({
    queryKey: ['node-suggestions', debouncedSearch],
    queryFn: () => apiService.getNodeSuggestions(debouncedSearch),
    enabled: debouncedSearch.length > 2,
  });

  // Graph analytics query
  const { data: graphAnalytics } = useQuery({
    queryKey: ['graph-analytics'],
    queryFn: () => apiService.getGraphAnalytics(),
  });

  // Available node types and categories
  const { data: graphSchema } = useQuery({
    queryKey: ['graph-schema'],
    queryFn: () => apiService.getGraphSchema(),
  });

  // Initialize graph visualization
  useEffect(() => {
    if (graphData && graphContainerRef.current) {
      initializeGraph();
    }
  }, [graphData, graphLayout]);

  // Initialize graph with vis.js or similar library
  const initializeGraph = useCallback(() => {
    if (!graphData || !graphContainerRef.current) return;

    // This would use a graph visualization library like vis.js, D3.js, or cytoscape.js
    // For now, we'll simulate the initialization
    console.log('Initializing graph with', graphData.nodes.length, 'nodes and', graphData.edges.length, 'edges');

    // Store graph instance for later manipulation
    graphInstanceRef.current = {
      // Graph instance methods would go here
      centerView: () => console.log('Centering view'),
      zoomIn: () => console.log('Zooming in'),
      zoomOut: () => console.log('Zooming out'),
      focusNode: (nodeId: string) => console.log('Focusing on node', nodeId),
      exportImage: () => console.log('Exporting image'),
      updateLayout: (layout: string) => console.log('Updating layout to', layout)
    };
  }, [graphData, graphLayout]);

  // Handle node selection
  const handleNodeSelect = useCallback((node: KnowledgeNode) => {
    setSelectedNode(node);
    setShowNodeDetails(true);
    graphInstanceRef.current?.focusNode(node.id);
  }, []);

  // Handle edge selection
  const handleEdgeSelect = useCallback((edge: KnowledgeEdge) => {
    setSelectedEdge(edge);
    setShowEdgeDetails(true);
  }, []);

  // Handle search
  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);
  }, []);

  // Apply filters
  const applyFilters = useCallback((newFilters: Partial<GraphFilter>) => {
    setGraphFilters(prev => ({ ...prev, ...newFilters }));
  }, []);

  // Update layout
  const updateLayout = useCallback((newLayout: Partial<GraphLayout>) => {
    setGraphLayout(prev => ({ ...prev, ...newLayout }));
  }, []);

  // Graph control functions
  const centerView = () => graphInstanceRef.current?.centerView();
  const zoomIn = () => graphInstanceRef.current?.zoomIn();
  const zoomOut = () => graphInstanceRef.current?.zoomOut();
  const exportGraph = () => graphInstanceRef.current?.exportImage();

  // Get node color based on type
  const getNodeColor = useCallback((nodeType: string) => {
    const colorMap: Record<string, string> = {
      concept: '#4CAF50',
      disease: '#F44336',
      treatment: '#2196F3',
      symptom: '#FF9800',
      medication: '#9C27B0',
      procedure: '#00BCD4'
    };
    return colorMap[nodeType] || '#757575';
  }, []);

  // Get relationship color based on strength
  const getRelationshipColor = useCallback((strength: number) => {
    if (strength > 0.8) return '#4CAF50';
    if (strength > 0.6) return '#FF9800';
    if (strength > 0.4) return '#FFC107';
    return '#757575';
  }, []);

  // Render graph visualization area
  const renderGraphVisualization = () => (
    <Paper
      sx={{
        height: isFullscreen ? '100vh' : 600,
        position: isFullscreen ? 'fixed' : 'relative',
        top: isFullscreen ? 0 : 'auto',
        left: isFullscreen ? 0 : 'auto',
        right: isFullscreen ? 0 : 'auto',
        bottom: isFullscreen ? 0 : 'auto',
        zIndex: isFullscreen ? 9999 : 1,
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      {/* Graph Controls */}
      <Box sx={{ p: 1, borderBottom: 1, borderColor: 'divider', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Center View">
            <IconButton size="small" onClick={centerView}>
              <CenterIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom In">
            <IconButton size="small" onClick={zoomIn}>
              <ZoomInIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom Out">
            <IconButton size="small" onClick={zoomOut}>
              <ZoomOutIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Toggle 3D">
            <IconButton
              size="small"
              color={graphLayout.is3D ? 'primary' : 'default'}
              onClick={() => updateLayout({ is3D: !graphLayout.is3D })}
            >
              <ThreeDIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Export">
            <IconButton size="small" onClick={exportGraph}>
              <DownloadIcon />
            </IconButton>
          </Tooltip>
        </Box>

        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip
            label={`${graphData?.nodes.length || 0} nodes`}
            size="small"
            icon={<GraphIcon />}
          />
          <Chip
            label={`${graphData?.edges.length || 0} connections`}
            size="small"
            icon={<LinkIcon />}
          />
          <Tooltip title="Fullscreen">
            <IconButton size="small" onClick={() => setIsFullscreen(!isFullscreen)}>
              <FullscreenIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Graph Container */}
      <Box
        ref={graphContainerRef}
        sx={{
          flexGrow: 1,
          bgcolor: 'grey.50',
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        {graphLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <CircularProgress size={60} />
            <Box sx={{ ml: 2 }}>
              <Typography variant="h6">Loading Knowledge Graph...</Typography>
              <Typography variant="body2" color="text.secondary">
                Analyzing {graphData?.nodes.length || 0} concepts and their relationships
              </Typography>
            </Box>
          </Box>
        ) : graphError ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Alert severity="error">Failed to load knowledge graph</Alert>
          </Box>
        ) : (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography variant="h6" color="text.secondary">
              Interactive Knowledge Graph Visualization
            </Typography>
          </Box>
        )}

        {/* Graph Legend */}
        <Box sx={{ position: 'absolute', top: 16, left: 16, bgcolor: 'white', p: 2, borderRadius: 1, boxShadow: 2 }}>
          <Typography variant="subtitle2" gutterBottom>Legend</Typography>
          {Object.entries({
            concept: 'Concepts',
            disease: 'Diseases',
            treatment: 'Treatments',
            symptom: 'Symptoms',
            medication: 'Medications',
            procedure: 'Procedures'
          }).map(([type, label]) => (
            <Box key={type} sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  bgcolor: getNodeColor(type)
                }}
              />
              <Typography variant="caption">{label}</Typography>
            </Box>
          ))}
        </Box>
      </Box>
    </Paper>
  );

  // Render search and filters
  const renderSearchAndFilters = () => (
    <Paper sx={{ p: 2, mb: 3 }}>
      <Grid container spacing={2} alignItems="center">
        <Grid item xs={12} md={4}>
          <Autocomplete
            options={nodeSuggestions || []}
            value={searchQuery}
            onInputChange={(_, value) => handleSearch(value)}
            renderInput={(params) => (
              <TextField
                {...params}
                placeholder="Search knowledge graph..."
                InputProps={{
                  ...params.InputProps,
                  startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
                }}
              />
            )}
          />
        </Grid>

        <Grid item xs={12} md={8}>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Node Types</InputLabel>
              <Select
                multiple
                value={graphFilters.nodeTypes}
                onChange={(e) => applyFilters({ nodeTypes: e.target.value as string[] })}
              >
                {graphSchema?.nodeTypes?.map((type: string) => (
                  <MenuItem key={type} value={type}>{type}</MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Layout</InputLabel>
              <Select
                value={graphLayout.algorithm}
                onChange={(e) => updateLayout({ algorithm: e.target.value as any })}
              >
                <MenuItem value="force-directed">Force Directed</MenuItem>
                <MenuItem value="hierarchical">Hierarchical</MenuItem>
                <MenuItem value="circular">Circular</MenuItem>
                <MenuItem value="cluster">Clustered</MenuItem>
              </Select>
            </FormControl>

            <Tooltip title="Minimum Confidence">
              <Box sx={{ width: 150 }}>
                <Typography variant="caption">Confidence: {(graphFilters.minimumConfidence * 100).toFixed(0)}%</Typography>
                <Slider
                  size="small"
                  value={graphFilters.minimumConfidence}
                  onChange={(_, value) => applyFilters({ minimumConfidence: value as number })}
                  min={0}
                  max={1}
                  step={0.1}
                />
              </Box>
            </Tooltip>

            <FormControlLabel
              control={
                <Switch
                  checked={graphLayout.animation}
                  onChange={(e) => updateLayout({ animation: e.target.checked })}
                />
              }
              label="Animation"
            />
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );

  // Render graph statistics
  const renderGraphStatistics = () => {
    if (!graphData?.statistics) return null;

    return (
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <GraphIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="primary">
                {graphData.statistics.totalNodes}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Concepts
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <ConnectionIcon color="info" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="info.main">
                {graphData.statistics.totalEdges}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Relationships
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <TrendIcon color="success" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="success.main">
                {graphData.statistics.averageConnections.toFixed(1)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Connections
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <BrainIcon color="secondary" sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h4" color="secondary.main">
                {Object.keys(graphData.clusters).length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Knowledge Clusters
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  // Render central nodes
  const renderCentralNodes = () => {
    if (!graphData?.statistics.centralNodes?.length) return null;

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Most Connected Concepts
        </Typography>
        <List>
          {graphData.statistics.centralNodes.slice(0, 10).map((node: KnowledgeNode) => (
            <ListItem
              key={node.id}
              button
              onClick={() => handleNodeSelect(node)}
              sx={{ borderRadius: 1, mb: 1 }}
            >
              <ListItemIcon>
                <Avatar
                  sx={{
                    bgcolor: getNodeColor(node.type),
                    width: 32,
                    height: 32,
                    fontSize: '0.75rem'
                  }}
                >
                  {node.name.charAt(0).toUpperCase()}
                </Avatar>
              </ListItemIcon>
              <ListItemText
                primary={node.name}
                secondary={
                  <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                    <Chip label={node.type} size="small" />
                    <Chip
                      label={`${node.connections} connections`}
                      size="small"
                      variant="outlined"
                    />
                    <Chip
                      label={`${(node.confidence * 100).toFixed(0)}% confidence`}
                      size="small"
                      color="primary"
                    />
                  </Box>
                }
              />
            </ListItem>
          ))}
        </List>
      </Paper>
    );
  };

  // Render knowledge clusters
  const renderKnowledgeClusters = () => {
    if (!graphData?.clusters) return null;

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Knowledge Clusters
        </Typography>
        {Object.entries(graphData.clusters).map(([clusterName, nodes]) => (
          <Accordion key={clusterName}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CategoryIcon color="primary" />
                <Typography variant="subtitle2">{clusterName}</Typography>
                <Chip label={`${nodes.length} concepts`} size="small" />
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={1}>
                {nodes.slice(0, 12).map((node: KnowledgeNode) => (
                  <Grid item xs={6} sm={4} md={3} key={node.id}>
                    <Chip
                      label={node.name}
                      size="small"
                      clickable
                      onClick={() => handleNodeSelect(node)}
                      sx={{ width: '100%' }}
                    />
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        ))}
      </Paper>
    );
  };

  // Node details dialog
  const renderNodeDetailsDialog = () => (
    <Dialog
      open={showNodeDetails}
      onClose={() => setShowNodeDetails(false)}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Avatar sx={{ bgcolor: getNodeColor(selectedNode?.type || '') }}>
            {selectedNode?.name.charAt(0).toUpperCase()}
          </Avatar>
          <Box>
            <Typography variant="h6">{selectedNode?.name}</Typography>
            <Typography variant="body2" color="text.secondary">
              {selectedNode?.type} • {selectedNode?.category}
            </Typography>
          </Box>
        </Box>
      </DialogTitle>
      <DialogContent>
        {selectedNode && (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>Properties</Typography>
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="Confidence"
                    secondary={`${(selectedNode.confidence * 100).toFixed(1)}%`}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Importance"
                    secondary={`${(selectedNode.importance * 100).toFixed(1)}%`}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Connections"
                    secondary={selectedNode.connections}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Evidence Level"
                    secondary={selectedNode.evidence_level}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Last Updated"
                    secondary={new Date(selectedNode.lastUpdated).toLocaleDateString()}
                  />
                </ListItem>
              </List>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>Sources</Typography>
              <List dense>
                {selectedNode.sources.slice(0, 5).map((source, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <ResearchIcon fontSize="small" />
                    </ListItemIcon>
                    <ListItemText primary={source} />
                  </ListItem>
                ))}
              </List>
            </Grid>
          </Grid>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setShowNodeDetails(false)}>Close</Button>
        <Button variant="contained" startIcon={<ViewIcon />}>
          Explore Connections
        </Button>
      </DialogActions>
    </Dialog>
  );

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Knowledge Graph Visualizer
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

      {/* Search and Filters */}
      {renderSearchAndFilters()}

      {/* Statistics */}
      {renderGraphStatistics()}

      {/* Main Content */}
      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          {/* Graph Visualization */}
          {renderGraphVisualization()}
        </Grid>

        <Grid item xs={12} lg={4}>
          {/* Tabs for side panel */}
          <Box sx={{ mb: 2 }}>
            <Tabs value={selectedTab} onChange={(_, newValue) => setSelectedTab(newValue)}>
              <Tab icon={<GraphIcon />} label="Nodes" />
              <Tab icon={<CategoryIcon />} label="Clusters" />
              <Tab icon={<AnalyticsIcon />} label="Analytics" />
            </Tabs>
          </Box>

          {/* Tab Content */}
          {selectedTab === 0 && renderCentralNodes()}
          {selectedTab === 1 && renderKnowledgeClusters()}
          {selectedTab === 2 && (
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6">Graph Analytics</Typography>
              <Typography variant="body2" color="text.secondary">
                Advanced analytics coming soon...
              </Typography>
            </Paper>
          )}
        </Grid>
      </Grid>

      {/* Dialogs */}
      {renderNodeDetailsDialog()}

      {/* Floating Action Button for AI Suggestions */}
      <Zoom in={!isFullscreen}>
        <Fab
          color="primary"
          sx={{ position: 'fixed', bottom: 24, right: 24 }}
          onClick={() => {/* Open AI suggestions */}}
        >
          <AIIcon />
        </Fab>
      </Zoom>
    </Box>
  );
};

export default KnowledgeGraphVisualizer;