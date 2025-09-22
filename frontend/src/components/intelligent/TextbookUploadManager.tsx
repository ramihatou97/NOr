/**
 * Textbook Upload Manager Component
 * Advanced PDF textbook upload, processing, and management interface
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box, Paper, Typography, Grid, Card, CardContent,
  Button, LinearProgress, Alert, Dialog, DialogTitle,
  DialogContent, DialogActions, TextField, FormControl,
  InputLabel, Select, MenuItem, Chip, List, ListItem,
  ListItemText, ListItemIcon, ListItemSecondaryAction,
  IconButton, Accordion, AccordionSummary, AccordionDetails,
  Tooltip, Badge, CircularProgress, Divider, Stepper,
  Step, StepLabel, StepContent, Tabs, Tab, Table,
  TableBody, TableCell, TableContainer, TableHead, TableRow
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  PictureAsPdf as PdfIcon,
  AutoAwesome as ProcessingIcon,
  CheckCircle as CompleteIcon,
  Error as ErrorIcon,
  Search as SearchIcon,
  Visibility as ViewIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  MenuBook as BookIcon,
  Psychology as AIIcon,
  Timeline as ProgressIcon,
  Assessment as QualityIcon,
  Science as MedicalIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';

// Services
import apiService from '../../services/api';

// Types
interface TextbookMetadata {
  title: string;
  authors: string[];
  edition: string;
  publisher: string;
  publicationYear: number;
  isbn?: string;
  specialty: string;
  priority: number;
  description?: string;
}

interface ProcessingStatus {
  stage: string;
  progress: number;
  currentOperation: string;
  estimatedTimeRemaining: number;
  pagesProcessed: number;
  totalPages: number;
  errors: string[];
  warnings: string[];
}

interface ProcessedTextbook {
  textbookId: string;
  metadata: TextbookMetadata;
  processingTimestamp: Date;
  totalContentBlocks: number;
  chaptersCount: number;
  processingStatus: 'completed' | 'processing' | 'failed';
  qualityScore: number;
  fileSize: number;
  extractedEntities: {
    medicalConcepts: number;
    anatomicalReferences: number;
    procedures: number;
    pathologies: number;
  };
}

interface ChapterSearchResult {
  contentId: string;
  title: string;
  content: string;
  pageNumbers: number[];
  similarityScore: number;
  contentType: string;
  qualityScore: number;
}

const TextbookUploadManager: React.FC = () => {
  // State
  const [selectedTab, setSelectedTab] = useState(0);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [metadataDialogOpen, setMetadataDialogOpen] = useState(false);
  const [searchDialogOpen, setSearchDialogOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<TextbookMetadata>({
    title: '',
    authors: [],
    edition: '',
    publisher: '',
    publicationYear: new Date().getFullYear(),
    specialty: 'neurosurgery',
    priority: 5,
    description: ''
  });
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTextbook, setSelectedTextbook] = useState<string>('');
  const [searchResults, setSearchResults] = useState<ChapterSearchResult[]>([]);

  const queryClient = useQueryClient();
  const processingRef = useRef<WebSocket | null>(null);

  // Textbooks query
  const { data: textbooks, isLoading: textbooksLoading } = useQuery({
    queryKey: ['processed-textbooks'],
    queryFn: () => apiService.getProcessedTextbooks(),
  });

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: async (data: { file: File; metadata: TextbookMetadata }) => {
      const formData = new FormData();
      formData.append('file', data.file);
      formData.append('metadata', JSON.stringify(data.metadata));
      return apiService.uploadTextbook(formData);
    },
    onSuccess: (data) => {
      setUploadDialogOpen(false);
      setMetadataDialogOpen(false);
      setSelectedFile(null);

      // Start monitoring processing status
      startProcessingMonitor(data.processingId);

      queryClient.invalidateQueries({ queryKey: ['processed-textbooks'] });
    },
  });

  // Search mutation
  const searchMutation = useMutation({
    mutationFn: (query: { textbookId: string; searchQuery: string; options?: any }) =>
      apiService.searchWithinTextbook(query.textbookId, query.searchQuery, query.options),
    onSuccess: (data) => {
      setSearchResults(data.results);
    },
  });

  // Dropzone configuration
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setSelectedFile(acceptedFiles[0]);
        setUploadDialogOpen(true);
      }
    }
  });

  // Start processing monitor
  const startProcessingMonitor = useCallback((processingId: string) => {
    const wsUrl = `ws://localhost:8000/ws/textbook-processing/${processingId}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProcessingStatus(data);

      if (data.stage === 'completed' || data.stage === 'failed') {
        ws.close();
        queryClient.invalidateQueries({ queryKey: ['processed-textbooks'] });
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    processingRef.current = ws;
  }, [queryClient]);

  // Handle file upload
  const handleUpload = useCallback(() => {
    if (selectedFile && metadata.title) {
      uploadMutation.mutate({ file: selectedFile, metadata });
    }
  }, [selectedFile, metadata, uploadMutation]);

  // Handle search
  const handleSearch = useCallback(() => {
    if (selectedTextbook && searchQuery) {
      searchMutation.mutate({
        textbookId: selectedTextbook,
        searchQuery,
        options: {
          maxResults: 20,
          contentTypes: ['chapter', 'section', 'figure', 'table'],
          minQualityScore: 0.5
        }
      });
    }
  }, [selectedTextbook, searchQuery, searchMutation]);

  // Handle metadata field changes
  const handleMetadataChange = useCallback((field: string, value: any) => {
    setMetadata(prev => ({ ...prev, [field]: value }));
  }, []);

  // Add author
  const addAuthor = useCallback(() => {
    const authorName = prompt('Enter author name:');
    if (authorName?.trim()) {
      setMetadata(prev => ({
        ...prev,
        authors: [...prev.authors, authorName.trim()]
      }));
    }
  }, []);

  // Remove author
  const removeAuthor = useCallback((index: number) => {
    setMetadata(prev => ({
      ...prev,
      authors: prev.authors.filter((_, i) => i !== index)
    }));
  }, []);

  // Render upload area
  const renderUploadArea = () => (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box
          {...getRootProps()}
          sx={{
            border: 2,
            borderStyle: 'dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.300',
            borderRadius: 2,
            p: 4,
            textAlign: 'center',
            cursor: 'pointer',
            bgcolor: isDragActive ? 'primary.light' : 'grey.50',
            transition: 'all 0.2s ease'
          }}
        >
          <input {...getInputProps()} />
          <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            {isDragActive ? 'Drop PDF here...' : 'Upload Medical Textbook (PDF)'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Drag and drop a PDF file here, or click to select
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Supported: PDF files up to 500MB
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );

  // Render textbook list
  const renderTextbookList = () => (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Processed Textbooks
      </Typography>

      {textbooksLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Title</TableCell>
                <TableCell>Authors</TableCell>
                <TableCell>Specialty</TableCell>
                <TableCell>Chapters</TableCell>
                <TableCell>Quality</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {textbooks?.map((textbook: ProcessedTextbook) => (
                <TableRow key={textbook.textbookId}>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <BookIcon color="primary" />
                      <Box>
                        <Typography variant="subtitle2">
                          {textbook.metadata.title}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {textbook.metadata.edition} ({textbook.metadata.publicationYear})
                        </Typography>
                      </Box>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {textbook.metadata.authors.slice(0, 2).join(', ')}
                      {textbook.metadata.authors.length > 2 && ` +${textbook.metadata.authors.length - 2} more`}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={textbook.metadata.specialty}
                      color="primary"
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {textbook.chaptersCount} chapters
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {textbook.totalContentBlocks} content blocks
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={textbook.qualityScore * 100}
                        sx={{ width: 60, height: 6, borderRadius: 3 }}
                        color={textbook.qualityScore > 0.8 ? 'success' : textbook.qualityScore > 0.6 ? 'warning' : 'error'}
                      />
                      <Typography variant="caption">
                        {(textbook.qualityScore * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={textbook.processingStatus}
                      color={
                        textbook.processingStatus === 'completed' ? 'success' :
                        textbook.processingStatus === 'processing' ? 'warning' : 'error'
                      }
                      size="small"
                      icon={
                        textbook.processingStatus === 'completed' ? <CompleteIcon /> :
                        textbook.processingStatus === 'processing' ? <ProcessingIcon /> : <ErrorIcon />
                      }
                    />
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <Tooltip title="Search Within">
                        <IconButton
                          size="small"
                          onClick={() => {
                            setSelectedTextbook(textbook.textbookId);
                            setSearchDialogOpen(true);
                          }}
                        >
                          <SearchIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="View Details">
                        <IconButton size="small">
                          <ViewIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton size="small" color="error">
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Paper>
  );

  // Render processing status
  const renderProcessingStatus = () => {
    if (!processingStatus) return null;

    return (
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Processing Status
        </Typography>

        <Stepper activeStep={getProcessingStep(processingStatus.stage)} orientation="vertical">
          <Step>
            <StepLabel>PDF Structure Analysis</StepLabel>
            <StepContent>
              <Typography variant="body2">
                Analyzing PDF layout, fonts, and structure patterns
              </Typography>
            </StepContent>
          </Step>
          <Step>
            <StepLabel>Chapter Detection</StepLabel>
            <StepContent>
              <Typography variant="body2">
                Identifying chapter boundaries and hierarchical structure
              </Typography>
            </StepContent>
          </Step>
          <Step>
            <StepLabel>Content Extraction</StepLabel>
            <StepContent>
              <Typography variant="body2">
                Extracting text, images, tables, and figures
              </Typography>
              <LinearProgress
                variant="determinate"
                value={(processingStatus.pagesProcessed / processingStatus.totalPages) * 100}
                sx={{ mt: 1 }}
              />
              <Typography variant="caption">
                {processingStatus.pagesProcessed} / {processingStatus.totalPages} pages
              </Typography>
            </StepContent>
          </Step>
          <Step>
            <StepLabel>Medical Entity Recognition</StepLabel>
            <StepContent>
              <Typography variant="body2">
                Identifying medical concepts, anatomy, procedures, and pathologies
              </Typography>
            </StepContent>
          </Step>
          <Step>
            <StepLabel>Semantic Indexing</StepLabel>
            <StepContent>
              <Typography variant="body2">
                Creating semantic embeddings and search indexes
              </Typography>
            </StepContent>
          </Step>
          <Step>
            <StepLabel>Quality Assessment</StepLabel>
            <StepContent>
              <Typography variant="body2">
                Evaluating content quality and clinical relevance
              </Typography>
            </StepContent>
          </Step>
        </Stepper>

        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" gutterBottom>
            Current Operation: {processingStatus.currentOperation}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Estimated time remaining: {Math.ceil(processingStatus.estimatedTimeRemaining / 60)} minutes
          </Typography>
        </Box>

        {processingStatus.errors.length > 0 && (
          <Alert severity="error" sx={{ mt: 2 }}>
            Processing errors: {processingStatus.errors.join(', ')}
          </Alert>
        )}
      </Paper>
    );
  };

  // Render search results
  const renderSearchResults = () => (
    <List>
      {searchResults.map((result, index) => (
        <ListItem key={index} sx={{ alignItems: 'flex-start', mb: 1, border: 1, borderColor: 'divider', borderRadius: 1 }}>
          <ListItemIcon>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <MedicalIcon color="primary" />
              <Typography variant="caption">
                {(result.similarityScore * 100).toFixed(0)}%
              </Typography>
            </Box>
          </ListItemIcon>
          <ListItemText
            primary={
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="subtitle2">{result.title}</Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Chip label={result.contentType} size="small" />
                  <Chip
                    label={`Pages ${result.pageNumbers.join(', ')}`}
                    size="small"
                    variant="outlined"
                  />
                </Box>
              </Box>
            }
            secondary={
              <Box>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {result.content.length > 200
                    ? `${result.content.substring(0, 200)}...`
                    : result.content
                  }
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                  <Chip
                    label={`Quality: ${(result.qualityScore * 100).toFixed(0)}%`}
                    size="small"
                    color={result.qualityScore > 0.8 ? 'success' : 'warning'}
                  />
                  <Chip
                    label={`Match: ${(result.similarityScore * 100).toFixed(0)}%`}
                    size="small"
                    color="info"
                  />
                </Box>
              </Box>
            }
          />
          <ListItemSecondaryAction>
            <IconButton size="small">
              <ViewIcon />
            </IconButton>
          </ListItemSecondaryAction>
        </ListItem>
      ))}
    </List>
  );

  // Helper function
  const getProcessingStep = (stage: string): number => {
    const stages = ['structure_analysis', 'chapter_detection', 'content_extraction', 'medical_ner', 'semantic_indexing', 'quality_assessment'];
    return stages.indexOf(stage);
  };

  // Render metadata dialog
  const renderMetadataDialog = () => (
    <Dialog open={metadataDialogOpen} onClose={() => setMetadataDialogOpen(false)} maxWidth="md" fullWidth>
      <DialogTitle>Textbook Metadata</DialogTitle>
      <DialogContent>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Title"
              value={metadata.title}
              onChange={(e) => handleMetadataChange('title', e.target.value)}
              required
            />
          </Grid>

          <Grid item xs={12}>
            <Box>
              <Typography variant="subtitle2" gutterBottom>Authors</Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1 }}>
                {metadata.authors.map((author, index) => (
                  <Chip
                    key={index}
                    label={author}
                    onDelete={() => removeAuthor(index)}
                    size="small"
                  />
                ))}
              </Box>
              <Button size="small" onClick={addAuthor}>
                Add Author
              </Button>
            </Box>
          </Grid>

          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Edition"
              value={metadata.edition}
              onChange={(e) => handleMetadataChange('edition', e.target.value)}
            />
          </Grid>

          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Publisher"
              value={metadata.publisher}
              onChange={(e) => handleMetadataChange('publisher', e.target.value)}
            />
          </Grid>

          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Publication Year"
              type="number"
              value={metadata.publicationYear}
              onChange={(e) => handleMetadataChange('publicationYear', parseInt(e.target.value))}
            />
          </Grid>

          <Grid item xs={6}>
            <FormControl fullWidth>
              <InputLabel>Specialty</InputLabel>
              <Select
                value={metadata.specialty}
                onChange={(e) => handleMetadataChange('specialty', e.target.value)}
              >
                <MenuItem value="neurosurgery">Neurosurgery</MenuItem>
                <MenuItem value="anatomy">Anatomy</MenuItem>
                <MenuItem value="radiology">Radiology</MenuItem>
                <MenuItem value="pathology">Pathology</MenuItem>
                <MenuItem value="general_medicine">General Medicine</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12}>
            <Typography gutterBottom>Priority Level: {metadata.priority}</Typography>
            <Box sx={{ px: 2 }}>
              <input
                type="range"
                min="1"
                max="10"
                value={metadata.priority}
                onChange={(e) => handleMetadataChange('priority', parseInt(e.target.value))}
                style={{ width: '100%' }}
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              Higher priority textbooks are searched first
            </Typography>
          </Grid>

          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Description"
              multiline
              rows={3}
              value={metadata.description}
              onChange={(e) => handleMetadataChange('description', e.target.value)}
              placeholder="Brief description of the textbook content..."
            />
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setMetadataDialogOpen(false)}>Cancel</Button>
        <Button
          variant="contained"
          onClick={handleUpload}
          disabled={!metadata.title || uploadMutation.isPending}
        >
          {uploadMutation.isPending ? 'Processing...' : 'Upload & Process'}
        </Button>
      </DialogActions>
    </Dialog>
  );

  // Render search dialog
  const renderSearchDialog = () => (
    <Dialog open={searchDialogOpen} onClose={() => setSearchDialogOpen(false)} maxWidth="lg" fullWidth>
      <DialogTitle>Search Within Textbook</DialogTitle>
      <DialogContent>
        <Box sx={{ mb: 2, display: 'flex', gap: 2 }}>
          <TextField
            fullWidth
            placeholder="Search for medical topics, procedures, anatomy..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          />
          <Button
            variant="contained"
            startIcon={<SearchIcon />}
            onClick={handleSearch}
            disabled={!searchQuery || searchMutation.isPending}
          >
            Search
          </Button>
        </Box>

        {searchMutation.isPending && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
            <CircularProgress />
          </Box>
        )}

        {searchResults.length > 0 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Search Results ({searchResults.length})
            </Typography>
            {renderSearchResults()}
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setSearchDialogOpen(false)}>Close</Button>
      </DialogActions>
    </Dialog>
  );

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Textbook Management
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={() => queryClient.invalidateQueries({ queryKey: ['processed-textbooks'] })}
        >
          Refresh
        </Button>
      </Box>

      {/* Upload Area */}
      {renderUploadArea()}

      {/* Processing Status */}
      {renderProcessingStatus()}

      {/* Textbook List */}
      {renderTextbookList()}

      {/* Dialogs */}
      {renderMetadataDialog()}
      {renderSearchDialog()}

      {/* Upload confirmation dialog */}
      <Dialog open={uploadDialogOpen} onClose={() => setUploadDialogOpen(false)}>
        <DialogTitle>Upload Textbook</DialogTitle>
        <DialogContent>
          {selectedFile && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Selected File
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                <PdfIcon color="error" sx={{ fontSize: 40 }} />
                <Box>
                  <Typography variant="subtitle2">{selectedFile.name}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </Typography>
                </Box>
              </Box>
              <Typography variant="body2" sx={{ mt: 2 }}>
                Please provide metadata for this textbook to enable intelligent processing and search.
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => setMetadataDialogOpen(true)}
          >
            Add Metadata
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default TextbookUploadManager;