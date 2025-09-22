/**
 * Loading Screen Component
 * Provides elegant loading states for the KOO Platform
 */

import React from 'react';
import {
  Box,
  CircularProgress,
  Typography,
  Fade,
  LinearProgress,
} from '@mui/material';
import { Psychology as AIIcon } from '@mui/icons-material';

interface LoadingScreenProps {
  message?: string;
  variant?: 'page' | 'component' | 'inline';
  size?: 'small' | 'medium' | 'large';
  showProgress?: boolean;
  progress?: number;
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({
  message = 'Loading KOO Platform...',
  variant = 'page',
  size = 'medium',
  showProgress = false,
  progress,
}) => {
  // Size configurations
  const sizeConfig = {
    small: {
      spinner: 24,
      icon: 32,
      typography: 'body2',
      spacing: 2,
    },
    medium: {
      spinner: 40,
      icon: 48,
      typography: 'h6',
      spacing: 3,
    },
    large: {
      spinner: 60,
      icon: 64,
      typography: 'h5',
      spacing: 4,
    },
  };

  const config = sizeConfig[size];

  // Variant-specific styles
  const getContainerProps = () => {
    switch (variant) {
      case 'page':
        return {
          position: 'fixed' as const,
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 9999,
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(4px)',
        };
      case 'component':
        return {
          position: 'absolute' as const,
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(2px)',
        };
      case 'inline':
        return {
          py: config.spacing,
          px: 2,
        };
      default:
        return {};
    }
  };

  const containerProps = getContainerProps();

  return (
    <Fade in timeout={300}>
      <Box
        {...containerProps}
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        textAlign="center"
      >
        {/* Loading animation container */}
        <Box
          sx={{
            position: 'relative',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mb: config.spacing,
          }}
        >
          {/* Background circle */}
          <CircularProgress
            size={config.spinner + 8}
            thickness={1}
            sx={{
              color: 'grey.300',
              position: 'absolute',
            }}
            variant="determinate"
            value={100}
          />

          {/* Main spinner */}
          <CircularProgress
            size={config.spinner}
            thickness={4}
            sx={{
              color: 'primary.main',
              position: 'relative',
            }}
            {...(showProgress && progress !== undefined
              ? { variant: 'determinate', value: progress }
              : {}
            )}
          />

          {/* Center icon */}
          <Box
            sx={{
              position: 'absolute',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: config.icon,
              height: config.icon,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #1976d2, #dc004e)',
              color: 'white',
              animation: 'pulse 2s ease-in-out infinite',
              '@keyframes pulse': {
                '0%, 100%': {
                  transform: 'scale(1)',
                  opacity: 1,
                },
                '50%': {
                  transform: 'scale(1.05)',
                  opacity: 0.8,
                },
              },
            }}
          >
            <AIIcon sx={{ fontSize: config.icon * 0.6 }} />
          </Box>
        </Box>

        {/* Loading message */}
        <Typography
          variant={config.typography as any}
          color="text.primary"
          sx={{
            fontWeight: 600,
            mb: showProgress ? 1 : 0,
            maxWidth: 300,
          }}
        >
          {message}
        </Typography>

        {/* Progress bar */}
        {showProgress && (
          <Box sx={{ width: 200, mt: 1 }}>
            <LinearProgress
              variant={progress !== undefined ? 'determinate' : 'indeterminate'}
              value={progress}
              sx={{
                height: 6,
                borderRadius: 3,
                backgroundColor: 'grey.200',
                '& .MuiLinearProgress-bar': {
                  background: 'linear-gradient(90deg, #1976d2, #dc004e)',
                  borderRadius: 3,
                },
              }}
            />
            {progress !== undefined && (
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 0.5, display: 'block', textAlign: 'center' }}
              >
                {Math.round(progress)}%
              </Typography>
            )}
          </Box>
        )}

        {/* Subtitle for page loading */}
        {variant === 'page' && (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mt: 1, opacity: 0.8 }}
          >
            Initializing AI-powered medical intelligence...
          </Typography>
        )}
      </Box>
    </Fade>
  );
};

// Preset loading components for common use cases
export const PageLoader: React.FC<{ message?: string }> = ({ message }) => (
  <LoadingScreen variant="page" size="large" message={message} />
);

export const ComponentLoader: React.FC<{ message?: string }> = ({ message }) => (
  <LoadingScreen variant="component" size="medium" message={message} />
);

export const InlineLoader: React.FC<{ message?: string; size?: 'small' | 'medium' }> = ({
  message,
  size = 'small',
}) => (
  <LoadingScreen variant="inline" size={size} message={message} />
);

export const ProgressLoader: React.FC<{
  message?: string;
  progress: number;
  variant?: 'page' | 'component';
}> = ({ message, progress, variant = 'component' }) => (
  <LoadingScreen
    variant={variant}
    size="medium"
    message={message}
    showProgress
    progress={progress}
  />
);

export default LoadingScreen;