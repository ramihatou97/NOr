# Enhanced KOO Platform Frontend Dockerfile
# Multi-stage build for optimized production deployment

# ============================================================================
# Base Node.js Image
# ============================================================================

FROM node:18-alpine as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache \
    git \
    curl \
    && rm -rf /var/cache/apk/*

# ============================================================================
# Dependencies Stage
# ============================================================================

FROM base as dependencies

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Create a separate stage for dev dependencies
FROM base as dev-dependencies

# Copy package files
COPY package.json package-lock.json ./

# Install all dependencies (including dev)
RUN npm ci && npm cache clean --force

# ============================================================================
# Development Stage
# ============================================================================

FROM dev-dependencies as development

# Copy source code
COPY . .

# Expose development port
EXPOSE 3000

# Development command with hot reload
CMD ["npm", "start"]

# ============================================================================
# Build Stage
# ============================================================================

FROM dev-dependencies as builder

# Build arguments for environment configuration
ARG REACT_APP_API_URL=http://localhost:8000
ARG REACT_APP_WS_URL=ws://localhost:8000
ARG REACT_APP_ENVIRONMENT=production
ARG REACT_APP_VERSION=1.0.0
ARG REACT_APP_SENTRY_DSN
ARG REACT_APP_GOOGLE_ANALYTICS_ID

# Set environment variables
ENV REACT_APP_API_URL=$REACT_APP_API_URL \
    REACT_APP_WS_URL=$REACT_APP_WS_URL \
    REACT_APP_ENVIRONMENT=$REACT_APP_ENVIRONMENT \
    REACT_APP_VERSION=$REACT_APP_VERSION \
    REACT_APP_SENTRY_DSN=$REACT_APP_SENTRY_DSN \
    REACT_APP_GOOGLE_ANALYTICS_ID=$REACT_APP_GOOGLE_ANALYTICS_ID \
    GENERATE_SOURCEMAP=false \
    CI=true

# Copy source code
COPY . .

# Build the application
RUN npm run build

# ============================================================================
# Production Stage
# ============================================================================

FROM nginx:alpine as production

# Install additional tools
RUN apk add --no-cache \
    curl \
    && rm -rf /var/cache/apk/*

# Copy built application from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create nginx cache directory
RUN mkdir -p /var/cache/nginx && \
    chown -R nginx:nginx /var/cache/nginx

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]