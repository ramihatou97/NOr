/**
 * Main Application Layout Component
 * Provides consistent navigation and layout structure
 */

import React, { useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  IconButton,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Avatar,
  Menu,
  MenuItem,
  Badge,
  Tooltip,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Psychology as AIIcon,
  Science as ResearchIcon,
  Assessment as QualityIcon,
  Timeline as WorkflowIcon,
  AccountTree as KnowledgeIcon,
  PictureAsPdf as PDFIcon,
  CloudUpload as UploadIcon,
  Notifications as NotificationsIcon,
  Settings as SettingsIcon,
  AccountCircle as AccountIcon,
  Logout as LogoutIcon,
  Help as HelpIcon,
  Brightness4 as ThemeIcon,
  ChevronLeft as ChevronLeftIcon,
} from '@mui/icons-material';

// Constants
const DRAWER_WIDTH = 280;

// Navigation items configuration
const navigationItems = [
  {
    id: 'dashboard',
    label: 'Intelligent Dashboard',
    path: '/',
    icon: <DashboardIcon />,
    description: 'Overview and insights',
  },
  {
    id: 'chapters',
    label: 'Chapter Editor',
    path: '/chapters',
    icon: <AIIcon />,
    description: 'AI-powered content creation',
  },
  {
    id: 'research',
    label: 'Research Assistant',
    path: '/research',
    icon: <ResearchIcon />,
    description: 'Multi-source research tools',
  },
  {
    id: 'quality',
    label: 'Quality Assessment',
    path: '/quality',
    icon: <QualityIcon />,
    description: 'Content quality analysis',
  },
  {
    id: 'workflow',
    label: 'Workflow Optimizer',
    path: '/workflow',
    icon: <WorkflowIcon />,
    description: 'Productivity enhancement',
  },
  {
    id: 'knowledge-graph',
    label: 'Knowledge Graph',
    path: '/knowledge-graph',
    icon: <KnowledgeIcon />,
    description: 'Concept visualization',
  },
  {
    id: 'pdf-processor',
    label: 'PDF Processor',
    path: '/pdf-processor',
    icon: <PDFIcon />,
    description: 'Document processing',
  },
  {
    id: 'upload',
    label: 'Upload Manager',
    path: '/upload',
    icon: <UploadIcon />,
    description: 'File management',
  },
];

const AppLayout: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const location = useLocation();
  const navigate = useNavigate();

  // State management
  const [mobileOpen, setMobileOpen] = useState(false);
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);

  // Handlers
  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };

  const handleNavigation = (path: string) => {
    navigate(path);
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  // Get current page info
  const currentItem = navigationItems.find(item =>
    item.path === location.pathname ||
    (item.path !== '/' && location.pathname.startsWith(item.path))
  ) || navigationItems[0];

  // Drawer content
  const drawerContent = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Logo and title */}
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Box
          sx={{
            width: 40,
            height: 40,
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #1976d2, #dc004e)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
            fontWeight: 'bold',
            fontSize: '1.2rem',
          }}
        >
          KOO
        </Box>
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 700, color: 'primary.main' }}>
            KOO Platform
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Medical Intelligence
          </Typography>
        </Box>
      </Box>

      <Divider />

      {/* Navigation items */}
      <List sx={{ flex: 1, py: 1 }}>
        {navigationItems.map((item) => {
          const isActive = location.pathname === item.path ||
            (item.path !== '/' && location.pathname.startsWith(item.path));

          return (
            <ListItem key={item.id} disablePadding sx={{ px: 1 }}>
              <Tooltip title={item.description} placement="right">
                <ListItemButton
                  onClick={() => handleNavigation(item.path)}
                  selected={isActive}
                  sx={{
                    borderRadius: 2,
                    mb: 0.5,
                    '&.Mui-selected': {
                      backgroundColor: 'primary.main',
                      color: 'primary.contrastText',
                      '&:hover': {
                        backgroundColor: 'primary.dark',
                      },
                      '& .MuiListItemIcon-root': {
                        color: 'primary.contrastText',
                      },
                    },
                    '&:hover': {
                      backgroundColor: 'action.hover',
                    },
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: 40,
                      color: isActive ? 'inherit' : 'text.secondary',
                    }}
                  >
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.label}
                    primaryTypographyProps={{
                      fontSize: '0.875rem',
                      fontWeight: isActive ? 600 : 400,
                    }}
                  />
                </ListItemButton>
              </Tooltip>
            </ListItem>
          );
        })}
      </List>

      <Divider />

      {/* Footer */}
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary" align="center" display="block">
          Version 1.0.0
        </Typography>
        <Typography variant="caption" color="text.secondary" align="center" display="block">
          AI-Powered Medical Platform
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${DRAWER_WIDTH}px)` },
          ml: { md: `${DRAWER_WIDTH}px` },
          backgroundColor: 'background.paper',
          color: 'text.primary',
          boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
        }}
      >
        <Toolbar>
          {/* Mobile menu button */}
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>

          {/* Page title */}
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              {currentItem.label}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {currentItem.description}
            </Typography>
          </Box>

          {/* Header actions */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {/* Notifications */}
            <Tooltip title="Notifications">
              <IconButton color="inherit">
                <Badge badgeContent={3} color="error">
                  <NotificationsIcon />
                </Badge>
              </IconButton>
            </Tooltip>

            {/* Theme toggle */}
            <Tooltip title="Toggle theme">
              <IconButton color="inherit">
                <ThemeIcon />
              </IconButton>
            </Tooltip>

            {/* Help */}
            <Tooltip title="Help & Documentation">
              <IconButton color="inherit">
                <HelpIcon />
              </IconButton>
            </Tooltip>

            {/* User menu */}
            <Tooltip title="Account settings">
              <IconButton
                onClick={handleUserMenuOpen}
                color="inherit"
                sx={{ ml: 1 }}
              >
                <Avatar
                  sx={{
                    width: 32,
                    height: 32,
                    bgcolor: 'primary.main',
                    fontSize: '0.875rem',
                  }}
                >
                  U
                </Avatar>
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      {/* User menu */}
      <Menu
        anchorEl={userMenuAnchor}
        open={Boolean(userMenuAnchor)}
        onClose={handleUserMenuClose}
        onClick={handleUserMenuClose}
        PaperProps={{
          elevation: 8,
          sx: {
            mt: 1.5,
            minWidth: 200,
            '& .MuiMenuItem-root': {
              gap: 2,
            },
          },
        }}
      >
        <MenuItem>
          <AccountIcon fontSize="small" />
          Profile
        </MenuItem>
        <MenuItem>
          <SettingsIcon fontSize="small" />
          Settings
        </MenuItem>
        <Divider />
        <MenuItem>
          <LogoutIcon fontSize="small" />
          Logout
        </MenuItem>
      </Menu>

      {/* Navigation drawer */}
      <Box
        component="nav"
        sx={{ width: { md: DRAWER_WIDTH }, flexShrink: { md: 0 } }}
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile
          }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: DRAWER_WIDTH,
            },
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 1 }}>
            <IconButton onClick={handleDrawerToggle}>
              <ChevronLeftIcon />
            </IconButton>
          </Box>
          <Divider />
          {drawerContent}
        </Drawer>

        {/* Desktop drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: DRAWER_WIDTH,
              borderRight: '1px solid',
              borderColor: 'divider',
            },
          }}
          open
        >
          {drawerContent}
        </Drawer>
      </Box>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${DRAWER_WIDTH}px)` },
          height: '100vh',
          overflow: 'auto',
          backgroundColor: 'background.default',
        }}
      >
        {/* Toolbar spacer */}
        <Toolbar />

        {/* Page content */}
        <Box sx={{ height: 'calc(100vh - 64px)', overflow: 'auto' }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
};

export default AppLayout;