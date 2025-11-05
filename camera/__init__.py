#!/usr/bin/env python3
"""
Camera and computer vision modules for CyberCrawl Spider Robot
"""

from .camera_yolo import EnhancedCameraYOLO

# Backward compatibility alias
CameraYOLO = EnhancedCameraYOLO

__all__ = ['EnhancedCameraYOLO', 'CameraYOLO']
