"""
Configuration loader for detector configurations.

This module provides functionality to load detector configurations from Python files
and build detector objects based on those configurations.
"""

import importlib.util
import sys
import os
from typing import Dict, Any, Optional
from .build_larcube import build_detector


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load a configuration from a Python file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    # Get the absolute path
    abs_path = os.path.abspath(config_path)
    
    # Extract the module name from the file path
    module_name = os.path.splitext(os.path.basename(abs_path))[0]
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load configuration file: {config_path}")
        
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = config_module
    spec.loader.exec_module(config_module)
    
    # Check if the module has a get_config function
    if not hasattr(config_module, "get_config"):
        raise AttributeError(f"Configuration file {config_path} must define a get_config() function")
    
    # Get the configuration dictionary
    config = config_module.get_config()
    
    return config


def build_detector_from_config(config_path: str, **kwargs):
    """
    Build a detector from a configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
    auto_build_bvh : bool, optional
        Whether to auto-build the BVH for the detector
    **kwargs
        Additional parameters to override configuration values
        
    Returns
    -------
    detector.Detector
        Detector object
    """
    # Load the configuration
    try:
        _config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", config_path+'.py')
        
        config = load_config_from_file(_config_path)
    except Exception as e:
        config = load_config_from_file(config_path)
    
    # Override with any provided kwargs
    config.update(kwargs)

    # build det
    return build_detector(**config)


def build_detector_from_dict(config: Dict[str, Any] = None):
    """
    Build a detector from a configuration dictionary.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary. If None, default configuration is used.
        
    Returns
    -------
    detector.Detector
        Detector object
    """
    return build_detector(**config) 