# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import os
import yaml
from argparse import Namespace
from pathlib import Path

def get_package_config_path(config_name: str) -> str:
    """Get the path to a config file within the package."""
    package_dir = Path(__file__).parent
    config_path = package_dir / "configs" / config_name
    if config_path.exists():
        return str(config_path)
    # Fall back to just the config name (assumes it's a full path)
    return config_name

def load_config(yaml_path: str) -> Namespace:
    """Load configuration from a YAML file.
    
    Args:
        yaml_path: Path to the YAML config file. Can be:
            - An absolute path
            - A relative path from the current directory
            - Just the config filename (will look in package configs directory)
    
    Returns:
        Namespace object containing the configuration.
    """
    # Check if it's a direct path that exists
    if os.path.exists(yaml_path):
        config_path = yaml_path
    # Check if it's a config name that exists in the package
    elif not os.path.isabs(yaml_path):
        # Try to find it relative to package configs
        package_config = get_package_config_path(os.path.basename(yaml_path))
        if os.path.exists(package_config):
            config_path = package_config
        else:
            config_path = yaml_path
    else:
        config_path = yaml_path
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Namespace(**config_dict)
