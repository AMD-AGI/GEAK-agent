#!/usr/bin/env python3
"""
Verify that GEAK is correctly reading temperature from .env file
"""

import os
import sys
from pathlib import Path

def verify_env_setup():
    print("=" * 70)
    print("GEAK Environment Configuration Verification")
    print("=" * 70)
    
    # Check if .env file exists
    env_file = Path(__file__).parent.parent.parent / '.env'
    print(f"\n1. Checking .env file: {env_file}")
    
    if env_file.exists():
        print(f"   ✓ Found: {env_file}")
        with open(env_file, 'r') as f:
            content = f.read()
            print(f"   Content:\n   {content.replace(chr(10), chr(10) + '   ')}")
    else:
        print(f"   ✗ NOT FOUND: {env_file}")
        return False
    
    # Check if temperature is in env
    print("\n2. Checking environment variable:")
    if 'TEMPERATURE' in os.environ:
        print(f"   ✓ TEMPERATURE is set to: {os.environ['TEMPERATURE']}")
    else:
        print(f"   ⚠ TEMPERATURE not yet in environment (will be loaded when config is loaded)")
    
    # Try to load config
    print("\n3. Testing config loading:")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from args_config import load_config
        
        config_path = Path(__file__).parent / 'configs/hipbench_gaagent_config.yaml'
        config = load_config(str(config_path))
        
        print(f"   ✓ Config loaded successfully")
        print(f"   ✓ Temperature from config: {config.temperature}")
        
        if abs(config.temperature - 0.4) < 0.001:
            print(f"   ✓ CONFIRMED: Temperature is correctly set to 0.4")
            return True
        else:
            print(f"   ✗ ERROR: Temperature is {config.temperature}, expected 0.4")
            return False
    
    except Exception as e:
        print(f"   ✗ Error loading config: {e}")
        return False

if __name__ == "__main__":
    success = verify_env_setup()
    print("\n" + "=" * 70)
    if success:
        print("✅ VERIFICATION PASSED: GEAK will use temperature=0.4 from .env")
    else:
        print("❌ VERIFICATION FAILED: Check setup above")
    print("=" * 70 + "\n")
    
    sys.exit(0 if success else 1)

