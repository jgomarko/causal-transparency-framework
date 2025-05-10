"""
Utility functions for CTF example notebooks.

This module helps the notebooks import the CTF package by temporarily adding the parent
directory to the Python path, allowing the notebooks to be run without installing the package.
"""

import os
import sys
from pathlib import Path

def add_ctf_to_path():
    """Add the parent directory to the Python path."""
    # Get the current file's directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Get the parent directory (repository root)
    parent_dir = str(current_dir.parent.absolute())
    
    # Add to path if not already there
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"Added {parent_dir} to Python path")
    
    # Verify the import works
    try:
        import ctf
        print(f"Successfully imported CTF module from {ctf.__file__}")
        return True
    except ImportError as e:
        print(f"Failed to import CTF module: {e}")
        return False