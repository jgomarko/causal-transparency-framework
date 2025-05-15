#!/usr/bin/env python3
"""Fix Jupyter notebooks by adding missing execution_count properties."""

import json
import os


def fix_notebook(notebook_path):
    """Fix a Jupyter notebook by adding missing execution_count properties."""
    print(f"Fixing notebook: {notebook_path}")
    
    try:
        # Read the notebook
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        # Fix cells
        if 'cells' in notebook:
            for i, cell in enumerate(notebook['cells']):
                # Add execution_count to code cells
                if cell.get('cell_type') == 'code':
                    if 'execution_count' not in cell:
                        cell['execution_count'] = i + 1
                
                # Ensure outputs exist for code cells
                if cell.get('cell_type') == 'code' and 'outputs' not in cell:
                    cell['outputs'] = []
        
        # Ensure metadata exists
        if 'metadata' not in notebook:
            notebook['metadata'] = {}
        
        # Ensure nbformat and nbformat_minor exist
        if 'nbformat' not in notebook:
            notebook['nbformat'] = 4
        if 'nbformat_minor' not in notebook:
            notebook['nbformat_minor'] = 4
        
        # Write the fixed notebook
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"Successfully fixed: {notebook_path}")
        
    except Exception as e:
        print(f"Error fixing {notebook_path}: {e}")


def main():
    """Main function to fix all notebooks in the examples directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Fix MIMIC-III example notebook
    mimic_notebook = os.path.join(script_dir, 'mimic_iii_example.ipynb')
    if os.path.exists(mimic_notebook):
        fix_notebook(mimic_notebook)
    else:
        print(f"Not found: {mimic_notebook}")
    
    # Fix COMPAS example notebook
    compas_notebook = os.path.join(script_dir, 'compas_example.ipynb')
    if os.path.exists(compas_notebook):
        fix_notebook(compas_notebook)
    else:
        print(f"Not found: {compas_notebook}")


if __name__ == '__main__':
    main()