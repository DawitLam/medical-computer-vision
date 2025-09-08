import json
import re
import os

def convert_vscode_notebook_to_jupyter(input_path, output_path):
    """Convert VS Code notebook format to standard Jupyter format"""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract cells using regex
    cell_pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
    cells = re.findall(cell_pattern, content, re.DOTALL)
    
    jupyter_cells = []
    
    for cell_id, language, cell_content in cells:
        cell_content = cell_content.strip()
        
        if language == "markdown":
            cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line.rstrip() for line in cell_content.split('\n')]
            }
        else:
            cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line.rstrip() for line in cell_content.split('\n')]
            }
        
        jupyter_cells.append(cell)
    
    # Create Jupyter notebook structure
    jupyter_notebook = {
        "cells": jupyter_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write Jupyter notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(jupyter_notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {input_path} to {output_path}")

# Convert both notebooks
if __name__ == "__main__":
    # Convert workspace setup notebook
    convert_vscode_notebook_to_jupyter(
        "notebooks/00_workspace_setup.ipynb",
        "notebooks/00_workspace_setup_jupyter.ipynb"
    )
    
    # Convert colab setup notebook
    convert_vscode_notebook_to_jupyter(
        "colab_setup.ipynb",
        "colab_setup_jupyter.ipynb"
    )
