import os
import datetime

def create_ml_project_structure(project_name, author="Uttam Pipaliya", github="https://github.com/Uttampatel1"):
    """
    Create a standardized Machine Learning / Deep Learning project structure.
    
    Args:
        project_name (str): Name of the project
        author (str): Author's name
        github (str): GitHub profile URL
    """
    # Root directory
    if not os.path.exists(project_name):
        os.makedirs(project_name)
    
    # Directory structure with descriptions
    dirs = {
        'data': {
            'raw': 'Store raw data',
            'processed': 'Store processed data',
            'external': 'Store external data sources'
        },
        'notebooks': {
            'exploratory': 'Jupyter notebooks for EDA',
            'modeling': 'Notebooks for model development'
        },
        'src': {
            'data': 'Data processing scripts',
            'features': 'Feature engineering code',
            'models': 'Model architectures and training code',
            'utils': 'Utility functions',
            'evaluation': 'Model evaluation scripts',
            'visualization': 'Visualization scripts'
        },
        'configs': 'Configuration files',
        'models': 'Saved model artifacts',
        'tests': 'Unit tests',
        'docs': 'Documentation',
        'reports': {
            'figures': 'Generated graphics and figures',
            'results': 'Experimental results'
        }
    }
    
    def create_dirs(base_path, structure):
        for key, value in structure.items():
            path = os.path.join(base_path, key)
            if not os.path.exists(path):
                os.makedirs(path)
            
            # Create subdirectories if value is a dictionary
            if isinstance(value, dict):
                create_dirs(path, value)
            
            # Create a README.md file in each directory
            readme_path = os.path.join(path, 'README.md')
            if not os.path.exists(readme_path):
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(f'# {key.capitalize()}\n\n')
                    if isinstance(value, str):
                        f.write(f'{value}\n')
    
    # Create all directories
    create_dirs(project_name, dirs)
    
    # Create main README.md with ASCII characters instead of Unicode
    readme_content = f"""# {project_name}

## Project Overview
Brief description of your project goes here.

## Project Structure
```
{project_name}/
|
|-- data/              # Data files
|   |-- raw/          # Raw data
|   |-- processed/    # Processed data
|   +-- external/     # External data sources
|
|-- notebooks/        # Jupyter notebooks
|   |-- exploratory/  # EDA notebooks
|   +-- modeling/     # Modeling notebooks
|
|-- src/              # Source code
|   |-- data/         # Data processing
|   |-- features/     # Feature engineering
|   |-- models/       # Model implementations
|   |-- utils/        # Utility functions
|   |-- evaluation/   # Model evaluation
|   +-- visualization/# Data visualization
|
|-- configs/          # Configuration files
|-- models/           # Saved models
|-- tests/            # Unit tests
|-- docs/             # Documentation
+-- reports/          # Reports and results
    |-- figures/      # Generated graphics
    +-- results/      # Experimental results
```

## Setup and Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Author
- {author}
- {github}

Created on: {datetime.datetime.now().strftime('%Y-%m-%d')}
"""
    
    # Write files with UTF-8 encoding
    with open(os.path.join(project_name, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # Create requirements.txt
    requirements = """# Core ML/DL
numpy
pandas
scikit-learn
tensorflow>=2.0.0
torch
torchvision

# Data processing
scipy
nltk
pillow

# Visualization
matplotlib
seaborn
plotly

# Jupyter
jupyter
notebook

# Utilities
tqdm
pyyaml
python-dotenv

# Testing
pytest
pytest-cov

# Documentation
sphinx
"""
    
    with open(os.path.join(project_name, 'requirements.txt'), 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    # Create .gitignore
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDEs
.idea/
.vscode/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.csv
*.json
*.pkl
*.h5
*.hdf5
*.npy

# Model files
*.pt
*.pth
*.weights
*.pb

# Logs
logs/
*.log

# Environment variables
.env

# Mac OS
.DS_Store
"""
    
    with open(os.path.join(project_name, '.gitignore'), 'w', encoding='utf-8') as f:
        f.write(gitignore)
    
    print(f"Project structure created successfully at: {os.path.abspath(project_name)}")

# Example usage
if __name__ == "__main__":
    create_ml_project_structure("my_ml_project")
