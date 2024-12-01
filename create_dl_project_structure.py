import os
import datetime

def create_dl_project_structure(project_name, author="Uttam Pipaliya", github="https://github.com/Uttampatel1"):
    """
    Create a standardized Deep Learning project structure with boilerplate code.
    
    Args:
        project_name (str): Name of the project
        author (str): Author's name
        github (str): GitHub profile URL
    """
    # Root directory
    if not os.path.exists(project_name):
        os.makedirs(project_name)
    
    # Directory structure
    structure = {
        'data': {
            'raw': 'Store raw data',
            'processed': 'Store processed data',
            'interim': 'Store intermediate processing data'
        },
        'src': {
            'data': {
                'data_loader.py': '''
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        
    def __len__(self):
        # Implement dataset length
        pass
        
    def __getitem__(self, idx):
        # Implement data loading logic
        pass
''',
                'preprocessing.py': '''
def preprocess_data():
    """
    Implement data preprocessing pipeline
    """
    pass
'''
            },
            'models': {
                'model.py': '''
import torch
import torch.nn as nn

class DeepModel(nn.Module):
    def __init__(self):
        super(DeepModel, self).__init__()
        # Define model architecture
        
    def forward(self, x):
        # Implement forward pass
        pass
''',
                'layers.py': '''
import torch.nn as nn

# Define custom layers here
'''
            },
            'training': {
                'trainer.py': '''
class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
    def train_epoch(self):
        # Implement training loop
        pass
        
    def evaluate(self):
        # Implement evaluation
        pass
''',
                'utils.py': '''
def save_checkpoint(model, optimizer, epoch, path):
    """Save model checkpoint"""
    pass

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    pass
'''
            },
            'evaluation': {
                'metrics.py': '''
def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics"""
    pass
''',
                'visualization.py': '''
def plot_training_history():
    """Plot training metrics"""
    pass

def visualize_predictions():
    """Visualize model predictions"""
    pass
'''
            },
            'config': {
                'config.py': '''
from dataclasses import dataclass

@dataclass
class Config:
    # Model parameters
    model_name: str = "DeepModel"
    input_size: int = 224
    num_classes: int = 10
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    
    # Paths
    data_dir: str = "data/processed"
    model_dir: str = "models"
    log_dir: str = "logs"
'''
            }
        },
        'notebooks': {
            'exploratory': 'Jupyter notebooks for exploration',
            'experiments': 'Experimental notebooks'
        },
        'models': 'Saved model checkpoints',
        'configs': {
            'model_config.yaml': '''
model:
  name: DeepModel
  input_size: 224
  num_classes: 10

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: adam
  
data:
  train_path: data/processed/train
  val_path: data/processed/val
  test_path: data/processed/test
''',
        },
        'logs': 'Training logs and tensorboard files',
        'tests': {
            'test_model.py': '''
import unittest

class TestModel(unittest.TestCase):
    def setUp(self):
        # Setup test environment
        pass
        
    def test_forward_pass(self):
        # Test model forward pass
        pass
''',
            'test_data_loader.py': '''
import unittest

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Setup test environment
        pass
        
    def test_data_loading(self):
        # Test data loading
        pass
'''
        },
        'docs': {
            'README.md': '''# Model Documentation

## Architecture
Describe model architecture here

## Training Process
Describe training process here

## Evaluation
Describe evaluation metrics and results
''',
        },
        'scripts': {
            'train.py': '''
import argparse
from src.config.config import Config
from src.models.model import DeepModel
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/model_config.yaml')
    args = parser.parse_args()
    
    # Implement training pipeline
    
if __name__ == '__main__':
    main()
''',
            'evaluate.py': '''
import argparse
from src.models.model import DeepModel
from src.evaluation.metrics import calculate_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    
    # Implement evaluation pipeline

if __name__ == '__main__':
    main()
''',
            'predict.py': '''
import argparse
from src.models.model import DeepModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    args = parser.parse_args()
    
    # Implement prediction pipeline

if __name__ == '__main__':
    main()
'''
        }
    }
    
    def create_structure(base_path, structure):
        for name, content in structure.items():
            path = os.path.join(base_path, name)
            
            if isinstance(content, dict):
                os.makedirs(path, exist_ok=True)
                create_structure(path, content)
            else:
                directory = os.path.dirname(path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                    
                if isinstance(content, str):
                    with open(path, 'w') as f:
                        f.write(content.strip())
                        
                # Create README for directories
                if os.path.isdir(path) and not os.path.exists(os.path.join(path, 'README.md')):
                    with open(os.path.join(path, 'README.md'), 'w', encoding='utf-8') as f:
                        f.write(f'# {name.capitalize()}\n\n{content}')
    
    # Create project structure
    create_structure(project_name, structure)
    
    # Create main README.md
    readme_content = f"""# {project_name}

## Deep Learning Project Template

### Project Overview
Brief description of your project goes here.

### Project Structure
```
{project_name}/
│
├── data/              # Data files
│   ├── raw/          # Raw data
│   ├── processed/    # Processed data
│   └── interim/      # Intermediate data
│
├── src/              # Source code
│   ├── data/         # Data loading and preprocessing
│   ├── models/       # Model architectures
│   ├── training/     # Training logic
│   ├── evaluation/   # Evaluation metrics and visualization
│   └── config/       # Configuration files
│
├── notebooks/        # Jupyter notebooks
│   ├── exploratory/  # Data exploration
│   └── experiments/  # Experimental notebooks
│
├── models/           # Saved models
├── configs/          # Configuration files
├── logs/            # Training logs
├── tests/           # Unit tests
├── docs/            # Documentation
└── scripts/         # Training and evaluation scripts
```

### Setup and Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage
1. Prepare your data in the `data/` directory
2. Configure your model in `configs/model_config.yaml`
3. Train the model:
   ```bash
   python scripts/train.py --config configs/model_config.yaml
   ```
4. Evaluate the model:
   ```bash
   python scripts/evaluate.py --model_path models/model.pth --data_path data/processed/test
   ```

## Author
- {author}
- {github}

Created on: {datetime.datetime.now().strftime('%Y-%m-%d')}
"""
    
    with open(os.path.join(project_name, 'README.md'), 'w' , encoding='utf-8') as f:
        f.write(readme_content)
    
    # Create requirements.txt
    requirements = """# Deep Learning
torch>=1.9.0
torchvision>=0.10.0
tensorflow>=2.6.0
transformers>=4.5.0

# Data Processing
numpy>=1.19.5
pandas>=1.3.0
scikit-learn>=0.24.2
pillow>=8.2.0
albumentations>=1.0.0

# Visualization
matplotlib>=3.4.2
seaborn>=0.11.1
tensorboard>=2.6.0

# Utilities
tqdm>=4.61.0
pyyaml>=5.4.1
python-dotenv>=0.19.0

# Experiment Tracking
wandb>=0.12.0
mlflow>=1.19.0

# Testing
pytest>=6.2.5
pytest-cov>=2.12.1

# Documentation
sphinx>=4.1.2
"""
    
    with open(os.path.join(project_name, 'requirements.txt'), 'w' , encoding='utf-8') as f:
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

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Data
data/raw/
data/processed/
data/interim/
*.csv
*.json
*.pkl
*.h5
*.npy

# Models
models/*.pth
models/*.pt
models/*.h5
models/*.pb
models/*.onnx

# Logs
logs/
runs/
*.log
wandb/

# Environment variables
.env

# OS
.DS_Store
"""
    
    with open(os.path.join(project_name, '.gitignore'), 'w', encoding='utf-8') as f:
        f.write(gitignore)
    
    print(f"Deep Learning project structure created successfully at: {os.path.abspath(project_name)}")

# Example usage
if __name__ == "__main__":
    create_dl_project_structure("deep_learning_project")
