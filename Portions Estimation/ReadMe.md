  # Project description, setup instructions, and how to use the code
  
# Source Code
This directory contains the main source code for the project, including scripts for data preprocessing, model building, training, evaluation, and utility functions.

## Directory Structure

- `__init__.py`: Initialization file for the `src/` module.
- `data_preprocessing.py`: Contains functions for preprocessing raw data (e.g., resizing images, normalizing values, generating embeddings).
- `model.py`: Defines the architecture of the CNN and fusion model.
- `utils.py`: Includes helper functions (e.g., for saving/loading models, logging, and evaluation metrics).
- `train.py`: Script for training the CNN model, including data loading, training loops, and saving weights.
- `evaluate.py`: Script for evaluating the trained model on validation/test data, including portion size estimation and accuracy calculations.
