# Credit Card Fraud Detection Project

## Description

The Credit Card Fraud Detection Project is a comprehensive solution designed to identify and prevent fraudulent activities in credit card transactions. This project encompasses data processing components, a machine learning model training pipeline, and an application for testing the trained model.

## Project Structure

```plaintext
V MLPROJECT
├── .ebextensions             # Elastic Beanstalk configurations
├── artifacts                 # Data and model artifacts
│   ├── data.csv              # Raw data file
│   ├── model.pkl             # Trained machine learning model
│   ├── preprocessor.pkl      # Data preprocessor for model input
│   ├── test.csv              # Testing dataset
│   └── train.csv             # Training dataset
├── catboost_info             # CatBoost model information
├── logs                      # Log files
├── mlproject.egg-info        # Python egg info
│   └── ...                   # Other files in mlproject.egg-info
├── Notebook
│   ├── creditcard (1).csv    # Additional credit card data for analysis
│   ├── Creditcard model training.ipynb  # Jupyter Notebook for model training
│   └── CreditcardEDA.ipynb   # Jupyter Notebook for exploratory data analysis
├── src
│   ├── components            # Data processing components
│   │   ├── data_ingestion.py       # Module for ingesting raw data
│   │   ├── data_transformation.py  # Module for transforming and cleaning data
│   │   └── model_trainer.py        # Module for training machine learning model
│   ├── pipeline              # Model training pipeline
│   │   ├── predict_pipeline.py     # Module for making predictions using the trained model
│   │   └── train_pipeline.py       # Module for orchestrating the model training process
│   ├── __init__.py
│   ├── exception.py          # Custom exception classes
│   ├── logger.py             # Logging utility
│   └── utils.py              # Utility functions
├── templates                 # HTML templates for the application
│   ├── home.html             # Template for the home page
│   └── index.html            # Template for the index page
├── venv                      # Virtual environment
├── .gitignore                # Gitignore file
├── application.py            # Main application script
├── python                    # Python version file
├── README.md                 # Project documentation (you are here!)
├── requirements.txt          # Python dependencies
└── setup.py                  # Setup script for packaging and distribution



a