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
```

## Installation
1. Clone the repository
git clone https://github.com/Devam42/MLPROJECT.git

2. Navigate to the project directory:
cd MLPROJECT

3. Install dependencies:
pip install -r requirements.txt

## Usage
1. Prepare data:
Ensure the necessary data files (data.csv, train.csv, test.csv) are present in the artifacts folder.

2. Run the application:
python application.py

3. Follow instructions:
The application will guide you through the process of testing the credit card fraud detection model.

A. Run python application.py in your terminal.

B. Search http://127.0.0.1:5000 in your local browser to land on welcome page.

C. Search http://127.0.0.1:5000/predictdata in your local browser to land on "Credit Card Fraud Prediction page.

D. Give all the inputs to get the prediction of is credit card claim is fraud or genuine.


## Features

### 1. Data processing
a. Ingest raw data with data_ingestion.py.
b. Clean and transform data with data_transformation.py.

### 2. Model Training
a. Train the machine learning model with model_trainer.py.
b. Utilize the CatBoost algorithm for enhanced performance.
c. *Note: Hyperparameter tuning has not been performed due to hardware constraints on the development machine. Code for hyperparameter tuning is included in comments in `model_trainer.py` for future implementation when more computational resources are available.*


### 3. Pipeline
a. Predictions can be made using predict_pipeline.py.
b. The entire training process is orchestrated with train_pipeline.py.

### 4. Application
User-friendly application (application.py) for testing and validating the trained model.


### Modular Coding
The project follows a modular coding approach to enhance maintainability and readability. Key components and functionalities are organized into separate modules within the src folder. This modular structure allows for easy updates, improvements, and collaboration.

### Flask
The application is built using Flask, a micro web framework for Python. Flask simplifies the process of developing web applications by providing a lightweight and flexible framework. It is utilized to create the backend of the credit card fraud detection application, facilitating seamless communication between the model and the user interface.

### Frontend with HTML
HTML templates (home.html and index.html) are used to create the frontend of the application. These templates provide a user-friendly interface for interacting with the credit card fraud detection system. The integration of HTML enhances the overall user experience and allows for customization of the application's appearance.

### Hyperparameter Tuning
Due to hardware constraints on the development machine, hyperparameter tuning was not performed during the model training process. Hyperparameter tuning can be a computationally expensive task, and on the available hardware, it resulted in impractical processing times.

However, to demonstrate the awareness of hyperparameter tuning and its importance, the code for hyperparameter tuning has been included in the 'model_trainer.py' file as comments. The commented-out sections provide an outline of how hyperparameter tuning could be implemented when computing resources with higher specifications are available.

Feel free to uncomment and execute the relevant sections in the future when access to a more powerful computing environment is available. The current model achieved satisfactory results even without hyperparameter tuning, showcasing its robustness.

## Testing 

A. Run python application.py in your terminal.
B. Search http://127.0.0.1:5000 in your local browser to land on welcome page.
C. Search http://127.0.0.1:5000/predictdata in your local browser to land on "Credit Card Fraud Prediction page.
D. Give all the inputs to get the prediction of is credit card claim is fraud or genuine.

## Deployment
#### AWS Elastic Beanstalk Deployment with AWS CodePipeline
This project utilizes AWS Elastic Beanstalk for deployment and AWS CodePipeline for continuous delivery. Follow the steps below to deploy the application to AWS Elastic Beanstalk:

1. Elastic Beanstalk Setup:
a. Create an Elastic Beanstalk environment for your application.
b. Configure the necessary environment variables, such as AWS credentials and application-specific configurations.

2. AWS CodePipeline Setup:
a. Set up an AWS CodePipeline to automate the deployment process.
b. Configure the pipeline stages, including source, build, and deploy.
c. Connect your GitHub repository as the source.

3. Deployment Configuration:
a. Configure the deployment settings in the Elastic Beanstalk environment within the AWS Management Console.
b. Define environment variables required for the application.

4. Automatic Deployment:
Once the CodePipeline is set up, any changes pushed to the connected GitHub repository's main branch will automatically trigger a deployment to the Elastic Beanstalk environment.

### Troubleshooting
If you encounter issues during deployment, consider the following troubleshooting steps:
a. Check AWS Elastic Beanstalk and AWS CodePipeline logs for error messages.
b. Verify that all necessary environment variables are correctly configured.
c. Ensure the IAM role used by AWS CodePipeline has the required permissions.


## Conclusion
Thank you for exploring Credit Card Fraud Detection Project! Hope this tool proves valuable in identifying and preventing fraudulent activities in credit card transactions.

If you have any questions, feedback, or encounter issues, please don't hesitate to reach out:

Email: dkathane42@gmail.com


We appreciate your interest and contributions.

Happy coding!