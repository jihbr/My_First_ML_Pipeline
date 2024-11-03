## Spaceship Titanic: Data Processing and Modeling Pipeline
Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery! The Spaceship Titanic, an interstellar passenger liner, was tragically affected by a spacetime anomaly,
resulting in nearly half of its passengers being transported to an alternate dimension. This repository provides a machine learning pipeline designed to process and analyze data from the spaceship’s damaged computer system to predict which passengers were affected.

### Project Structure
data_load.py - Script to load CSV data into a DataFrame
make_pipeline.py - Script to create data preprocessing and model pipeline
train_model.py - Script to train and evaluate the model pipeline
README.md - Project documentation

### Requirements
Python: 3.7 or higher
Libraries: pandas, scikit-learn, argparse


### Usage
1. Load Data
The data_load.py script loads a CSV file into a pandas DataFrame with the first column as the index.
2. Create the Preprocessing Pipeline
The make_pipeline.py script creates a preprocessing and modeling pipeline for the Spaceship Titanic dataset. The pipeline includes transformations for categorical and numerical columns, as well as a RandomForestRegressor model.
3. Train and Evaluate the Model
The train_model.py script trains the model using the pipeline created in make_pipeline.py. It splits the data into training and testing sets, fits the model, and outputs performance metrics for both sets.


### Parameters
file_path (str): Path to the CSV file.
--X_cols (list): List of feature columns for the model.
--y_col (str): Target variable column, indicating transported passengers.
--ts (float): Ratio for test set size (default is 0.2).
--random (int): Random seed for reproducibility (default is 42).

### Model Metrics
After running train_model.py, the model metrics are printed as follows:

### Training Data:
score: The model’s score on the training data.
mae: Mean Absolute Error for the training set predictions.

### Test Data:
score: The model’s score on the test data.
mae: Mean Absolute Error for the test set predictions.

### Project Details
The pipeline was tested with the Spaceship Titanic dataset on Kaggle, which contains features such as HomePlanet, CryoSleep, Cabin, and more to help predict the transported status of passengers.

This machine learning pipeline is designed for flexibility, with customizable file paths, feature columns, and model parameters, enabling adaptation to other datasets with minimal adjustments.

After downloading the training data from Kaggle https://www.kaggle.com/competitions/spaceship-titanic/discussion/309323, this terminal command python train_model.py ~/train.csv --X_cols HomePlanet CryoSleep Cabin Destination Age VIP RoomService FoodCourt ShoppingMall Spa VRDeck Name --y_col Transported --ts 0.2 --random 42
should return metrics for a trained model as follows: Model Metrics: {'train_data': {'score': 0.910298462056415, 'mae': 0.08226200747771066}, 'test_data': {'score': 0.32535496403239306, 'mae': 0.2296779758481886}}


### Future Enhancements
Future improvements could include:

Adding additional error handling to validate input columns and data types.
Expanding the pipeline to support other machine learning models.
Enhancing the feature engineering process.

### License
This project is licensed under the MIT License.
