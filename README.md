# Steam Game Recommendation System using Spark ALS

## Overview
This repository contains a project that develops a Collaborative Filtering Recommender System. It leverages Apache Spark's ALS (Alternating Least Squares) algorithm to recommend games based on user behavior from the Steam dataset.

## Objective
The main objective of this project is to recommend games to users based on their historical behavior within the Steam system.

## Key Features
- **Collaborative Filtering**: Implements a recommendation system using the Alternating Least Squares (ALS) algorithm.
- **Dataset Utilization**: Uses Steam's user play history dataset, which includes User IDs, Game titles, Behavior (purchase or play), and Hours played or purchase indicators.
- **Hyperparameter Tuning**: Optimizes the ALS model performance through hyperparameter tuning using `TrainValidationSplit` and `ParamGridBuilder`.
- **Model Evaluation**: Evaluates the model's performance using metrics such as Root Mean Squared Error (RMSE).
- **MLflow Integration**: Utilizes MLflow for automatic logging of PySpark ML models, parameters, and metrics, simplifying experiment tracking.

## Technologies Used
- **Apache Spark**: For distributed data processing and building the recommendation system.
- **PySpark**: Python API for Spark, used for data manipulation and machine learning.
- **MLflow**: For managing the machine learning lifecycle, including experiment tracking.
- **ALS (Alternating Least Squares)**: Collaborative filtering algorithm for building the recommendation model.
- **PySpark ML Libraries**:
    - `StringIndexer`: Converts categorical strings into numerical indices.
    - `RegressionEvaluator`: For model evaluation using RMSE.
    - `ParamGridBuilder`: Helps define hyperparameter grids for tuning.
    - `TrainValidationSplit` and `CrossValidator`: Tools for hyperparameter tuning and model selection.

## Implementation Steps
1.  **Importing Libraries**: Essential libraries like `mlflow`, `mlflow.spark`, `pyspark.sql.functions`, `StringIndexer`, `ALS`, `RegressionEvaluator`, `ParamGridBuilder`, `TrainValidationSplit`, and `CrossValidator` are imported for the recommendation pipeline.
2.  **MLflow AutoLogging**: MLflow's autologging feature is activated to automatically track parameters, metrics, and models.
3.  **Load and Preprocess the Dataset**: The `steam_200k.csv` dataset is loaded into a Spark DataFrame, and columns are renamed for clarity (`user_id`, `game`, `behavior`, `value`).
4.  **Exploratory Data Analysis (EDA)**: Basic EDA is performed, including counting behavior types (purchase, play) and filtering the dataset to focus on "play" behavior.
5.  **Data Preparation for ALS Model**: User and game IDs are encoded into numerical indices using `StringIndexer`, which is necessary for the ALS algorithm.
6.  **Building the ALS Model**: The ALS model is built with implicit feedback and a cold start strategy to handle new users or items.
7.  **Hyperparameter Tuning**: Hyperparameters for the ALS model are tuned using `ParamGridBuilder` and `TrainValidationSplit` to find the optimal model configuration.
8.  **Model Evaluation**: The tuned model's performance is evaluated using the Root Mean Squared Error (RMSE).
9.  **Generating Recommendations**: The final model is used to generate personalized game recommendations for users.
