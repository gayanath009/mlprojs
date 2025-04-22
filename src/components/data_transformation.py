import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer    
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Import OneHotEncoder for categorical encoding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging  # Import the logging module from the src.logger module
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    #model_path = os.path.join('artifacts', 'model.pkl')
    #train_data_path: str=os.path.join('artifacts',"train.csv")
    #test_data_path: str=os.path.join('artifacts',"test.csv")


class DataTransformation: 
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.preprocessor = None

    def get_data_transformer_object (self):

        '''
        This function creates a preprocessor object that applies different transformations to numerical and categorical columns. 
        It uses pipelines to handle missing values and scaling for numerical columns, and one-hot encoding for categorical columns.
        The function returns the preprocessor object.
        
        '''
        try:    
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
              'gender',
              'race_ethnicity',
              'parental_level_of_education',
              'lunch',
              'test_preparation_course']
        
            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='median')), # Impute missing values with median
                       ('scaler', StandardScaler()) # Scale numerical features
            ])      

            logging.info("Numerical pipelines created")

            # Categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing values with most frequent value
                       ('onehotencoder', OneHotEncoder(handle_unknown='ignore')), # One-hot encode categorical features
                       ('scaler', StandardScaler(with_mean=False)) # Scale numerical features
                ])
            
            logging.info("Categorical pipelines created")

            # Create a preprocessor object to apply the pipelines to the respective columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', numerical_pipeline, numerical_columns), # Apply numerical pipeline to numerical columns
                    ('categorical_pipeline', categorical_pipeline, categorical_columns) # Apply categorical pipeline to categorical columns
                ])
            

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        


    def initiate_data_trasnformation (self, train_path,test_path) :

        try:
            train_df = pd.read_csv(train_path) # Read the training data
            test_df = pd.read_csv(test_path) # Read the testing data

            logging.info("Read train and test data")
            logging.info("Obtaining preprocessing object")
            self.preprocessor = self.get_data_transformer_object() # Get the preprocessor object

            target_column_name = 'math_score' # Define the target column name
            numerical_columns = ['writing_score', 'reading_score'] # Define the numerical columns   
            
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1) # Drop the target column from the training data
            target_feature_train_df = train_df[target_column_name] # Extract the target column from the training data
            
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1) # Drop the target column from the testing data
            target_feature_test_df = test_df[target_column_name] # Extract the target column from the testing data

            logging.info("Applying preprocessing object on training and testing dataframes")
            # Apply the preprocessor to the training and testing dataframes
            
            input_features_train_arr = self.preprocessor.fit_transform(input_features_train_df) # Transform the training data
            input_features_test_arr = self.preprocessor.transform(input_features_test_df) # Transform the testing data

            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)] # Combine the transformed features and target for training data
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)] # Combine the transformed features and target for testing data
            
            logging.info("Saving preprocessor object")

            save_object (
                file_path=self.data_transformation_config.preprocessor_obj_file_path, # Path to save the preprocessor object
                obj=self.preprocessor # Preprocessor object to save
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path 
            # Return the transformed training and testing data along with the preprocessor object
            
        except Exception as e:
            raise CustomException(e, sys)
