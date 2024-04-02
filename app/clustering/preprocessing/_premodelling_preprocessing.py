"""preprocessing module for machine learning
"""
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler, 
    OneHotEncoder, 
    StandardScaler,

)


# Building preprocessing pipeline for ml
def create_pipe_ml(ls_num_features: List[str],
                   ls_cat_features: List[str] = None) -> object:
    """
    The `create_pipe_ml` function is used to create a preprocessing pipeline for a dataset before machine learning. 
    The function has the following arguments:

    Arguments:
    - `ls_num_features`: A list of strings containing the names of the numerical features to preprocess.
    - `ls_cat_features`: A list of strings containing the names of the categorical features to preprocess. 
    This argument is optional and defaults to `None`.

    Returns:
    - `Pipeline`: A scikit-learn pipeline object containing the preprocessing steps.

    The function first creates a pipeline object for numerical features, which includes an imputer to replace 
    missing values with 0, a standard scaler to scale the data to have a mean of 0 and standard deviation of 1, 
    and a min-max scaler to scale the data to the range [0,1]. 

    If `ls_cat_features` is not `None`, the function also creates a pipeline object for categorical features, 
    which includes a one-hot encoder to encode categorical variables as binary features.

    Finally, the function creates a column transformer object to apply the numerical and categorical pipelines 
    to the appropriate features in the dataset. The function returns a scikit-learn pipeline object containing the preprocessing steps.    
    """    

    numeric_transformer_ml = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            # ('scaler', StandardScaler()),
            # ('minmax_scaler2', MinMaxScaler())     # This is to make the dataset model-agnostic and easy to compare btw models
            ]   
        )
    
    if ls_cat_features is not None:
        # Function to convert input to strings
        def convert_to_string(input_data):
            return input_data.astype(str)

        onehot_transformer_ml = Pipeline(
            steps=[
                ('to_string', FunctionTransformer(convert_to_string)),
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary')),
                ]
            )
    
    if ls_cat_features is not None:
        preprocessor_ml = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer_ml, ls_num_features),
                ('cat_onehot', onehot_transformer_ml, ls_cat_features),
                ]
            )
    else: 
        preprocessor_ml = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer_ml, ls_num_features),
                ]
            )

    return Pipeline(steps=[('preprocessor', preprocessor_ml),])
