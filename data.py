#Data processing
import numpy as np
import pandas as pd
import os

def get_relative_csv_data(file_name: str, delimiter: str = ','):
    '''
        Reads a CSV file from the current working directory.

        Parameters:
        file_name (str): The name of the CSV file.

        Returns:
        pd.DataFrame: The dataframe containing the CSV data.
    '''
    data_file = os.path.join(os.getcwd(), file_name)
    return pd.read_csv(data_file, delimiter=delimiter)

def remove_constant_columns(df):
    '''
        Removes columns with constant values from the dataframe.

        Parameters:
        df (pd.DataFrame): The input dataframe.
    '''
    return df.loc[:, (df.nunique() > 1)]

def remove_duplicate_rows(df):
    '''
        Removes duplicate rows from the dataframe.

        Parameters:
        df (pd.DataFrame): The input dataframe.
    '''
    return df.drop_duplicates()

def remove_outliers_in_column(df, column, method='iqr', threshold=1.5):
    """
    Removes rows containing outliers in a specific column.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        column (str): The column in which to identify outliers.
        method (str): Method to detect outliers ('iqr' or 'zscore').
        threshold (float): The threshold for outlier detection (default: 1.5 for IQR, 3 for Z-score).
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if method == 'iqr':
        # Compute Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for non-outliers
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Filter the dataframe
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    elif method == 'zscore':
        # Compute Z-scores
        mean = df[column].mean()
        std = df[column].std()
        df['z_score'] = (df[column] - mean) / std
        
        # Filter based on threshold
        df_filtered = df[abs(df['z_score']) < threshold].drop(columns=['z_score'])
        return df_filtered

    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")


def remove_outliers_all_columns(df, method='iqr', threshold=1.5):
    """
    Removes rows containing outliers in all numerical columns.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        method (str): Method to detect outliers ('iqr' or 'zscore').
        threshold (float): The threshold for outlier detection (default: 1.5 for IQR, 3 for Z-score).
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    df_cleaned = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns  # Select only numeric columns
    
    for column in numeric_cols:
        if method == 'iqr':
            Q1 = df_cleaned[column].quantile(0.25)
            Q3 = df_cleaned[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
        
        elif method == 'zscore':
            mean = df_cleaned[column].mean()
            std = df_cleaned[column].std()
            df_cleaned['z_score'] = (df_cleaned[column] - mean) / std
            df_cleaned = df_cleaned[abs(df_cleaned['z_score']) < threshold].drop(columns=['z_score'])

        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")

    return df_cleaned

# Example usage:
# df = remove_outliers_all_columns(df, method='iqr')

#def balance_dataset_by_target(df, target_column):


def transform_student_data(input_file: str, delimiter, output_file: str):
    '''
        Transforms the student data from the input file and saves the processed data to the output file.

        Parameters:
        input_file (str): The name of the input file.
        output_file (str): The name of the output file.    
    '''
    
    get_relative_csv_data(input_file, delimiter=delimiter)
    print("Loaded datafile")
    
    # Encoding categorical variables into binary values
    df['school'] = df['school'].map({'GP': 0, 'MS': 1})
    df['sex'] = df['sex'].map({'M': 0, 'F': 1})
    df['address'] = df['address'].map({'U': 0, 'R': 1})
    df['famsize'] = df['famsize'].map({'LE3': 0, 'GT3': 1})
    df['Pstatus'] = df['Pstatus'].map({'T': 0, 'A': 1})
    
    # Feature expansion for parent's job
    for job in ['teacher', 'health', 'services', 'at_home', 'other']:
        df[f'Mjob{job.capitalize()}'] = (df['Mjob'] == job).astype(int)
        df[f'Fjob{job.capitalize()}'] = (df['Fjob'] == job).astype(int)
    
    # Feature expansion for reason
    for reason in ['home', 'reputation', 'course', 'other']:
        df[f'reason{reason.capitalize()}'] = (df['reason'] == reason).astype(int)
    
    # Feature expansion for guardian
    for guardian in ['mother', 'father', 'other']:
        df[f'guardian{guardian.capitalize()}'] = (df['guardian'] == guardian).astype(int)
    
    # Encoding binary categorical variables
    binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    
    # Dropping original categorical columns that were expanded
    df.drop(columns=['Mjob', 'Fjob', 'reason', 'guardian'], inplace=True)
    
    # Dropping the age column as per the provided suggestion
    df.drop(columns=['age'], inplace=True)
    
    # Save transformed CSV
    df.to_csv(os.path.join(os.getcwd(), output_file), index=False)
    
    print(f"Processed data saved to {output_file}")



# Transform the data

#School name to binary GP - 0, MS - 1
#Sex to binary M - 0, F - 1
#Age has no change - Consider dropping, how do we handle different ages and hypothesis around it?
#Address to binary U - 0, R - 1
#Famsize to binary LE3 - 0, GT3 - 1
#Pstatus to binary T - 0, A - 1

#For parents jobs we do feature expansion:
#MjobTeacher Yes - 1, No - 0 (Mjob = "teacher")
#MjobHealth Yes - 1, No - 0 (Mjob = "health")
#MjobServices Yes - 1, No - 0 (Mjob = "services")
#MjobAt_home Yes - 1, No - 0 (Mjob = "at_home")
#MjobOther Yes - 1, No - 0 (Mjob = "other")

#FjobTeacher Yes - 1, No - 0 (Fjob = "teacher")
#FjobHealth Yes - 1, No - 0 (Fjob = "health")
#FjobServices Yes - 1, No - 0 (Fjob = "services")
#FjobAt_home Yes - 1, No - 0 (Fjob = "at_home")
#FjobOther Yes - 1, No - 0 (Fjob = "other")

#For reason we do feature expansion:
#reasonHome Yes - 1, No - 0 (reason = "home")
#reasonReputation Yes - 1, No - 0 (reason = "reputation")
#reasonCourse Yes - 1, No - 0 (reason = "course")
#reasonOther Yes - 1, No - 0    (reason = "other")

#For guardian we do feature expansion:
#guardianMother Yes - 1, No - 0 (guardian = "mother")
#guardianFather Yes - 1, No - 0 (guardian = "father")
#guardianOther Yes - 1, No - 0 (guardian = "other")

#traveltime has no change
#studytime has no change
#failures has no change
#schoolsup to binary Yes - 1, No - 0
#famsup to binary Yes - 1, No - 0
#paid to binary Yes - 1, No - 0
#activities to binary Yes - 1, No - 0
#nursery to binary Yes - 1, No - 0
#higher to binary Yes - 1, No - 0
#internet to binary Yes - 1, No - 0
#romantic to binary Yes - 1, No - 0

#famrel has no change
#freetime has no change
#goout has no change
#Dalc has no change
#Walc has no change
#health has no change
#absences has no change

#G1 has no change
#G2 has no change
#G3 has no change

