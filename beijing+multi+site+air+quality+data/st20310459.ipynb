# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, r2_score

# # Task 1: Data Handling (Load and Merge Data)
# def load_and_merge_data():
#     file_paths = [
#         r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Aotizhongxin_20130301-20170228.csv',
#         r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Changping_20130301-20170228.csv',
#         # Add other file paths here as needed
#     ]
    
#     # Load and combine all datasets
#     dataframes = [pd.read_csv(file) for file in file_paths]
#     combined_df = pd.concat(dataframes, ignore_index=True)
    
#     return combined_df

# # Task 2: Handle Missing Values and Remove Duplicates
# def handle_missing_and_duplicates(df):
#     print("\nMissing Values Count per Column:")
#     print(df.isnull().sum())  # Count of missing values in each column
    
#     # Fill missing values using forward fill
#     df.fillna(method='ffill', inplace=True)
    
#     # Remove duplicate rows
#     df.drop_duplicates(inplace=True)
    
#     return df

# # Task 3: Feature Engineering (Date-Time Parsing, Creating New Features, etc.)
# def feature_engineering(df):
#     # Convert date columns to datetime if necessary
#     if 'date' in df.columns:
#         df['date'] = pd.to_datetime(df['date'])
#         print("\nConverted 'date' column to datetime.")
    
#     # Extract year, month, and day from the datetime column
#     if 'date' in df.columns:
#         df['year'] = df['date'].dt.year
#         df['month'] = df['date'].dt.month
#         df['day'] = df['date'].dt.day
    
#     # Create a 'season' feature based on the month
#     df['season'] = df['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else
#                                                 'Summer' if 6 <= x <= 8 else
#                                                 'Fall' if 9 <= x <= 11 else 'Winter')
    
#     return df

# # Task 4: Data Cleaning (Remove Unnecessary Columns, Handle Data Types)
# def data_cleaning(df):
#     # Drop columns not required for analysis
#     if 'No' in df.columns:
#         df.drop('No', axis=1, inplace=True)
    
#     # Convert object columns to category data type
#     for col in df.select_dtypes(include=['object']).columns:
#         df[col] = df[col].astype('category')
#         print(f"Converted column '{col}' to 'category' data type.")
    
#     return df

# # Task 5: Statistical Analysis and Visualizations
# def perform_analysis_and_visualization(df):
#     # Check the 'PM2.5' column for missing values and type issues
#     print("\nSummary statistics for 'PM2.5':")
#     print(df['PM2.5'].describe())
#     print(f"Missing values in 'PM2.5': {df['PM2.5'].isnull().sum()}")

#     # Handle missing values for PM2.5 (e.g., drop NaNs for the following plots)
#     pm25_clean = df['PM2.5'].dropna()

#     # Univariate Analysis - Distribution of PM2.5
#     plt.figure(figsize=(10, 6))
#     sns.histplot(pm25_clean, bins=30)  # Drop NA values for plot
#     plt.title('Distribution of PM2.5 Concentration')
#     plt.xlabel('PM2.5')
#     plt.ylabel('Frequency')
#     plt.xlim([0, pm25_clean.max()])  # Set x-axis limits
#     plt.show()

#     # Bivariate Analysis - Scatter plot between PM2.5 and PM10
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x='PM2.5', y='PM10', data=df.dropna(subset=['PM2.5', 'PM10']))
#     plt.title('Scatter plot of PM2.5 vs. PM10')
#     plt.xlabel('PM2.5')
#     plt.ylabel('PM10')
#     plt.show()

#     # Correlation Matrix and Heatmap - Only include numeric columns
#     numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns  # Select numeric columns only
#     corr_matrix = df[numeric_columns].corr()

#     plt.figure(figsize=(12, 8))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
#     plt.title('Heatmap of Correlation between Variables')
#     plt.show()

#     # Box Plot - PM2.5 by Season
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x='season', y='PM2.5', data=df)
#     plt.title('Box Plot of PM2.5 Concentration by Season')
#     plt.xlabel('Season')
#     plt.ylabel('PM2.5')
#     plt.show()

#     # Multivariate Analysis - Pair Plot for Selected Numeric Variables
#     numeric_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'TEMP']  # List only numeric columns
#     plt.figure(figsize=(12, 8))
#     sns.pairplot(df[numeric_columns].dropna())
#     plt.title('Pair Plot of Selected Variables')
#     plt.show()

# # Task 6: Building and Evaluating the Machine Learning Model
# def build_and_evaluate_model(df):
#     # Define the features and target variable
#     X = df.drop('PM2.5', axis=1)  # Use all columns except the target
#     y = df['PM2.5']  # Target variable

#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Create a preprocessing pipeline
#     numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
#     categorical_features = X_train.select_dtypes(include=['category']).columns.tolist()

#     # Preprocessing for numeric features
#     numeric_transformer = Pipeline(steps=[
#         ('scaler', StandardScaler())  # Feature scaling
#     ])

#     # Preprocessing for categorical features
#     categorical_transformer = Pipeline(steps=[
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding
#     ])

#     # Combine preprocessing steps
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numeric_features),
#             ('cat', categorical_transformer, categorical_features)
#         ])

#     # Create the model pipeline
#     model = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', RandomForestRegressor())
#     ])

#     # Hyperparameter tuning
#     param_distributions = {
#         'regressor__n_estimators': [50, 100, 150],
#         'regressor__max_depth': [10, 20, 30],
#         'regressor__min_samples_split': [2, 5],
#         'regressor__min_samples_leaf': [1, 2]
#     }

#     search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, scoring='r2', verbose=2, random_state=42)
#     search.fit(X_train, y_train)

#     print(f"Best parameters: {search.best_params_}")

#     # Evaluate the model
#     y_pred = search.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print(f"Mean Squared Error: {mse}")
#     print(f"R² Score: {r2}")

# # Main Function to Run Preprocessing Steps and Analysis
# if __name__ == "__main__":
#     # Step 1: Load and combine data
#     combined_data = load_and_merge_data()

#     # Step 2: Handle missing values and remove duplicates
#     preprocessed_data = handle_missing_and_duplicates(combined_data)

#     # Step 3: Perform feature engineering
#     preprocessed_data = feature_engineering(preprocessed_data)

#     # Step 4: Clean the data by removing unnecessary columns and converting data types
#     preprocessed_data = data_cleaning(preprocessed_data)

#     # Step 5: Perform statistical analysis and visualizations
#     perform_analysis_and_visualization(preprocessed_data)

#     # Step 6: Build and evaluate the machine learning model
#     build_and_evaluate_model(preprocessed_data)
# data should calculate average base or hourly based also

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Task 1: Data Handling (Load and Merge Data)
def load_and_merge_data():
    file_paths = [
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Aotizhongxin_20130301-20170228.csv',
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Changping_20130301-20170228.csv',
        # Add other file paths here as needed
    ]
    
    # Load and combine all datasets
    dataframes = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    return combined_df

# Task 2: Handle Missing Values and Remove Duplicates
def handle_missing_and_duplicates(df):
    print("\nMissing Values Count per Column:")
    print(df.isnull().sum())  # Count of missing values in each column
    
    # Fill missing values using forward fill
    df.fillna(method='ffill', inplace=True)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df

# Task 3: Feature Engineering (Date-Time Parsing, Creating New Features, etc.)
def feature_engineering(df):
    # Convert date columns to datetime if necessary
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        print("\nConverted 'date' column to datetime.")
    
    # Extract year, month, and day from the datetime column
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
    
    # Create a 'season' feature based on the month
    df['season'] = df['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else
                                                'Summer' if 6 <= x <= 8 else
                                                'Fall' if 9 <= x <= 11 else 'Winter')
    
    return df

# Task 4: Data Cleaning (Remove Unnecessary Columns, Handle Data Types)
def data_cleaning(df):
    # Drop columns not required for analysis
    if 'No' in df.columns:
        df.drop('No', axis=1, inplace=True)
    
    # Convert object columns to category data type
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
        print(f"Converted column '{col}' to 'category' data type.")
    
    return df

# Task 5: Statistical Analysis and Visualizations
def perform_analysis_and_visualization(df):
    # Check the 'PM2.5' column for missing values and type issues
    print("\nSummary statistics for 'PM2.5':")
    print(df['PM2.5'].describe())
    print(f"Missing values in 'PM2.5': {df['PM2.5'].isnull().sum()}")

    # Handle missing values for PM2.5 (e.g., drop NaNs for the following plots)
    pm25_clean = df['PM2.5'].dropna()

    # Univariate Analysis - Distribution of PM2.5
    plt.figure(figsize=(10, 6))
    sns.histplot(pm25_clean, bins=30)  # Drop NA values for plot
    plt.title('Distribution of PM2.5 Concentration')
    plt.xlabel('PM2.5')
    plt.ylabel('Frequency')
    plt.xlim([0, pm25_clean.max()])  # Set x-axis limits
    plt.show()

    # Bivariate Analysis - Scatter plot between PM2.5 and PM10
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PM2.5', y='PM10', data=df.dropna(subset=['PM2.5', 'PM10']))
    plt.title('Scatter plot of PM2.5 vs. PM10')
    plt.xlabel('PM2.5')
    plt.ylabel('PM10')
    plt.show()

    # Correlation Matrix and Heatmap - Only include numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns  # Select numeric columns only
    corr_matrix = df[numeric_columns].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap of Correlation between Variables')
    plt.show()

    # Box Plot - PM2.5 by Season
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='season', y='PM2.5', data=df)
    plt.title('Box Plot of PM2.5 Concentration by Season')
    plt.xlabel('Season')
    plt.ylabel('PM2.5')
    plt.show()

    # Multivariate Analysis - Pair Plot for Selected Numeric Variables
    numeric_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'TEMP']  # List only numeric columns
    plt.figure(figsize=(12, 8))
    sns.pairplot(df[numeric_columns].dropna())
    plt.title('Pair Plot of Selected Variables')
    plt.show()

# Task 6: Building and Evaluating the Machine Learning Model
def build_and_evaluate_model(df):
    # Define the features and target variable
    X = df.drop('PM2.5', axis=1)  # Use all columns except the target
    y = df['PM2.5']  # Target variable

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a preprocessing pipeline
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['category']).columns.tolist()

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())  # Feature scaling
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    # Hyperparameter tuning
    param_distributions = {
        'regressor__n_estimators': [50, 100, 150],
        'regressor__max_depth': [10, 20, 30],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2]
    }

    search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, scoring='r2', verbose=2, random_state=42)
    search.fit(X_train, y_train)

    print(f"Best parameters: {search.best_params_}")

    # Evaluate the model
    y_pred = search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

# Main Function to Run Preprocessing Steps and Analysis
if __name__ == "__main__":
    # Step 1: Load and combine data
    combined_data = load_and_merge_data()

    # Step 2: Handle missing values and remove duplicates
    preprocessed_data = handle_missing_and_duplicates(combined_data)

    # Step 3: Perform feature engineering
    preprocessed_data = feature_engineering(preprocessed_data)

    # Step 4: Clean the data by removing unnecessary columns and converting data types
    preprocessed_data = data_cleaning(preprocessed_data)

    # Step 5: Perform statistical analysis and visualizations
    perform_analysis_and_visualization(preprocessed_data)

    # Step 6: Build and evaluate the machine learning model
    build_and_evaluate_model(preprocessed_data)
