
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import folium
from folium.plugins import MarkerCluster

# adding the css component for making more interactive component
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f8f9fa;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        font-family: 'Poppins', sans-serif;
        color: #3c4043;
        font-weight: 600;
    }
    .stSidebar {
        font-family: 'Poppins', sans-serif;
        color: #3c4043;
    }
    .stButton button {
        font-family: 'Poppins', sans-serif;
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    .stMarkdown {
        font-family: 'Poppins', sans-serif;
        color: #3c4043;
    }
    .stAlert {
        font-family: 'Poppins', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load and preprocess data
def data_handling(data_files):
    # Initialize an empty list to store the dataframes
    air_quality_data_list = []
    for data_file in data_files:
        df = pd.read_csv(data_file)
        air_quality_data_list.append(df)
    
    # Combine all datasets into one DataFrame
    combination_air_quality_df = pd.concat(air_quality_data_list, ignore_index=True)
    
    # Handle missing values (forward fill)
    combination_air_quality_df.fillna(method='ffill', inplace=True)
    
    # Remove duplicate entries
    combination_air_quality_df.drop_duplicates(inplace=True)
    
    # Feature engineering (if 'air_quality_year' column exists)
    if 'air_quality_year' in combination_air_quality_df.columns:
        combination_air_quality_df['air_quality_year'] = pd.to_datetime(combination_air_quality_df['air_quality_year'], errors='coerce')
        combination_air_quality_df['air_quality_Month'] = combination_air_quality_df['air_quality_year'].dt.month
        combination_air_quality_df.dropna(subset=['air_quality_year'], inplace=True)
    
    return combination_air_quality_df

# Function to create a map visualization
def map_state_air_quality_beijing(data):
    # Initialize the map centered around a default location (latitude, longitude)
    air_quality_map_center = [data['latitude'].mean(), data['longitude'].mean()]
    my_map = folium.Map(location=air_quality_map_center, zoom_start=10)
    
    # Add markers for each location in the dataset
    air_quality_marker_cluster = MarkerCluster().add_to(my_map)
    for _, row in data.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Station: {row['Station']}, {row['air_quality_year']}, Value: {row.get('value', 'N/A')}"
        ).add_to(air_quality_marker_cluster)
    
    return my_map

# Streamlit App
def main():
    st.markdown("<h1 style='text-align: center; color: #3c4043;'>Air Quality Data Analysis System</h1>", unsafe_allow_html=True)
    st.sidebar.title("Option items listed")

    # Upload CSV files for different sections (one or more files)
    air_quality_upload_csv_file = st.sidebar.file_uploader("Manual uploading the csv file and user friendly environment:", type=["csv"], accept_multiple_files=True)

    # Process uploaded files
    if air_quality_upload_csv_file:
        data = data_handling(air_quality_upload_csv_file)
        st.success(f"{len(air_quality_upload_csv_file)} file(s) loaded and processed successfully!")
        # Show dataset insights
        st.markdown("<h2 style='text-align: center; color: #3c4043;'>Dataset Information</h2>", unsafe_allow_html=True)
        air_quality_row, columns = data.shape
        st.write(f"The dataset contains in air quality of china-beijing**{air_quality_row} air_quality_row** and **{columns} columns**.")
        st.write("The columns in the dataset of air quality data set for china-beijing:")
        st.write(data.columns.tolist())

        # Show data types and check for missing values
        st.markdown("<h3 style='text-align: center; color: #3c4043;'>Data Types and Missing Values</h3>", unsafe_allow_html=True)
        st.write(data.dtypes)
        st.write("\n** Air quality Missing Values:**")
        st.write(data.isnull().sum())

        # Show raw data
        if st.sidebar.checkbox("Task 1 which mainly includes"):
            st.markdown("<h3 style='text-align: center; color: #3c4043;'>Dataset Preview</h3>", unsafe_allow_html=True)
            st.dataframe(data.head())

        # Show summary statistics
        if st.sidebar.checkbox("Exploratory Data Analysis (EDA)"):
            st.markdown("<h3 style='text-align: center; color: #3c4043;'>Summary Statistics</h3>", unsafe_allow_html=True)
            st.write(data.describe())

            # Display graphs for each numeric column
            st.markdown("<h3 style='text-align: center; color: #3c4043;'>Visualizations</h3>", unsafe_allow_html=True)
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

            for column in numeric_columns:
                st.markdown(f"<h5 style='text-align: center; color: #00FFFF;'>Visualization for {column}</h5>", unsafe_allow_html=True)
                plt.figure(figsize=(10, 6))
                sns.histplot(data[column], kde=True, color='skyblue', bins=30)
                st.pyplot(plt)

                # Box Plot
                plt.figure(figsize=(10, 6))
                sns.boxplot(data[column], color='lightgreen')
                st.pyplot(plt)

        # Show Map if relevant columns are present
        if 'latitude' in data.columns and 'longitude' in data.columns:
            st.markdown("<h3 style='text-align: center; color: #3c4043;'>Geographical Locations on Map</h3>", unsafe_allow_html=True)
            st.subheader("Map Showing Locations from Dataset")
            map_data = map_state_air_quality_beijing(data)
            st_folium(map_data, width=725, height=500)

        # Machine Learning Model Building
        if st.sidebar.checkbox("Machine Learning Model Building"):
            st.markdown("<h3 style='text-align: center; color: #3c4043;'>Machine Learning Model Building</h3>", unsafe_allow_html=True)
            target_column = st.selectbox("Select Target Variable:", numeric_columns)

            features = data.drop(columns=['air_quality_year', 'Station', target_column], errors='ignore')
            target = data[target_column]

            # Remove rows where the target variable contains NaN
            data_clean = data.dropna(subset=[target_column])
            features_clean = data_clean.drop(columns=['air_quality_year', 'Station', target_column], errors='ignore')
            target_clean = data_clean[target_column]

            # Encoding categorical features (if any)
            features_clean = pd.get_dummies(features_clean, drop_first=True)

            # Handle missing values using SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            features_imputed = imputer.fit_transform(features_clean)

            # Feature scaling
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_imputed)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_clean, test_size=0.2, random_state=42)

            # Choose model
            model_type = st.selectbox("Select Machine Learning Model:", ["Linear Regression", "Random Forest"])
            model = LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(random_state=42)

            # Fit the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Model Evaluation
            st.markdown("<h3 style='text-align: center; color: #3c4043;'>Model Evaluation</h3>", unsafe_allow_html=True)
            st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
            st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
            st.write("R-Squared:", r2_score(y_test, y_pred))

if __name__ == '__main__':
    main()
