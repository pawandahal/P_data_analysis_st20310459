import pandas as pd

# Task 1: Data Handling (Load and Merge Data)
def load_and_merge_data():
    # List of CSV file paths
    file_paths = [
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Aotizhongxin_20130301-20170228.csv',
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Changping_20130301-20170228.csv',
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Dingling_20130301-20170228.csv',
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Dongsi_20130301-20170228.csv',
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Guanyuan_20130301-20170228.csv',
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Gucheng_20130301-20170228.csv',
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Huairou_20130301-20170228.csv',
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Nongzhanguan_20130301-20170228.csv',
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Shunyi_20130301-20170228.csv',
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Tiantan_20130301-20170228.csv',
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Wanliu_20130301-20170228.csv',
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Wanshouxigong_20130301-20170228.csv'
    ]
    
    # Load and combine all datasets
    # dataframes = [pd.read_csv(file) for file in file_paths]
    # combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Print the combined DataFrame
    print("Combined DataFrame (first 5 rows):")
    print(combined_df.head())  # Print the first 5 rows of the combined DataFrame
    
    return combined_df

# Task 2: Handle Missing Values and Remove Duplicates
def handle_missing_and_duplicates(df):
    # Check for missing values
    print("\nMissing Values Count per Column:")
    print(df.isnull().sum())  # Count of missing values in each column

    # Option 1: Drop rows with any missing values
    # df = df.dropna()

    # Option 2: Fill missing values using forward fill (you can use backward fill 'bfill' as well)
    df.fillna(method='ffill', inplace=True)

    # Remove duplicate rows (if there are any)
    df.drop_duplicates(inplace=True)

    # After handling missing values and duplicates, print the resulting data
    print("\nData after handling missing values and removing duplicates (first 5 rows):")
    print(df.head())  # Show the top 5 rows after cleaning
    
    return df

# Main Function to Run the Tasks
if __name__ == "__main__":
    # Load and combine data
    combined_data = load_and_merge_data()

    # Handle missing values and remove duplicates
    cleaned_data = handle_missing_and_duplicates(combined_data)

    # (Optional) Save the cleaned data to a new CSV file
    output_path = r'C:\Users\Acer\Desktop\cleaned_air_quality_data.csv'
    cleaned_data.to_csv(output_path, index=False)

    print(f"\nCleaned data has been saved to '{output_path}'.")
