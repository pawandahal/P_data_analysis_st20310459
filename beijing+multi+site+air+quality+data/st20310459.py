import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Step 1: Load and concatenate data from multiple CSV files
air_quality_data_set_file_paths = [
    r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Aotizhongxin_20130301-20170228.csv',
    r"C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Changping_20130301-20170228.csv",
    r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Dingling_20130301-20170228.csv',
    r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Dongsi_20130301-20170228.csv',
    r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Guanyuan_20130301-20170228.csv',
    r"C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Gucheng_20130301-20170228.csv",
    r"C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Huairou_20130301-20170228.csv",
    r"C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Nongzhanguan_20130301-20170228.csv",
    r"C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Shunyi_20130301-20170228.csv",
    r"C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Tiantan_20130301-20170228.csv",
    r"C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Wanliu_20130301-20170228.csv",
    r"C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Wanshouxigong_20130301-20170228.csv"
]

data_set_frames = [pd.read_csv(file) for file in air_quality_data_set_file_paths]
df = pd.concat(data_set_frames)

# Step 2: Get an overview of the data
print("How much number of row and colum is there in dataset:", df.shape)
print("\nAs well as how much Data Types and Non-null Counts in dataset:")

#step 3:Find the rows number of the dataset
df.info()
print("\nFirst Few Rows of the Dataset:")

#step 4:Find the statistics value
print(df.head())
print("\nThe value of  Statistics contain:")
print(df.describe())


# Step 5: Checking the missing value of data set which is contain in the bejing data set which mainly contain the air quality data 
Air_quality_missing_values = df.isnull().sum()
print("\nFinding missing air quality data  Values in Each Column Before Handling the data set:")
print(Air_quality_missing_values[Air_quality_missing_values > 0])



# Step 5:Try to  handling the missing value 
def handle_missing_air_quality_data_values(column):
    different_type_of_case_value = {
        'float64': lambda col: col.interpolate(method='linear'),
        'int64': lambda col: col.interpolate(method='linear'),
        'object': lambda col: col.fillna(col.mode()[0]),
    }

    return different_type_of_case_value.get(str(column.dtype), lambda col: col)(column)

# using for loop for calculating the missing value after handling of missing value 
for air_quality_column in df.columns:
    df[air_quality_column] = handle_missing_air_quality_data_values(df[air_quality_column])
air_quality_total_missing_after_calculating = df.isnull().sum().sum()
print("\nCalculating the total Missing Values After Handling of missing value :", air_quality_total_missing_after_calculating)


# Step 6: removing the duplicating value in air quality control of pollution
df.drop_duplicates(inplace=True)

# From this step duplicated value will be remove 
duplicates_Value_after_column = df.duplicated().sum()
print("Find the number of Duplicates air quality control value After Removing the air quality value :\n", duplicates_Value_after_column) 

# Step 7: find the number of column value 
data_air_control_columns_to_check = {'year', 'month', 'day', 'hour'}
index = 0
data_air_control_columns_list = list(df.columns)

while index < len(data_air_control_columns_list):
    if data_air_control_columns_to_check.issubset(data_air_control_columns_list):
      
        df['air_quality_data_value_datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df.set_index('air_quality_data_value_datetime', inplace=True)
        break  
    index += 1

print("\nPreprocessing the value")

# Step 8:find the names of  column 
index = 0
while index < len(df.columns):
    col = df.columns[index]
    df.rename(columns={col: col.lower().replace(" ", "_")}, inplace=True)
    index += 1

# Step 9: Output the shape of the dataframe
print("\nFinding  the air quality data set find the number of rows and columns:", df.shape)

# Step 9: Handle Outliers 
air_quality_pollutant_columns = [col for col in df.columns if 'pm' in col or 'pollutant' in col]
print("Finding the missing values in the final result dataset value:\n", df.isnull().sum().sum())


for col in air_quality_pollutant_columns:
    df = df[df[col] <= 500]  # Example threshold; adjust as per data specifics
print("Find the Sample dataset for air pollution:\n", df.head())