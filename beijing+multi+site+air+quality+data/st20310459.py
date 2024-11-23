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