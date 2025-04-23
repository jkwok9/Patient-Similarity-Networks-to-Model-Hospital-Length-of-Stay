# Patient-Similarity-Networks-to-Model-Hospital-Length-of-Stay

# Objective
The current phase of the research focuses on preprocessing and engineering features from structured electronic health records (EHRs) to support graph construction and LOS prediction. Specifically, we aim to generate clean, apply NLP to unstructured clinical notes, and use high-dimensional representations of each patient based on diagnostic and procedural history for use in downstream GNN models. 

# Process

# Physical Attributes Preprocessing
```
import pandas as pd

ad_df = pd.read_csv("/content/admissions.csv")
pt_df = pd.read_csv("/content/patients.csv")
cols = ["subject_id","admission_type","marital_status","race","admittime","dischtime"]
df_phy_ad = ad_df[cols]
df_phy_ad = pd.merge(df_phy_ad,pt_df[["subject_id","gender","anchor_age"]], on = "subject_id", how="outer")
# ----- calculating LOS and appending LOS column ----------
df_phy_ad['admittime'] = pd.to_datetime(df_phy_ad['admittime'])
df_phy_ad['dischtime'] = pd.to_datetime(df_phy_ad['dischtime'])
df_phy_ad["LOS"] = round((df_phy_ad['dischtime'] - df_phy_ad['admittime']).dt.total_seconds() / (3600*24),2)
df_phy_ad.drop(['admittime','dischtime'], axis = 1, inplace = True)
# ----- binary/numeric represenation mapping ------
gender_nums = {"F":1,"M":2,"nan":0}
race_nums = {'WHITE':1, 'OTHER':2, 'BLACK/AFRICAN AMERICAN':3, 'UNABLE TO OBTAIN':4, 'UNKNOWN':5,
 'WHITE - RUSSIAN':6, 'BLACK/CAPE VERDEAN':7,
 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER':8, 'PORTUGUESE':9,
 'WHITE - OTHER EUROPEAN':10, 'HISPANIC/LATINO - PUERTO RICAN':11, 'ASIAN':12,
 'ASIAN - CHINESE':13, 'HISPANIC/LATINO - DOMINICAN':14,
 'HISPANIC/LATINO - SALVADORAN':15, 'BLACK/AFRICAN':16,
 'HISPANIC/LATINO - GUATEMALAN':17, 'ASIAN - SOUTH EAST ASIAN':18,
 'WHITE - BRAZILIAN':19, 'SOUTH AMERICAN':20, 'HISPANIC OR LATINO':21,
 'ASIAN - KOREAN':22, 'BLACK/CARIBBEAN ISLAND':23, 'HISPANIC/LATINO - MEXICAN':24,
 'PATIENT DECLINED TO ANSWER':25, 'HISPANIC/LATINO - CUBAN':26,
 'AMERICAN INDIAN/ALASKA NATIVE':27, 'MULTIPLE RACE/ETHNICITY':28,
 'WHITE - EASTERN EUROPEAN':29, 'HISPANIC/LATINO - HONDURAN':30,
 'HISPANIC/LATINO - CENTRAL AMERICAN':31, 'ASIAN - ASIAN INDIAN':32,
 'HISPANIC/LATINO - COLUMBIAN':33,"nan":0 }
marital_nums = {'WIDOWED':1,'SINGLE':2, 'MARRIED':3, 'DIVORCED':4, "nan":5}
admission_nums = {'URGENT':1, 'EW EMER.':2,'EU OBSERVATION':3, 'OBSERVATION ADMIT':4,
 'SURGICAL SAME DAY ADMISSION':5, 'AMBULATORY OBSERVATION':6, 'DIRECT EMER.':7,
 'DIRECT OBSERVATION':8, 'ELECTIVE':9, "nan":0}
 # ---------- mapping values ------------
df_phy_ad["admission_type"] = df_phy_ad["admission_type"].map(admission_nums)
df_phy_ad["admission_type"] = df_phy_ad["admission_type"].fillna(0)
df_phy_ad["marital_status"] = df_phy_ad["marital_status"].map(marital_nums)
df_phy_ad["marital_status"] = df_phy_ad["marital_status"].fillna(0)
df_phy_ad["race"] = df_phy_ad["race"].map(race_nums)
df_phy_ad["race"] = df_phy_ad["race"].fillna(0)
df_phy_ad["gender"] = df_phy_ad["gender"].map(gender_nums)
df_phy_ad["gender"] = df_phy_ad["gender"].fillna(0)
df_phy_ad["LOS"] = df_phy_ad["LOS"].fillna(0)
unique_ids = "Unique number of Subject_Ids: " + str(len(df_phy_ad["subject_id"].unique()))
print(unique_ids)
# ---- converting to final csv output -------
df_phy_ad.to_csv("PhysicalandAdmission.csv", index = False)
print(df_phy_ad)
```
Going through 20+ csv files contained in this dataset, I have gone through and extracted the physical attribute/ background info of each subject that would be useful into a new csv. Additionaly, I calculated Length of Stay (LOS) by subtracting admission time from discharge time in days (used for prediction later). Since many of these fields were categorical, mapping was used to provide numeric values using custom dictionaries and assigned 0 to any missing or unrecognized entries to keep things consistent.

# ICD processing
After preparing the physical attribute data, the next step focused on extracting and encoding diagnosis ICD codes. Each subject in the dataset can have multiple ICD codes, with each one appearing in a separate row. Total unique ICD codes for all patientswere over 100,000. To make this data usable for modeling, I converted the ICD codes into a one-hot encoded format using a pivot table. Each column represents a unique ICD code, and a value of 1 indicates that the patient had that diagnosis at least once. This creates a uniform, high-dimensional vector for each patient that summarizes their medical history in a structured way. Using this format, models can easily learn patterns across patients based on shared diagnoses, without any implicit ordering or bias that might come from label encoding.

## Challenges
The inital csv of the physical attributes had millions of rows of data. With the addition of the ICD codes, it would add over 100,000 columns of ICD codes, creating a high dimension dataset. This affected the operations that was performed on this data, as performing grouping and pivoting for one-hot encoding became expensive and created the needed for extremely high computing power.

### Solution to challenges
<img src="https://github.com/user-attachments/assets/4ad0c02b-cef3-41ce-ba80-54c1ab21845e" alt="Alt Text" width="450" height="300">

To address this, visualizations was created based on the Length of Stay (LOS) distribution to help identify long-stay patients. The histogram included markers for the mean LOS, one standard deviation above the mean, and two standard deviations above the mean. Based on this, the dataset was filtered to only include patients whose LOS was two standard deviations above the mean, focusing on why the outliers has such a long length of stay. This significantly reduced the size of the data and allowed one-hot encoding to run more efficiently.

```
df_phy = pd.read_csv("/content/PhysicalandAdmission.csv")
df_icd = pd.read_csv("/content/diagnoses_icd.csv")

# ------ dropping all values below the 2nd std from the mean
los_col = 'LOS'
std_los = df_phy[los_col].std()
print("std",std_los)
mean = df_phy[los_col].mean()
print("mean",mean)
print(mean + (2*std_los))
print("2 std above mean", mean + (2*std_los))
df_phy = df_phy[df_phy[los_col] >= (mean + (2*std_los))]
print("unqiue",str(len(df_phy["subject_id"].unique())))
print("max_day:",max(df_phy[los_col]))
# --------- drop icd version 9 -------- #
df_icd = df_icd[df_icd["icd_version"] != 9]
print("icd_unique:",str(df_icd["icd_version"].value_counts()))
print(df_icd)
# ---- one hot encode ------- #
df_icd = df_icd[df_icd['subject_id'].isin(df_phy['subject_id'])]
print("len after filtering: ", len(df_icd['subject_id'].unique()))

# ---------- Step 4: One-hot encode ICD codes (efficiently) ---------- #
df_icd['value'] = 1  # Placeholder column to count 1s for presence

# Optional: Convert to category to reduce memory usage
df_icd['icd_code'] = df_icd['icd_code'].astype('category')

# Create pivot table (one-hot encoding per subject_id)
df_icd_encoded = df_icd.pivot_table(
    index='subject_id',
    columns='icd_code',
    values='value',
    aggfunc='max',
    fill_value=0
)

# Add prefix "d_" to ICD code columns
df_icd_encoded.columns = ['d_' + str(col) for col in df_icd_encoded.columns]

# Bring subject_id back as a column
df_icd_encoded = df_icd_encoded.reset_index()

print("\nOne-hot encoded ICD shape:", df_icd_encoded.shape)

# ---------- Step 5: Merge with physical attributes ---------- #
merged_df = pd.merge(df_phy, df_icd_encoded, on='subject_id', how='left')
merged_df = merged_df.fillna(0)

print("\nFinal merged shape:", merged_df.shape)
print("Sample of final dataset:")
print(merged_df.head())
merged_df = merged_df.fillna(0)
merged_df.to_csv("/content/phy_dicd.csv", index = False)
```
This contains filter the dataset to dropping subjects below the 2 standard deviations above the mean, performing the one-hot encoding using pivot tables, concatenating strings to ICD codes to readability and differentiability between different types of ICD codes, as well as merging with physical attributes csv resulting in a csv with ICD codes added.

```
df_phy = pd.read_csv("/content/PhysicalandAdmission.csv")
df_icd = pd.read_csv("/content/procedures_icd.csv")
df_dicd = pd.read_csv("phy_dicd.csv")

# ------ dropping all values below the 2nd std from the mean
los_col = 'LOS'
std_los = df_phy[los_col].std()
print("std",std_los)
mean = df_phy[los_col].mean()
print("mean",mean)
print(mean + (2*std_los))
print("2 std above mean", mean + (2*std_los))
df_phy = df_phy[df_phy[los_col] >= (mean + (2*std_los))]
print("unqiue",str(len(df_phy["subject_id"].unique())))
print("max_day:",max(df_phy[los_col]))
# --------- drop icd version 9 -------- #
df_icd = df_icd[df_icd["icd_version"] != 9]
print("icd_unique:",str(df_icd["icd_version"].value_counts()))
print(df_icd)
# ---- one hot encode ------- #
df_icd = df_icd[df_icd['subject_id'].isin(df_phy['subject_id'])]
print("len after filtering: ", len(df_icd['subject_id'].unique()))

df_icd['value'] = 1  # Placeholder column to count 1s for presence

# Optional: Convert to category to reduce memory usage
df_icd['icd_code'] = df_icd['icd_code'].astype('category')

# Create pivot table (one-hot encoding per subject_id)
df_icd_encoded = df_icd.pivot_table(
    index='subject_id',
    columns='icd_code',
    values='value',
    aggfunc='max',
    fill_value=0
)

# Add prefix "d_" to ICD code columns
df_icd_encoded.columns = ['p_' + str(col) for col in df_icd_encoded.columns]

# Bring subject_id back as a column
df_icd_encoded = df_icd_encoded.reset_index()

print("\nOne-hot encoded ICD shape:", df_icd_encoded.shape)

# ---------- Step 5: Merge with physical attributes ---------- #
merged_df = pd.merge(df_dicd, df_icd_encoded, on='subject_id', how='left')
merged_df = merged_df.fillna(0)
print("\nFinal merged shape:", merged_df.shape)
print("Sample of final dataset:")
print(merged_df.head())
merged_df = merged_df.fillna(0)
merged_df.to_csv("/content/allICD.csv", index = False)
```
Continuation of ICD one-hot encoding but for the procedures ICD codes.



