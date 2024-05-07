import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


#We are using the "Raw_data" variable in our Python code to access the uncleaned data that we are reading from a CSV file.
Raw_data = pd.read_csv('BankChurners.csv')

#Examining the first few(5) rows the data
Raw_data.head()

#As we examined the values of the dataset by above piece of code, we found the rows with Unknown values.
#Now fining the columns which has Unknown values in the dataset using the below code
Unknown_values = []

# Iterating through columns and checking for "Unknown" values
for column in Raw_data.columns:
    if (Raw_data[column] == 'Unknown').any():
        Unknown_values.append(column)

# Printing the columns with "Unknown" values
print("Columns with 'Unknown' values:", Unknown_values)

#Replacing the Missing or Unknown Values in the dataset with NA because, 
#the value 'Unknown' is not noticeable technically while performing the processing
Raw_data['Marital_Status'] = Raw_data['Marital_Status'].replace('Unknown', pd.NA)
Raw_data['Education_Level'] = Raw_data['Education_Level'].replace('Unknown', pd.NA)
Raw_data['Income_Category'] = Raw_data['Income_Category'].replace('Unknown', pd.NA)


#looking at the overview of the data types, structure, and presence of any missing values in the columns.
Raw_data.info()


#Counting the missing values in the corresponding columns of the DataFrame.
Raw_data.isnull().sum()

#Looking for how much of each column's data is missing as a percentage of the total number of rows.
Raw_data.isna().sum() / len(Raw_data)


#generating a heatmap that visually represents the frequency of missing data in the dataset.
#This shows whether we can ignore the missing data or to fill them with values, so those value do not mislead the further analysis 
plt.figure(figsize=(10,6))
sns.heatmap(Raw_data.isna().transpose(),
            cmap="cividis",
            cbar_kws={'label': 'Missing Data'})
#The below code saves the visualization of missing data in the machine 
#plt.savefig('Missing_values.png')


#By looking at the visualisation and count of missing values in each column, we came to the conclusion to fill the missing data. 
Raw_data['Marital_Status'].fillna(Raw_data['Marital_Status'].mode()[0], inplace=True)
Raw_data['Education_Level'].fillna(Raw_data['Education_Level'].mode()[0], inplace=True)
Raw_data['Income_Category'].fillna(Raw_data['Income_Category'].mode()[0], inplace=True)


# After handling the missing values rechecking the data set info, making sure the data is cleaned completely 
Raw_data.info()


#Examining the first few(5) rows the data after cleaning
Raw_data.head()


# Calculating the mean for 'Customer_Age' because high or low mean could indicate data entry errors or outliers in the age values.
customer_age_mean = Raw_data['Customer_Age'].mean()
print("Mean Customer Age:", customer_age_mean)


# Calculating the median for 'Credit_Limit' to analyse the distribution of credit limits within the customer base and their relationship to churn behavior.
credit_limit_median = Raw_data['Credit_Limit'].median()
print("Median Credit Limit:", credit_limit_median)

# Calculating the standard deviation for 'Total_Trans_Amt' It displays the degree to which the mean is deviated from by each transaction amount. 
#To evaluate the diversity of the data
total_trans_amt_std = Raw_data['Total_Trans_Amt'].std()
print("Standard Deviation of Total Transaction Amount:", total_trans_amt_std)


# Calculating the minimum and maximum for 'Total_Ct_Chng_Q4_Q1'
total_ct_chng_min = Raw_data['Total_Ct_Chng_Q4_Q1'].min()
total_ct_chng_max = Raw_data['Total_Ct_Chng_Q4_Q1'].max()
print("Min Total Count Change:", total_ct_chng_min)
print("Max Total Count Change:", total_ct_chng_max)


# Calculating the 25th and 75th percentiles for 'Total_Trans_Ct'
total_trans_ct_25th = Raw_data['Total_Trans_Ct'].quantile(0.25)
total_trans_ct_75th = Raw_data['Total_Trans_Ct'].quantile(0.75)
print("25th Percentile Total Transaction Count:", total_trans_ct_25th)
print("75th Percentile Total Transaction Count:", total_trans_ct_75th)


# Calculate the covariance matrix for selected numeric attributes
cov_matrix = Raw_data[['Customer_Age', 'Credit_Limit', 'Total_Trans_Amt']].cov()
print("Covariance Matrix:")
print(cov_matrix)


# Calculate the correlation matrix for selected numeric attributes
correlation_matrix = Raw_data[['Customer_Age', 'Credit_Limit', 'Total_Trans_Amt']].corr()
print("Correlation Matrix:")
print(correlation_matrix)


# Group the data by 'Income_Category' and calculate the average 'Credit_Limit' in each group
income_credit_avg = Raw_data.groupby('Income_Category')['Credit_Limit'].mean()
print("Average Credit Limit by Income Category:")
print(income_credit_avg)

# Calculate the z-scores for 'Total_Trans_Amt'
total_trans_amt_zscores = (Raw_data['Total_Trans_Amt'] - Raw_data['Total_Trans_Amt'].mean()) / Raw_data['Total_Trans_Amt'].std()
print("Z-scores for Total Transaction Amount:")
print(total_trans_amt_zscores)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.countplot(data=Raw_data, x='Gender')
plt.title('Gender Distribution')

plt.subplot(1, 2, 2)
sns.countplot(data=Raw_data, x='Education_Level', hue='Attrition_Flag')
plt.xticks(rotation=90)
plt.title('Education Level vs. Churn')

#The below code saves the Bar plot visualization in the machine 
#plt.savefig('Visualisation1_Barplot.png')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(data=Raw_data, x='Customer_Age', kde=True)
plt.title('Customer Age Distribution')

plt.subplot(1, 2, 2)
sns.histplot(data=Raw_data, x='Credit_Limit', kde=True)
plt.title('Credit Limit Distribution')

#The below code saves the Histogram visualization in the machine 
#plt.savefig('Visualisation2_Histogram.png')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=Raw_data, x='Attrition_Flag', y='Total_Trans_Amt')
plt.title('Total Transaction Amount by Churn')

plt.subplot(1, 2, 2)
sns.boxplot(data=Raw_data, x='Attrition_Flag', y='Total_Ct_Chng_Q4_Q1')
plt.title('Total Count Change by Churn')

#The below code saves the Box plot visualization in the machine 
#plt.savefig('Visualisation3_Boxplot.png'


temp = Raw_data[['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 
                 'Months_Inactive_12_mon','Credit_Limit', 'Total_Revolving_Bal', 'Total_Trans_Amt']]
plt.figure(figsize = (14, 9))
correlation = temp.corr()
sns.heatmap(correlation, annot = True, cmap = 'coolwarm', vmin=-1, vmax=1)
plt.title('correlation heatmap')
plt.show()

#The below code saves the Correlation heatmap visualization in the machine 
#plt.savefig('Visualisation4_Correlation_heatmap.png')

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot with color-coded points
scatter = ax.scatter(Raw_data['Total_Trans_Amt'], Raw_data['Total_Trans_Ct'], Raw_data['Credit_Limit'],
                     c=Raw_data['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0))

ax.set_xlabel('Total Transaction Amount')
ax.set_ylabel('Total Transaction Count')
ax.set_zlabel('Credit Limit')

# Create a colorbar and set the label
cbar = fig.colorbar(scatter)
cbar.set_label('Customer Status (Attrited Customer = 1, Existing Customer = 0)')

plt.show()

#The below code saves the 3D-Scatter plot visualization in the machine 
#plt.savefig('Visualisation5_Scatterplot.png')

sns.pairplot(Raw_data, hue='Attrition_Flag', plot_kws={'alpha': 0.5})

#The below code saves the Pair plot visualization in the machine 
#plt.savefig('Visualisation6_Pairplot.png')


# Calculate z-scores for numerical features
z_scores = zscore(Raw_data[['Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1']])

# Absolute z-scores to identify outliers
abs_z_scores = np.abs(z_scores)

# Define a threshold for identifying outliers
threshold = 3
outliers = (abs_z_scores > threshold).any(axis=1)

# Remove outliers from the dataset
Raw_data_no_outliers = Raw_data[~outliers]

# Box plot after handling outliers using z-score
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='Attrition_Flag', y='Total_Trans_Amt', data=Raw_data_no_outliers)
plt.title('Total Transaction Amount by Churn (After Z-Score Outlier Handling)')

plt.subplot(1, 2, 2)
sns.boxplot(x='Attrition_Flag', y='Total_Ct_Chng_Q4_Q1', data=Raw_data_no_outliers)
plt.title('Total Count Change by Churn (After Z-Score Outlier Handling)')
plt.show()
#plt.savefig('ZscoreOH.png')

# Calculate IQR for numerical features
Q1 = Raw_data[['Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1']].quantile(0.25)
Q3 = Raw_data[['Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1']].quantile(0.75)
IQR = Q3 - Q1

# Identify and remove outliers based on IQR
outliers = ((Raw_data[['Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1']] < (Q1 - 1.5 * IQR)) | (Raw_data[['Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1']] > (Q3 + 1.5 * IQR))).any(axis=1)
Raw_data_no_outliers = Raw_data[~outliers]

# Box plot after handling outliers using IQR
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='Attrition_Flag', y='Total_Trans_Amt', data=Raw_data_no_outliers)
plt.title('Total Transaction Amount by Churn (After IQR Outlier Handling)')

plt.subplot(1, 2, 2)
sns.boxplot(x='Attrition_Flag', y='Total_Ct_Chng_Q4_Q1', data=Raw_data_no_outliers)
plt.title('Total Count Change by Churn (After IQR Outlier Handling)')
plt.show()
#plt.savefig('IQROH.png')


# Encode categorical variables
le = LabelEncoder()
for column in Raw_data.columns:
    if Raw_data[column].dtype == 'object':
        Raw_data[column] = le.fit_transform(Raw_data[column])

# Splitting the data into features and target
X = Raw_data.drop('Attrition_Flag', axis=1)  # Assuming 'Attrition_Flag' is the target
y = Raw_data['Attrition_Flag']

# Train a Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X, y)

# Get feature importances and sort them
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Select top 10 features
top_10_features = feature_importances.head(10).index.tolist()

print("Top 10 Features:", top_10_features)
top_10_features = ['Total_Trans_Ct', 'Total_Trans_Amt', 'Total_Revolving_Bal', 'Credit_Limit',
    'Months_on_book', 'Months_Inactive_12_mon', 'Customer_Age', 
    'Contacts_Count_12_mon', 'Total_Relationship_Count', 'Dependent_count']



# Encode categorical variables
le = LabelEncoder()
for column in Raw_data.columns:
    if Raw_data[column].dtype == 'object':
        Raw_data[column] = le.fit_transform(Raw_data[column])

# Splitting the data into features and target
X = Raw_data.drop('Attrition_Flag', axis=1)  # Assuming 'Attrition_Flag' is the target
y = Raw_data['Attrition_Flag']

# Train an XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X, y)

# Get feature importances
feature_importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)

print("Feature Importances:\n", feature_importances)


# Assuming you have a list of top 10 features. Replace this with your actual list.
top_10_features = ['Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Revolving_Bal', 'Total_Ct_Chng_Q4_Q1', 'Total_Relationship_Count', 'Avg_Utilization_Ratio', 'Total_Amt_Chng_Q4_Q1', 'Credit_Limit', 'Avg_Open_To_Buy', 'Customer_Age']

# Target variable column name (replace 'Target' with the actual column name)
target_column = 'Target'

# Splitting the dataset
#X = data[top_10_features]
#y = data[target_column]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_classifier.predict(X_test)

# Model Evaluation
accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

# Output the performance
print(f'Accuracy of Decision Tree Classifier: {accuracy_dt}')
print(f'F1 Score of Decision Tree Classifier: {f1_dt}')


# Encode categorical variables
le = LabelEncoder()
for column in Raw_data.columns:
    if Raw_data[column].dtype == 'object':
        Raw_data[column] = le.fit_transform(Raw_data[column])

# Selecting top 10 features for testing dataset
selected_features_test = [
'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Revolving_Bal', 'Total_Ct_Chng_Q4_Q1', 'Total_Relationship_Count', 'Avg_Utilization_Ratio', 'Total_Amt_Chng_Q4_Q1', 'Credit_Limit', 'Avg_Open_To_Buy', 'Customer_Age'
]
X_selected = Raw_data[selected_features_test]
y = Raw_data['Attrition_Flag']


# Split the dataset into training (80%) and testing (20%) sets
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost Classifier
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_classifier.fit(X_train_sel, y_train_sel)

# Make predictions
y_pred = xgb_classifier.predict(X_test_sel)

# Evaluate the model
accuracy = accuracy_score(y_test_sel, y_pred)
f1 = f1_score(y_test_sel, y_pred, average='weighted')

print(f'Accuracy of XGBoost Classifier: {accuracy}')
print(f'F1 Score of XGBoost Classifier: {f1}')


# Load the dataset
# Raw_data = pd.read_csv('your_dataset.csv')

# Initialize Label Encoder and encode object-type columns
le = LabelEncoder()
for column in Raw_data.columns:
    if Raw_data[column].dtype == 'object':
        Raw_data[column] = le.fit_transform(Raw_data[column])

# List of top 10 features identified by Random Forest (or any other method)
# Replace with your actual feature names
top_10_features = ['Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Revolving_Bal', 'Total_Ct_Chng_Q4_Q1', 'Total_Relationship_Count', 'Avg_Utilization_Ratio', 'Total_Amt_Chng_Q4_Q1', 'Credit_Limit', 'Avg_Open_To_Buy', 'Customer_Age']

# Splitting the data into features and target
X = Raw_data[top_10_features]  # Select only the top 10 features
y = Raw_data['Attrition_Flag']

# Splitting the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with Support Vector Classifier (SVC)
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)

# Predictions
y_pred_svc = svc_classifier.predict(X_test)

# Model Evaluation for SVC
accuracy_svc = accuracy_score(y_test, y_pred_svc)
f1_svc = f1_score(y_test, y_pred_svc, average='weighted')

print(f'Accuracy of SVM: {accuracy_svc}')
print(f'F1 Score of SVM: {f1_svc}')


# List of models and their corresponding metrics
models = ['Decision Tree', 'XGBoost', 'SVM']
accuracy_scores = [accuracy_dt, accuracy, accuracy_svc]
f1_scores = [f1_dt, f1, f1_svc]

# Bar plot for accuracy comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracy_scores, palette='viridis')
plt.title('Accuracy Comparison of Models')
plt.ylim(0, 1)
plt.show()
#plt.savefig('BarplotAccuracy.png')



# Bar plot for F1 score comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=f1_scores, palette='viridis')
plt.title('F1 Score Comparison of Models')
plt.ylim(0, 1)
plt.show()
#plt.savefig('BarplotF1score.png')


# Confusion matrix for each model
conf_matrices = [confusion_matrix(y_test, y_pred_dt),
                 confusion_matrix(y_test, y_pred),
                 confusion_matrix(y_test_sel, y_pred_svc)]

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (model, conf_matrix) in enumerate(zip(models, conf_matrices)):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'Confusion Matrix - {model}')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')

plt.tight_layout()
plt.show()
#plt.savefig('HeatMapConfusion_matrix.png')



