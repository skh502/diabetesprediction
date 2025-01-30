import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import warnings 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


warnings.filterwarnings('ignore')

# Load the diabetes dataset
data = pd.read_csv(r'D:\data\diabetes.csv')
df=data.copy()


# Replace zeros with median
select_col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in data[select_col]:
    data[col] = data[col].replace(0, data[col].median())


# Outlier Removal
outlier_column = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
new_df_cap = data.copy()

for rc in outlier_column:
    # Finding quartiles & range
    q1 = data[rc].quantile(0.25)
    q3 = data[rc].quantile(0.75)
    IQR = q3 - q1
    min_range = q1 - (1.5 * IQR)
    max_range = q3 + (1.5 * IQR)
    
    # Capping (winsorization)
    new_df_cap[rc] = np.where(
        new_df_cap[rc] > max_range,                    #-> x
        max_range,                                     #-> y
        np.where(                                      #-> z
            new_df_cap[rc] < min_range,
            min_range,
            new_df_cap[rc]
        )
    )
# Updating the original dataset after capping
for rc in outlier_column:
    data[rc] = new_df_cap[rc]



# Define the target and features
target = data['Outcome']
features = data.drop('Outcome', axis='columns')


# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=1, stratify=target)


# Initialize and train the SVC model
svc = SVC()  
svc.fit(xtrain, ytrain)
joblib.dump(svc, 'outputmodels/diabetes_svc_model.sav')


# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)
joblib.dump(knn, 'outputmodels/diabetes_knn_model.sav')




