import pandas as pd

app = pd.read_csv('C:/Users/yinji/OneDrive/Desktop/York/Projects/Credit_Risk_Prediction/Data/application_record.csv')
credit = pd.read_csv('C:/Users/yinji/OneDrive/Desktop/York/Projects/Credit_Risk_Prediction/Data/credit_record.csv')

print("Application record loaded:", app.shape)
print("Credit record loaded:", credit.shape)

print(app.head())
print(credit.head())

# If the customer ever had STATUS 2,3,4,5. default = 1 (bad). Otherwise default = 0 (good)
# mark risky rows (True if STATUS is 2,3,4,5)
credit['risky'] = credit['STATUS'].isin(['2', '3', '4', '5'])

print(credit[['ID', 'STATUS', 'risky']].head(10))

# Group by each ID, find if they ever had risky = True
default_table = credit.groupby('ID')['risky'].max().reset_index()

print(default_table.head(10))

# Merge personal info with risky label
merged = app.merge(default_table, on='ID', how='left')

# People who never appeared in credit file: risky = False
merged['risky'] = merged['risky'].fillna(False)

print(merged.head())
print("Merged dataset:", merged.shape)

# check for missing values
print(merged.isna().sum().sort_values(ascending=False).head(10))

# Fill missing OCCUPATION_TYPE with 'Unknown'
merged['OCCUPATION_TYPE'] = merged['OCCUPATION_TYPE'].fillna('Unknown')

# Double-check
print(merged['OCCUPATION_TYPE'].isna().sum())

# check which columns are not numeric
print(merged.select_dtypes(include='object').columns)

# Convert Y/N columns to numbers
merged['CODE_GENDER'] = merged['CODE_GENDER'].map({'M': 1, 'F': 0})
merged['FLAG_OWN_CAR'] = merged['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
merged['FLAG_OWN_REALTY'] = merged['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})


# One-hot encode categorical columns
merged_encoded = pd.get_dummies(
    merged,
    columns=['NAME_INCOME_TYPE',
             'NAME_EDUCATION_TYPE',
             'NAME_FAMILY_STATUS',
             'NAME_HOUSING_TYPE',
             'OCCUPATION_TYPE'],
    drop_first=True
)

print(merged_encoded.shape)
print(merged_encoded.head())

# Separate features and target
X = merged_encoded.drop('risky', axis=1)
y = merged_encoded['risky']

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split data (70% train, 30% test)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Accuracy: 0.9987458956585188
# Classification Report:
#                precision    recall  f1-score   support

#        False       1.00      1.00      1.00    131403
#         True       0.00      0.00      0.00       165

#     accuracy                           1.00    131568
#    macro avg       0.50      0.50      0.50    131568
# weighted avg       1.00      1.00      1.00    131568
# The accuracy looked very high (≈ 0.999), but the confusion matrix and classification report have some issues
# Almost every prediction was “Safe”.
# The model missed all risky customers, recall = 0 for the “True” class.
# Because 99.9 % of the data are safe, the model can get 99 %+ accuracy just by always saying “Safe.”
# High accuracy doesn't make a good model when classes are imbalanced.

# To handle the imbalance and capture rare risky cases, train a Random Forest classifier with
# class_weight='balanced', which gives extra importance to the minority “risky” class.


# Try Random Forest Model 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Drop ID column (not useful for prediction)
merged_encoded = merged_encoded.drop('ID', axis=1)

# Separate features (X) and target (y)
X = merged_encoded.drop('risky', axis=1)
y = merged_encoded['risky']

# Split into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Data split done")
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Create and train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,       
    max_depth=None,         
    random_state=42,         
    class_weight='balanced'  
)
rf_model.fit(X_train, y_train)
print("Model training complete")

# Predict on test data
y_pred_rf = rf_model.predict(X_test)

# Evaluate performance
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))


# Classification Report:
#                precision    recall  f1-score   support

#        False       1.00      1.00      1.00    131403
#         True       0.18      0.46      0.25       165

#     accuracy                           1.00    131568
#    macro avg       0.59      0.73      0.63    131568
# weighted avg       1.00      1.00      1.00    131568
# Accuracy stayed very high (99.7 %), but recall for risky clients increased from 0 to 46 %.
# Use Random Forest Regression and removed the “ID” column eliminated data leakage, making the model more realistic 
# and improving its ability to detect risky customers to 0.46.


# Get importances
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(10)
print("Top 10 important features:\n", top_features)

# Plot 
plt.figure(figsize=(8,5))
top_features.plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Top 10 Most Important Features Affecting Credit Risk')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
