import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)


# drop rows with any NaN values
df = df.dropna()
# print(df.info())

# print(df.columns)

df= df.rename(columns ={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'})

df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
print(df.info())

# Extracting a seasonality feature
def date_to_season(date):
    month = date.month
    if(month == 12) or (month ==1) or (month ==2):
        return 'Summer'
    elif(month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif(month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'
                            
# Exercise 1: Map the dates to seasons and drop the Date column

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply the function to the 'Date' column to create a 'Season' column
df['Season'] = df['Date'].apply(date_to_season)

# Drop the 'Date' column as it's no longer needed
df = df.drop(columns=['Date'])

print(df)

# Exercise 2. Define the feature and target dataframes
X = df.drop(columns=['RainToday'], axis=1)
y = df['RainToday']

# Exercise 3. How balanced are the classes?
y.value_counts()

# Exercise 4. What can you conclude from these counts?

"""
How often does it rain annually in Melbourne?
About 23.7% of days (1791 / (5766 + 1791) ≈ 0.237) have rain.

Accuracy if you always predict 'No rain':
You would be correct 76.3% of the time (5766 / (5766 + 1791) ≈ 0.763).

Is this a balanced dataset?
No, it is imbalanced. There are significantly more 'No' than 'Yes' days.

Next steps:
Consider using techniques to handle imbalance, such as:

Resampling (oversampling minority or undersampling majority class)
Using metrics like F1-score, precision, recall
Applying class weights in your models    
 """

# Exercise 5. Split data into training and test sets, ensuring target stratification
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing transformers for numerical and categorical features
# Exercise 6. Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Define separate transformers for both feature types and combine them into a single preprocessing transformer

# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals 
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Exercise 7. Combine the transformers into a single preprocessing column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
# Exercise 8. Create a pipeline by combining the preprocessing with a Random Forest classifier
pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
# Define a parameter grid to use in a cross validation grid search model optimizer
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Perform grid search cross-validation and fit the best model to the training data
#  Select a cross-validation method, ensuring target stratification during validation
cv = StratifiedKFold(n_splits=5, shuffle=True)

# Exercise 9. Instantiate and fit GridSearchCV to the pipeline
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters and best crossvalidation score
print("\nBest parameters found:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Exercise 10. Display your model's estimated score
test_score = grid_search.score(X_test,y_test)
print("Test set score: {:.2f}".format(test_score))

# Exercise 11. Get the model predictions from the grid search estimator on the unseen data
y_pred = grid_search.predict(X_test)

# Exercise 12. Print the classification report
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Exercise 13. Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()