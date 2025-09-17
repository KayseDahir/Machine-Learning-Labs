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