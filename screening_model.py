#importing basic libraries of machine learning
from xml.parsers.expat import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#loading the dataset (UCI Autism Screening Child Dataset)
from scipy.io import arff
data = arff.loadarff("Autism-Child-Data.arff")
df = pd.DataFrame(data[0])
for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode('utf-8')
df.to_csv("autism_child.csv", index=False)
df = pd.read_csv("autism_child.csv")

#performing EDA (Exploratory Data Analysis)

def EDA(df):
    print("Shape of the dataset", df.shape)
    print("Features of the dataset", df.columns)
    print("Data types of features", df.dtypes)
    print("reading the dataset", df.head())
    print("overall information",df.info())
    print("descriptive statistics of the dataset", df.describe())
    print("checking for null values", df.isnull().sum())

EDA(df)   

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()


def data_prepretion(df):
    df=df.drop(columns=["age_desc","relation","contry_of_res","used_app_before","ethnicity","result"])
    #removed the unnecessary columns which are not required for the model building
    print(df.head())
    #label encoding the categorical features
    for col in df.select_dtypes(include="object").columns:
        df[col]=label.fit_transform(df[col])
    print(df.head())
    #checking for imbalance data
    print(df["Class/ASD"].value_counts()) #0=151 , 1=141 no need 
    return df

df = data_prepretion(df)   # store processed dataframe


def data_visualization(df):
    sns.countplot(x="Class/ASD", data=df)
    plt.title("visualization of target variable")
    plt.xlabel("Class/ASD")
    plt.ylabel("count")
   # plt.show()
    #stop visualization , i will study and then comeback.

data_visualization(df) 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import decision_tree_classifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def model_building(df):
    X=df.drop(columns=["Class/ASD"])
    Y=df["Class/ASD"]
    X_train , X_test , Y_train , Y_test=train_test_split(X,Y , test_size=0.3, random_state=42)
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=None,    
      
        random_state=42    
    )
    model.fit(X_train,Y_train)
        
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    print("Training Accuracy:", accuracy_score(Y_train, Y_train_pred))
    print("Testing Accuracy :", accuracy_score(Y_test, Y_test_pred))
    confusion_matrix_test = confusion_matrix(Y_test, Y_test_pred)
    print("Confusion Matrix:\n", confusion_matrix_test)
    print("Classification Report:\n", classification_report(Y_test, Y_test_pred))
    return model


    
model = model_building(df)

import pickle
with open("autism_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")




