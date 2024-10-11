# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:29:19 2024

@author: LAPTOPCOM
"""
import pickle
import pandas as pd
import streamlit as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
# Load the dataset
data = pd.read_excel("bank.xlsx")

# Encoding Data
label_encoder = preprocessing.LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])


# Split the data into features (X) and target (y)
x = data[['financial_flexibility','credibility', 'competitiveness']]
y = data['class']
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

k=7

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
result_knn = cross_val_score(knn, x, y, cv=k)
#Accuracy
print(result_knn.mean())

filename = 'knn.pkl'
pickle.dump(knn, open(filename,'wb'))
knn.fit(x,y)
pk=knn.predict(x_test)

st.title("Bankruptcy-Prevention")

risk_mapping = {
    'Low': 0,
    'Medium': 0.5,
    'High': 1
}

# Now you can use the assigned numerical values in your predictions or further processing

Financial_flexibility = st.selectbox('Financial_flexibility',('Low','Medium','High'))
Credibility = st.selectbox('Credibility',('Low','Medium','High'))
Competitiveness = st.selectbox('Competitiveness',('Low','Medium','High'))

if st.button('Prevention Type'):
    df = {

        'financial_flexibility': risk_mapping[Financial_flexibility],
        'credibility': risk_mapping[Credibility],
        'competitiveness': risk_mapping[Competitiveness]
    }

    df1 = pd.DataFrame(df,index=[1])
    predictions = knn.predict(df1)

    if predictions.any() == 1:
        prediction_value = 'Non-Bankruptcy'
    else:
        prediction_value = 'Bankruptcy'
    
    st.title("Business type is " + str(prediction_value))