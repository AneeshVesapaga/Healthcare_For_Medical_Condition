import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

st.image(r"ino_img.jpg",width = 150)

st.title("Healthcare For Medical Condition")

df = pd.read_csv(r"Health.csv")

X = df.drop(["Medical Condition","Unnamed: 0"],axis = 1)
y = df["Medical Condition"]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=23)


model = pickle.load(open(r"dt.plk","rb"))

Age = st.slider("Age",min_value=13,max_value=89,step=1)


text5 = {1:"Male",2:"Female"}
Gender = st.selectbox("Choose the Gender", text5.keys())
st.write(f"Type : {text5[Gender]}")

text1 ={1:"B-",2:"A+",3:"A-",4:"O+",5:"AB+",6:"AB-",7:"B+",8:"O-"}
Blood_Type = st.selectbox("Choose the Blood Type", text1.keys())
st.write(f"Type : {text1[Blood_Type]}")

text = {1:"Blue Cross",2:"Medicare",3:"Aetna",4:"UnitedHealthcare",5:"Cigna"}
Insurance_provider = st.selectbox("Insurance Provider value", text.keys())
st.write(f"Type: {text[Insurance_provider]}")

Billing_Amount = st.slider("Billing Amount",min_value=-2008.4921398591305,max_value=52764.276736469175)

text4 = {1:"Urgent",2:"Emergency",3:"Elective"}
Admission_Type = st.selectbox("Choose the Admission Type", text4.keys())
st.write(f"Type : {text4[Admission_Type]}")

text2 = {1:"Paracetamol",2:"Ibuprofen",3:"Aspirin",4:"Penicillin",5:"Lipitor"}
Medication = st.selectbox("Choose the Medication", text2.keys())
st.write(f"Type : {text2[Medication]}")

text3 = {1:"Normal",2:"Inconclusive",3:"Abnormal"}
Test_Results = st.selectbox("Choose the Test Results", text3.keys())
st.write(f"Type : {text3[Test_Results]}")


features = np.array([[Age, Gender, Blood_Type, Insurance_provider, Billing_Amount, Admission_Type, Medication, Test_Results]])#,additional_feature]])
model = DecisionTreeRegressor()
model.fit(X_train, y_train)


medical_condition = model.predict([[Age, Gender, Blood_Type, Insurance_provider, Billing_Amount, Admission_Type, Medication, Test_Results]])[0]
#st.write(f"Predicted Medical Condition: {medical_condition}")

text8 = {1.0:"Cancer",2.0:"Obesity",3.0:"Diabetes",4.0:"Asthma",5.0:"Hypertension",6.0:"Arthritis"}


if st.button("Submit"):
    if medical_condition == 1.0:
        st.write(f"Predicted Medical condition: {medical_condition}")
        st.write(f"Selected Medical Condition Type : {text8[medical_condition]}")
        st.image(r"cancer.jpg",width=250)
    if medical_condition == 2.0:
        st.write(f"Predicted Medical condition: {medical_condition}")
        st.write(f"Selected Medical Condition Type : {text8[medical_condition]}")
        st.image(r"obesity.jpg",width=250)
    if medical_condition == 3.0:
        st.write(f"Predicted Medical condition: {medical_condition}")
        st.write(f"Selected Medical Condition Type : {text8[medical_condition]}")
        st.image(r"diabetes.jpg",width=250)
    if medical_condition == 4.0:
        st.write(f"Predicted Medical condition: {medical_condition}")
        st.write(f"Selected Medical Condition Type : {text8[medical_condition]}")
        st.image(r"asthma.jpg",width=250)
    if medical_condition == 5.0:
        st.write(f"Predicted Medical condition: {medical_condition}")
        st.write(f"Selected Medical Condition Type : {text8[medical_condition]}")
        st.image(r"hypertension.jpg",width=250)
    if medical_condition == 6.0:
        st.write(f"Predicted Medical condition: {medical_condition}")
        st.write(f"Selected Medical Condition Type : {text8[medical_condition]}")
        st.image(r"arthritis.jpg",width=250)
