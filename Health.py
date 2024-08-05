import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split



st.image(r"ino_img.jpg")

st.title("PREDICTING MEDICAL CONDITIONS")

df = pd.read_csv(r"Health2.csv")

X = df.drop(["Medical Condition","Unnamed: 0"],axis = 1)
y = df["Medical Condition"]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=23)


model = pickle.load(open(r"dt.plk","rb"))

Age = st.slider("Age",min_value=13,max_value=89,step=1)


text5 = {1:"Male",2:"Female"}
#st.subheader(f"Type : {text5[Gender]}")
Gender = st.selectbox("Choose the Gender", text5.keys())
st.markdown(f"<h3 style='font-size:20px;'>Gender Type: {text5[Gender]}</h3>", unsafe_allow_html=True)

text1 ={1:"B-",2:"A+",3:"A-",4:"O+",5:"AB+",6:"AB-",7:"B+",8:"O-"}
Blood_Type = st.selectbox("Choose the Blood Type", text1.keys())
st.markdown(f"<h3 style='font-size:20px;'>Blood Type Type: {text1[Blood_Type]}</h3>",unsafe_allow_html=True)

text = {1:"Blue Cross",2:"Medicare",3:"Aetna",4:"UnitedHealthcare",5:"Cigna"}
Insurance_provider = st.selectbox("Insurance Provider value", text.keys())
st.markdown(f"<h3 style='font-size:20px;'>Insurance Provider Type: {text[Insurance_provider]}</h3>",unsafe_allow_html=True)

#Billing_Amount = st.slider("Billing Amount",min_value=-2008.4921398591305,max_value=52764.276736469175)

text4 = {1:"Urgent",2:"Emergency",3:"Elective"}
Admission_Type = st.selectbox("Choose the Admission Value", text4.keys())
st.markdown(f"<h3 style='font-size:20px;'>Admission Type: {text4[Admission_Type]}</h3>",unsafe_allow_html=True)

text2 = {1:"Paracetamol",2:"Ibuprofen",3:"Aspirin",4:"Penicillin",5:"Lipitor"}
Medication = st.selectbox("Choose the Medication Value", text2.keys())
st.markdown(f"<h3 style='font-size:20px;'>Medication Type: {text2[Medication]}</h3>",unsafe_allow_html=True)

text3 = {1:"Normal",2:"Inconclusive",3:"Abnormal"}
Test_Results = st.selectbox("Choose the Test Results Value", text3.keys())
st.markdown(f"<h3 style='font-size:20px;'>Test Result Type: {text3[Test_Results]}</h3>",unsafe_allow_html=True)


features = np.array([[Age, Gender, Blood_Type, Insurance_provider, Admission_Type, Medication, Test_Results]])#,additional_feature]])
model = DecisionTreeRegressor()
model.fit(X_train, y_train)


medical_condition = model.predict([[Age, Gender, Blood_Type, Insurance_provider, Admission_Type, Medication, Test_Results]])[0]#,additional_feature])[0]
#st.write(f"Predicted Medical Condition: {medical_condition}")

text8 = {1.0:"Cancer",2.0:"Obesity",3.0:"Diabetes",4.0:"Asthma",5.0:"Hypertension",6.0:"Arthritis"}


if st.button("Submit"):
    if medical_condition == 1.0:
        st.write(f"Predicted Medical condition: {medical_condition}")
        st.write(f"Selected Medical Condition Type : {text8[medical_condition]}")
        st.image(r"cancer.jpg",width=200)
    if medical_condition == 2.0:
        st.write(f"Predicted Medical condition: {medical_condition}")
        st.write(f"Selected Medical Condition Type : {text8[medical_condition]}")
        st.image(r"obesity.jpg",width=200)
    if medical_condition == 3.0:
        st.write(f"Predicted Medical condition: {medical_condition}")
        st.write(f"Selected Medical Condition Type : {text8[medical_condition]}")
        st.image(r"diabetes.jpg",width=200)
    if medical_condition == 4.0:
        st.write(f"Predicted Medical condition: {medical_condition}")
        st.write(f"Selected Medical Condition Type : {text8[medical_condition]}")
        st.image(r"asthma.jpg",width=200)
    if medical_condition == 5.0:
        st.write(f"Predicted Medical condition: {medical_condition}")
        st.write(f"Selected Medical Condition Type : {text8[medical_condition]}")
        st.image(r"hypertension.jpg",width=200)
    if medical_condition == 6.0:
        st.write(f"Predicted Medical condition: {medical_condition}")
        st.write(f"Selected Medical Condition Type : {text8[medical_condition]}")
        st.image(r"arthritis.jpg",width=200)
