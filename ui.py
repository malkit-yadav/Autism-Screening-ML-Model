import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("autism_model.pkl", "rb"))

st.title("Autism Screening Questionnaire")

st.write("Answer the following questions")

def yes_no(question):
    return 1 if st.radio(question, ["No", "Yes"]) == "No" else 0

A1 = yes_no("A1: Does the child look at you when you call his/her name?")
A2 = yes_no("A2: Does the child smile back when you smile?")
A3 = yes_no("A3: Does the child show interest in other children?")
A4 = yes_no("A4: Does the child point to share interest?") 
A5 = yes_no("A5: Does the child bring objects to show you?")
A6 = yes_no("A6: Does the child respond to social cues?")
A7 = yes_no("A7: Does the child imitate actions?")
A8 = yes_no("A8: Does the child make eye contact?")
A9 = yes_no("A9: Does the child understand gestures?")
A10 = yes_no("A10: Does the child engage in pretend play?")

age = st.number_input("Age", 4, 11)
gender = st.radio("Gender", ["Female", "Male"])
gender = 1 if gender == "Male" else 0

jundice = st.radio("Jundice at birth?", ["No", "Yes"])
jundice = 1 if jundice == "Yes" else 0

austim = st.radio("Family history of autism?", ["No", "Yes"])
austim = 1 if austim == "Yes" else 0

if st.button("Sumbit For Results"):
    features = np.array([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,age,gender,jundice,austim]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Screening Result: ASD Positive")
    else:
        st.success("Screening Result: ASD Negative")  
 