import streamlit as st
import os
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

@st.cache
def load_data(dataset):
    """
    Receives a csv file and return a pandas Dataframe.
    """

    df = pd.read_csv(dataset)
    return df

loan_term_months = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
dependents_label = {'0': 0, '1': 1, '2': 2, '3+': 3}
education_labels = {'Graduate': 0, 'Not Graduate': 1}
self_employed_label = {'Yes': 0, 'No': 1}
credit_history_labels = {'No': 0, 'Yes': 1}


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_keys(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

@st.cache
def load_predict_model():
    """
    Load LGBM Model for loan prediction
    """
    loaded_model = joblib.load('model-lgbm.pkl')
    return loaded_model

pipeline = load_predict_model()

def main():
    """
    Menu for the app
    """

    st.title("Loan Predict App")

    # Menu

    menu = ['EDA', 'Model', 'Prediction']
    choices = st.sidebar.selectbox('Select:', menu)

    if choices == 'EDA':
        st.subheader('EDA')

        st.write('This is the DataFrame that was used in this predict model:')
        data = load_data('./data/train_ctrUa4K.csv')
        data = data.drop(columns=['Gender', 'Married', 'Property_Area', 'Loan_ID', 'CoapplicantIncome'])
        st.dataframe(data)

        st.write('Here you can check some statistical informations about the data:')
        if st.checkbox('Show summary'):
            st.write(data.describe())

        st.write('Is the dataset balanced?')
        if st.checkbox('Value Count Plot'):
            target = data['Loan_Status'].value_counts()
            fig = go.Figure([go.Bar(x=['Yes', 'No'], y=target, marker_color=['indianred','lightsalmon'])])
            st.plotly_chart(fig, use_container_width=True)

        st.write('Missing values for each column:')
        if st.checkbox('Missing Values'):
            na_values = pd.DataFrame(data.isna().sum())
            na_values.columns = ['missing_values_count']
            st.table(na_values)


    if choices == 'Model':
        st.subheader('Model')
    
    if choices == 'Prediction':

        st.subheader("Please, fill the spaces below to check if you're able to get a loan:")
        st.text('\n')

        name = st.text_input("What's your name?")

        income = st.number_input("How much do you earn in a month?")

        amount = st.number_input("How much do you want to borrow?")

        loan_term = st.selectbox("Loan term (months):", loan_term_months)

        dependents = st.selectbox("How many dependents do you have?", tuple(dependents_label.keys()))


        education = st.selectbox("Education:", tuple(education_labels.keys()))


        self_employed = st.selectbox("Are you a self-employed worker?", tuple(self_employed_label.keys()))


        credit_history = st.selectbox("Do you have your name on SERASA?", tuple(credit_history_labels.keys()))
        credit_history = get_value(credit_history, credit_history_labels)


        st.text('\n')
        apply_button = st.button('Will I get a loan?')

        user = pd.DataFrame({'Dependents': dependents,
                                    'Education': education,
                                    'Self_Employed': self_employed,
                                    'ApplicantIncome': income,
                                    'LoanAmount': amount,
                                    'Loan_Amount_Term': loan_term,
                                    'Credit_History': credit_history},
                                 index=[name])
                              


        if apply_button:
            prob = pipeline.predict_proba(user)[:, 1][0]
            treshold = 0.7
            if prob > treshold:
                st.write('Hello,',name, ', unfortunately we are not able to get the loan.')
                st.write(prob)
            else:
                st.write('Congratulations',name,'!!! You are able to get to borrow $',str(amount))

if __name__ == '__main__':
    main()
