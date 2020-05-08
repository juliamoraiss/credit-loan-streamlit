import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

@st.cache
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df


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
        data = data.drop(columns=['Gender', 'Married', 'Property_Area', 'Loan_ID'])
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
        st.subheader('Prediction')


        st.sidebar.title("Informations about you:")

        name = st.sidebar.text_input("What's your name?")

        income = st.sidebar.number_input("How much do you earn in a month?")

        amount = st.sidebar.number_input("How much do you want to borrow?")

        loan_term = st.sidebar.selectbox("Loan term (months):",
                                          [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])

        dependents = st.sidebar.selectbox("How many dependents do you have?",
                                        ("0","1", "2", "3+"))

        education = st.sidebar.selectbox("Education:",
                                        ("Graduated","Not Graduated"))

        self_employed = st.sidebar.selectbox("Are you a self-employed worker?",
                                        ("Yes","No"))

        apply_button = st.sidebar.button('Will I get a loan?')



if __name__ == '__main__':
    main()