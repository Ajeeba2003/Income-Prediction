import streamlit as st 
import joblib
import pandas as pd 
import numpy as np


Employment=joblib.load("workEncode.pkl")
Qualification=joblib.load("education.pkl")
ME=joblib.load("maritalEncode.pkl")
job=joblib.load("occupationEncode.pkl")
relation=joblib.load('relationEncoder.pkl')
ethnic=joblib.load('raceEcoder.pkl')
gender=joblib.load("genderEncode.pkl")
native=joblib.load("countryEncode.pkl")
Dte=joblib.load("decision_tree.pkl")






st.markdown("""
    <style>
        .main { background-color: #b8e3a9; 
            }   
        .stButton>button { background-color: #b8e3a9; 
        color: white; 
        font-size: 18px; 
        border-radius: 10px; 
        font-family: 'Times New Roman', serif;
        text-align: center;
        }
        .stButton>button:hover { background-color: #ff2020; }

         h1 { text-align: center;
         font-size: px; 
         font-weight: bold;
         color:  red; 
         font-family: 'Times New Roman', serif;
         background-color:#a9cce3 ;
          
          }
        .success-msg { text-align: center; font-size: 30px; font-weight: bold; color: green; }
        .warning-msg { text-align: center; font-size: 30px; font-weight: bold; color: red; }
    </style>
""", unsafe_allow_html=True)

st.title('Income Prediction App')

st.markdown("""
    <style>
        .stNumberInput, .stSelectbox, .stRadio {
            margin-bottom: 20px !important;  /* Adjust spacing */
        }
        
        div[data-baseweb="select"]  {
            background-color: #f5f5f5; /* Light gray */
            border: 2px solid #a9cce3 ;
            border-radius: 10px;
            # padding: 5px;
        }
        div[data-testid="stNumberInputContainer"]  {
            background-color: #f5f5f5; 
            border: 2px solid #a9cce3;
            border-radius: 10px; 
            # padding: 5px; 
    }

    </style>
""", unsafe_allow_html=True)


col1, spacer, col2 = st.columns([1, 0.1, 1])


with col1:
       age=st.number_input("**Enter your Age:**",value=None,placeholder=' ')
       workclass=st.selectbox("**Select your  Employment Category.**",[' State-gov', ' Self-emp-not-inc', ' Private', ' Federal-gov',
       ' Local-gov', ' Self-emp-inc', ' Without-pay'],index=None,placeholder=' ')
       Education=st.selectbox("**Choose your Qualification Level**",[' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',
       ' Some-college', ' Assoc-acdm', ' 7th-8th', ' Doctorate',
       ' Assoc-voc', ' Prof-school', ' 5th-6th', ' 10th', ' Preschool',
       ' 12th', ' 1st-4th'],index=None,placeholder=' ')
       Martial_status=st.selectbox("**Indicate your current marital status.**",[' Never-married', ' Married-civ-spouse', ' Divorced',
              ' Married-spouse-absent', ' Separated', ' Married-AF-spouse',
              ' Widowed'],index=None,placeholder=' ')
       occupation=st.selectbox("**Select your profession from the available options.**",[' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners',
       ' Prof-specialty', ' Other-service', ' Sales', ' Transport-moving',
       ' Farming-fishing', ' Machine-op-inspct', ' Tech-support',
       ' Craft-repair', ' Protective-serv', ' Armed-Forces',
       ' Priv-house-serv'],index=None,placeholder=' ')
       Race=st.radio("**Choose your Ethnic background.**",options=[' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo',
       ' Other'],index=None)

with col2:
       Relationship=st.selectbox("**Specify your role within the household.**",[' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried',
       ' Other-relative'],index=None,placeholder=' ')
       capital_gain=st.number_input("**Enter the monetary gains you have received.**" ,value=None,placeholder=' ')
       capital_loss=st.number_input(" **Enter any financial losses incurred**",value=None,placeholder=' ')
       HoursPerWeek=st.number_input("**Specify your average working hours per week**",value=None,placeholder=' ')
       country=st.selectbox("**Select the country you belong to**",[' United-States', ' Cuba', ' Jamaica', ' India', ' Mexico',
       ' Puerto-Rico', ' Honduras', ' England', ' Canada', ' Germany',
       ' Iran', ' Philippines', ' Poland', ' Columbia', ' Cambodia',
       ' Thailand', ' Ecuador', ' Laos', ' Taiwan', ' Haiti', ' Portugal',
       ' Dominican-Republic', ' El-Salvador', ' France', ' Guatemala',
       ' Italy', ' China', ' South', ' Japan', ' Yugoslavia', ' Peru',
       ' Outlying-US(Guam-USVI-etc)', ' Scotland', ' Trinadad&Tobago',
       ' Greece', ' Nicaragua', ' Vietnam', ' Hong', ' Ireland',
       ' Hungary', ' Holand-Netherlands'],index=None,placeholder=' ')
       sex=st.radio("**Select gender**",options=[' Male', ' Female'],index=None)


if st.button('Predict'):

    workclass=Employment.transform([workclass])[0]
    Education=Qualification.transform([[Education]])[0]
    Martial_status=ME.transform([[Martial_status]])[0]
    occupation=job.transform([[occupation]])[0]
    Relationship=relation.transform([[Relationship]])[0]
    Race=ethnic.transform([[Race]])[0]
    sex=gender.transform([[sex]])[0]
    country=native.transform([[country]])[0]
    


    data=np.array([[age,workclass,Education,Martial_status,occupation,
                Relationship,Race,sex,capital_gain,capital_loss,HoursPerWeek,country]])
   


    Prediction=Dte.predict(data)
    if Prediction == 1:
        st.markdown('<p class="success-msg">Your Income is greater than $50K!</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="warning-msg">Your Income is less than or equal to $50K.</p>', unsafe_allow_html=True)

       
