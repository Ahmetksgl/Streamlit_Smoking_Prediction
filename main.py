import streamlit as st
import plotly.express as px
import joblib
import pandas as pd


st.set_page_config(layout='wide')

@st.cache_data
def get_data():
    df = pd.read_csv('train.csv')
    return df

def get_model():
    model = joblib.load("model.joblib")
    return model

st.header(':red[Smoking Prediction]')

tab_home, tab_vis, tab_model = st.tabs(['Home', 'Graphics', 'Model'])

# TAB HOME
column_sig, column_sig2 = tab_home.columns(2)

column_sig.subheader('Research Data☠️')

column_sig.markdown("According to the report published by the World Health Organization in 2032, 8.7 million deaths worldwide are attributed to tobacco use (WHO Report, 2023). In Turkey, 83,100 people die each year due to cigarette use (Yeşilay, 2023). Tobacco use is a major factor in diseases such as COPD, tuberculosis, and lung cancer, which constitute six of the leading causes of death worldwide (Çalışkan and Metintaş, 2018). Turkey, among the top 10 countries with the highest smoking rates, is estimated to have approximately 16 million smokers (Çalışkan and Metintaş, 2018). While the tobacco use rate is 42.1% in males, it is 30.7% in females. Over the years, an increase in the tobacco use rate among women has been observed (BBC, 2022). Turkey ranks first among OECD countries in the proportion of people over 15 years old who smoke compared to the entire population (BBC, 2022).")

column_sig.subheader('Problem Definition')

column_sig.markdown("The project aims to enable insurance companies to make accurate assessments of their customers. Policy prices for individuals who smoke are adjusted to higher amounts. Ultimately, individuals who smoke may provide false statements. Approving a policy grants access to the company's personal information. A person who has given false information may be excluded from health insurance coverage when experiencing a smoking-related illness in the future.")


column_sig2.subheader('Project Objective')

column_sig2.markdown("The project aims to enable insurance companies to accurately identify customers who smoke and prevent false statements. Machine learning methods are used to detect smoking status from biological data, blood test results, etc. The goal is to provide insurance companies with faster and more accurate access to information.")

column_sig2.subheader('Dataset')
column_sig2.markdown("The competition dataset titled 'Binary Prediction of Smoking Status Using Bio-Signals', started in 2023 on the Kaggle platform, has been selected as the dataset for this project. The dataset contains a total of 23 variables and 265,427 records. The problem represents binary classification. The 'Smoking' variable is the target variable; smoking is represented by 1, and non-smoking is represented by 0. Biological data, also known as blood values, includes medical terms.")


df = get_data()
df_ = df.head(50)
column_sig2.dataframe(df_)

column_sig.subheader('Features👀')

column_sig.markdown('''
- :red[id:] Cardinal variable, dropped from modeling.
- :red[age:] Age in 5-year intervals.
- :red[height (cm):] Individual's height in centimeters.
- :red[weight (kg):] Individual's weight in kilograms.
- :red[waist (cm):] Waist circumference length in centimeters.
- :red[eyesight (left/right):] Eyesight measurement for the left/right eye.
- :red[hearing (left/right):] Hearing ability for the left/right ear.
- :red[systolic:] Systolic blood pressure measurement.
- :red[relaxation:] Diastolic blood pressure measurement.
- :red[fasting blood sugar:] Fasting blood sugar measurement.
- :red[Cholesterol (total):] Total cholesterol level.
- :red[triglyceride:] Triglyceride level.
- :red[HDL (High-Density Lipoprotein):] HDL cholesterol level (good cholesterol).
- :red[LDL (Low-Density Lipoprotein):] LDL cholesterol level (bad cholesterol).
- :red[hemoglobin:] Hemoglobin level.

''')

column_sig2.markdown('''
- :red[Urine protein:] Urine protein level.
- :red[serum creatinine:] Serum creatinine level.
- :red[AST (Aspartate Aminotransferase):] AST enzyme level.
- :red[ALT (Alanine Aminotransferase):] ALT enzyme level.
- :red[Gtp (γ-Glutamyltranspeptidase):] Gtp level.
- :red[dental caries:] Dental caries status.
- :red[smoking:] Smoking status.''')

# GRAPHICS

vis_1, vis_2 = tab_vis.columns(2)
dff = df.sample(500)

# Graphic 1
vis_1.subheader(':red[Target]')
fig = px.pie(dff, names='smoking', title='Smoking Distribution', color='smoking', color_discrete_map={0: 'lightcoral', 1: 'lightskyblue'})
vis_1.plotly_chart(fig)


# Graphic 2
color_scale = px.colors.qualitative.Set1
fig_2 = px.histogram(dff, x='age', color='smoking', nbins=20, marginal='rug', opacity=0.7, barmode='overlay',
                   title='Histogram of Smokers by Age', color_discrete_sequence=color_scale)
fig_2.update_layout(xaxis_title='Age', yaxis_title='Frequency')
fig_2.update_traces(overwrite=True, showlegend=False)
vis_2.plotly_chart(fig_2)

# Graphic 3
fig_3 = px.histogram(dff, x='triglyceride', color='smoking', nbins=20, opacity=0.7, barmode='overlay',
                   title='Histogram of Triglyceride by Smoking Status', color_discrete_sequence=color_scale)
fig_3.update_layout(xaxis_title='Triglyceride', yaxis_title='Frequency')
fig_3.update_traces(overwrite=True, showlegend=False)
vis_1.plotly_chart(fig_3)

# Graphic 4
scatter_fig = px.scatter(
    dff,
    x="age",
    y="triglyceride",
    size="hemoglobin",
    color="smoking",
    hover_name=df.index,
    log_x=True,
    size_max=60,
    title='Scatter Plot: Smoking Status, age, triglyceride, and hemoglobin',
)

vis_2.plotly_chart(scatter_fig, use_container_width=True)


# MODEL

model = get_model()

# Features
age = 30
height = 165.0
weight = 67.0
waist = 83.0
eyesightright = 1.0
eyesightleft = 1.0
dental_caries = 0
ALT = 25.0
triglyceride = 128
hemoglobin = 15.0
HDL = 54
Gtp = 30
relaxation = 76
serum_creatinine = 0.9

user_choice = tab_model.radio("Information you need to enter", ["Physical Values", "Medical Values"])

if user_choice == "Physical Values":

    age = tab_model.number_input("Enter Your Age", min_value=18, max_value=80, step=1, value=30,
                          help="Age is represented in years.")
    heightcm = tab_model.number_input("Enter Your Height", min_value=135.0, max_value=190.0, step=0.1, value=165.0,
                             help="Height is in centimeters.")
    weightkg = tab_model.number_input("Enter Your Weight", min_value=30.0, max_value=120.0, step=0.1, value=67.0,
                             help="Weight is in kilograms.")
    waistcm = tab_model.number_input("Enter Your Waist", min_value=51.0, max_value=104.0, step=0.1, value=83.0,
                            help="Waist circumference length is in centimeters.")
    eyesightright = tab_model.number_input("Enter Your Eyesight Right", min_value=0.1, max_value=1.5, step=0.1, value=1.0, help="Eyesight measurement for the right eye, where 0.1 represents poor eyesight and 1.0 represents normal eyesight.")
    eyesightleft = tab_model.number_input("Enter Your Eyesight Left", min_value=0.1, max_value=1.5, step=0.1, value=1.0, help="Eyesight measurement for the left eye, where 0.1 represents poor eyesight and 1.0 represents normal eyesight.")
    dental_caries = tab_model.number_input("Enter Your Dental Caries Status", min_value=0, max_value=1, step=1, value=0, help="Dental caries status (0: No, 1: Yes, where 0 represents no dental caries and 1 represents presence of dental caries).")

elif user_choice == "Medical Values":

    ALT = tab_model.number_input("Enter Your ALT Value", min_value=1.0, max_value=81.0, step=0.1, value=25.0, help="ALT (Alanine Aminotransferase) is an enzyme found mainly in the liver.")
    triglyceride = tab_model.number_input("Enter Your Triglyceride Value", min_value=8, max_value=330, step=1,
                                   value=128, help="Triglyceride is a type of fat found in your blood.")
    hemoglobin = tab_model.number_input("Enter Your Hemoglobin Value", min_value=4.9, max_value=17.5, step=0.1,
                                 value=15.0, help="Hemoglobin is a protein in red blood cells that carries oxygen.")
    HDL = tab_model.number_input("Enter Your HDL Value", min_value=9, max_value=90, step=1, value=54,
                          help="HDL (High-Density Lipoprotein) is often referred to as 'good cholesterol.'")
    Gtp = tab_model.number_input("Enter Your Gtp Value", min_value=2, max_value=150, step=1, value=30,
                          help="Gtp (γ-Glutamyltranspeptidase) is an enzyme found in the liver.")
    relaxation = tab_model.number_input("Enter Your relaxation Value", min_value=40, max_value=100, step=1, value=76, help="Relaxation is a measure of diastolic blood pressure.")
    serum_creatinine = tab_model.number_input("Enter Your serum creatinine Value",min_value=0.1, max_value=1.3, step=0.1, value=0.9,help="Serum creatinine is a measure of kidney function. Enter a value between 0.1 and 1.3.")

user_input = pd.DataFrame({'age': age, 'heightcm': heightcm, 'weightkg': weightkg, 'waistcm': waistcm,
                            'eyesightright': eyesightright, 'eyesightleft': eyesightleft,
                            'dental_caries': dental_caries, 'ALT': ALT, 'triglyceride': triglyceride,
                            'hemoglobin': hemoglobin, 'HDL': HDL, 'Gtp': Gtp, 'relaxation': relaxation, 'serum_creatinine': serum_creatinine}, index=[0])

if tab_model.button("Prediction!"):
    prediction = model.predict(user_input)

    # Tahmin sonucunu etiketlere dönüştür
    prediction_label = "Smoking" if prediction[0] == 1 else "Non-Smoking"

    tab_model.success(f"Estimated smoking status: {prediction_label}")


