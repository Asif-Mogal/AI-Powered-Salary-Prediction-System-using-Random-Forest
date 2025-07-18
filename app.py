import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from streamlit_extras.stylable_container import stylable_container

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Salary Data.csv")
    df = df.dropna()
    
    # Remove outliers
    Q1 = df['Salary'].quantile(0.25)
    Q3 = df['Salary'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Salary'] >= lower_bound) & (df['Salary'] <= upper_bound)]
    
    # Encode categoricals
    unique_jobs = df['Job Title'].unique()
    job_to_code = {job: i for i, job in enumerate(unique_jobs)}
    code_to_job = {i: job for job, i in job_to_code.items()}

    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Education Level'] = df['Education Level'].map({
        'High School': 0,
        "Bachelor's": 1,
        "Master's": 2,
        'PhD': 3
    })
    df['Job Title'] = df['Job Title'].map(job_to_code)
    
    return df, job_to_code

df, job_to_code = load_data()

# Train model
X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
score = r2_score(y_test, model.predict(X_test))

# UI Config
st.set_page_config(
    page_title="💰 SALARY PREDICTOR", 
    layout="wide",
    page_icon="💸"
)

# Custom CSS
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
        .main {{
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: white;
            font-family: 'Space Grotesk', sans-serif;
        }}
        
        .stApp {{
            background-color:#EEECE9;
            background-attachment: fixed;
        }}
        
        h1 {{
            font-weight: 800;
            letter-spacing: -1px;
            text-align: center;
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }}
        
        .title-accent {{
            height: 4px;
            width: 120px;
            margin: 0 auto 2rem;
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            border-radius: 2px;
        }}
        
        .prediction-card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }}
        
        .stButton>button {{
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            width: 100%;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 210, 255, 0.3);
            
        }}
        
        .stSlider>div>div>div>div {{
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%) !important;
        }}
        
        .stSelectbox>div>div>div {{
            background-color: rgba(255, 255, 255, 0.1) !important;
        }}
        
        .metric-value {{
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
        }}
        
        .metric-label {{
            text-align: center;
            font-size: 1rem;
            opacity: 0.8;
            margin-top: -0.5rem;
        }}
        
        .error-message {{
            background: rgba(255, 59, 48, 0.2);
            padding: 1rem;
            border-radius: 12px;
            border-left: 4px solid #ff3b30;
            margin: 1rem 0;
        }}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style="padding: 2rem 0 1rem;">
        <h1>SALARY PREDICTOR</h1>
        <div class="title-accent"></div>
        <p style="text-align: center; font-size: 1.1rem; opacity: 0.9; max-width: 700px; margin: 0 auto;">
            Use the power of machine learning to predict your earning potential based on 
            your professional profile and qualifications.
        </p>
    </div>
""", unsafe_allow_html=True)

# Main container
with st.container():
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        with stylable_container(
            key="input_container",
            css_styles="""
                {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 16px;
                    padding: 2rem;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
            """
        ):
            st.markdown("### 🧑‍💼 PROFILE INFORMATION")
            age = st.slider("**AGE**", 18, 65, 30, help="Select your current age")
            gender = st.selectbox("**GENDER**", ['Male', 'Female'])
            edu = st.selectbox("**EDUCATION LEVEL**", ['High School', "Bachelor's", "Master's", 'PhD'])
            job_title = st.selectbox("**JOB TITLE**", sorted(job_to_code.keys()))
            years_exp = st.slider("**YEARS OF EXPERIENCE**", 0, 40, 5, help="Your total professional experience")
            
            show = True
            if years_exp > age:
                with stylable_container(
                    key="error_container",
                    css_styles="""
                        {
                            background: rgba(255, 59, 48, 0.2);
                            padding: 1rem;
                            border-radius: 12px;
                            border-left: 4px solid #ff3b30;
                            margin: 1rem 0;
                        }
                    """
                ):
                    st.error("⚠️ Years of experience cannot exceed age!")
                show = False
    
    with col2:
        with stylable_container(
            key="result_container",
            css_styles="""
                {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 16px;
                    padding: 2rem;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    height: 100%;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                }
            """
        ):
            st.markdown("### 💰 SALARY PREDICTION")
            st.markdown("###")
            
            if st.button("**PREDICT MY SALARY**", type="primary", use_container_width=True) and show:
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Gender': [0 if gender == 'Male' else 1],
                    'Education Level': [{
                        'High School': 0,
                        "Bachelor's": 1,
                        "Master's": 2,
                        'PhD': 3
                    }[edu]],
                    'Job Title': [job_to_code[job_title]],
                    'Years of Experience': [years_exp]
                })
                prediction = model.predict(input_data)[0]
                
                with stylable_container(
                    key="prediction_result",
                    css_styles="""
                        {
                            background: linear-gradient(135deg, rgba(0, 210, 255, 0.1), rgba(58, 123, 213, 0.1));
                            border-radius: 16px;
                            padding: 2rem;
                            margin: 2rem 0;
                            border: 1px solid rgba(0, 210, 255, 0.3);
                            text-align: center;
                        }
                    """
                ):
                    st.markdown(f"""
                        <div style="margin: 1rem 0;">
                            <div class="metric-value">₹{prediction:,.0f}</div>
                            <div class="metric-label">ANNUAL SALARY</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                        <div style="font-size: 0.9rem; opacity: 0.8; text-align: center; margin-top: 1rem;">
                            Prediction powered by Random Forest Regressor (R² score: {:.2f})
                        </div>
                    """.format(score), unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="text-align: center; padding: 4rem 0; opacity: 0.6;">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20Z" fill="currentColor"/>
                            <path d="M12 7C11.45 7 11 7.45 11 8V12C11 12.55 11.45 13 12 13C12.55 13 13 12.55 13 12V8C13 7.45 12.55 7 12 7Z" fill="currentColor"/>
                            <path d="M11 16C11 15.45 11.45 15 12 15C12.55 15 13 15.45 13 16C13 16.55 12.55 17 12 17C11.45 17 11 16.55 11 16Z" fill="currentColor"/>
                        </svg>
                        <p style="margin-top: 1rem;">Fill in your details and click "Predict My Salary"</p>
                    </div>
                """, unsafe_allow_html=True)

# Model info section
with st.expander("**🧠 ABOUT THE AI MODEL**"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with stylable_container(
            key="metric_1",
            css_styles="""
                {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    padding: 1.5rem;
                    text-align: center;
                }
            """
        ):
            st.metric("Model Used", "Random Forest")
    
    with col2:
        with stylable_container(
            key="metric_2",
            css_styles="""
                {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    padding: 1.5rem;
                    text-align: center;
                }
            """
        ):
            st.metric("Accuracy (R² Score)", f"{score:.2f}")
    
    with col3:
        with stylable_container(
            key="metric_3",
            css_styles="""
                {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    padding: 1.5rem;
                    text-align: center;
                }
            """
        ):
            st.metric("Training Samples", f"{len(X_train):,}")
    
    st.markdown("""
        ### How It Works
        
        This salary predictor uses a **Random Forest Regressor** machine learning model trained on historical salary data. 
        The model considers multiple factors including:
        
        - Age
        - Gender
        - Education Level
        - Job Title
        - Years of Experience
        
        The model achieves an R² score of **{:.2f}**, meaning it explains {:.0f}% of the variance in salary data.
    """.format(score, score*100))