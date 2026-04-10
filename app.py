import streamlit as st
import pandas as pd
import ast
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Skill Gap Analyzer + Weighted ML", layout="wide")
st.title("Skill Gap Analyzer + ML Predictor")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("./dataset/data_science_job_posts_2025.csv")
    df['skills'] = df['skills'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    df['skills'] = df['skills'].apply(lambda skills: [s.lower() for s in skills])
    df = df[df['skills'].map(len) > 0]
    df = df.dropna(subset=['job_title', 'seniority_level'])
    return df

df = load_data()

# Filter Roles for ML
df_model = df[df['job_title'].str.contains("Data Scientist|Machine Learning Engineer", case=False, na=False)].copy()

def label_role(x):
    x = x.lower()
    if "data scientist" in x: return "Data Scientist"
    elif "machine learning engineer" in x: return "Machine Learning Engineer"
    else: return "Other"

df_model['role_label'] = df_model['job_title'].apply(label_role)
df_model = df_model[df_model['role_label'] != "Other"]

# ML Training
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df_model['skills'])
y = df_model['role_label']

model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=12,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X, y)

# Weighted Scoring Function
def get_weighted_match_score(user_skills, target_role):
    subset = df[df['job_title'].str.contains(target_role, case=False, na=False)]
    
    all_role_skills = [skill for sublist in subset['skills'] for skill in sublist]
    role_skill_counts = Counter(all_role_skills)
    
    if not role_skill_counts:
        return 0, []

    top_20_role_skills = dict(role_skill_counts.most_common(20))
    total_possible_weight = sum(top_20_role_skills.values())
    
    user_match_weight = sum([top_20_role_skills[s] for s in user_skills if s in top_20_role_skills])
    
    match_score = user_match_weight / total_possible_weight if total_possible_weight > 0 else 0
    return match_score, list(top_20_role_skills.keys())

# Sidebar for User Input
all_skills = [skill for sublist in df['skills'] for skill in sublist]
market_skills = [skill for skill, _ in Counter(all_skills).most_common(60)]

st.sidebar.header("Your Profile")
user_skills = st.sidebar.multiselect("Select your skills:", options=sorted(market_skills))
target_role = st.sidebar.selectbox("Select Target Role:", ["Data Scientist", "Machine Learning Engineer"])

# Main Logic
if user_skills:
    match_score, top_role_skills = get_weighted_match_score(user_skills, target_role)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Weighted Match Score")
        st.progress(match_score)
        st.write(f"**{round(match_score * 100, 1)}% match for {target_role}**")

    with col2:
        missing_skills = [s for s in top_role_skills if s not in user_skills][:7]
        st.subheader("Recommended Skills")
        if missing_skills:
            for s in missing_skills:
                st.write(f"- {s}")
        else:
            st.success("You're well matched!")

    st.markdown("---")

    # ML Prediction
    st.subheader("ML Role Prediction")
    user_vector = mlb.transform([user_skills])
    prediction = model.predict(user_vector)[0]
    probabilities = model.predict_proba(user_vector)[0]
    prob_dict = dict(zip(model.classes_, probabilities))

    c1, c2 = st.columns([2, 3])
    with c1:
        st.metric(label="Predicted Role", value=prediction)
    with c2:
        for role, prob in prob_dict.items():
            st.write(f"**{role}**: {round(prob*100, 1)}%")
            st.progress(prob)
else:
    st.info("Please select your skills from the sidebar to start analysis.")

# Footer
st.markdown("---")
st.markdown("**Built for M2M Tech DataTalent Program Capstone Project 3** by *Wonki Hwang*")
st.markdown("**Data source:** [Kaggle](https://www.kaggle.com/datasets/elahehgolrokh/data-science-job-postings-with-salaries-2025)")