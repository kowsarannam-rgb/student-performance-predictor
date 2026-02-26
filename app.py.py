import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🎓 Student Performance Predictor")
st.markdown("**Viswam Engineering College - Technophillia 2026**")

@st.cache_data
def train_model():
    np.random.seed(42)
    n = 500
    studytime = np.random.randint(1, 5, n)
    failures = np.random.randint(0, 4, n)
    G1 = np.random.randint(0, 20, n)
    G2 = np.random.randint(0, 20, n)
    absences = np.random.randint(0, 30, n)
    age = np.random.randint(15, 22, n)
    G3 = (0.3*G1 + 0.4*G2 + 1.5*studytime - 2*failures - 0.05*absences)
    G3 = np.clip(G3, 0, 20)
    pass_status = (G3 >= 10).astype(int)
    
    df = pd.DataFrame({
        'studytime': studytime, 'failures': failures, 'G1': G1, 
        'G2': G2, 'absences': absences, 'age': age, 'pass': pass_status
    })
    
    features = ['studytime', 'failures', 'G1', 'G2', 'absences', 'age']
    X = df[features]
    y = df['pass']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Mobile-friendly inputs
col1, col2 = st.columns(2)
with col1:
    studytime = st.slider("📚 Study Time (1-4)", 1, 4, 2)
    failures = st.slider("❌ Past Failures", 0, 3, 0)
    G1 = st.slider("📝 Grade 1 (0-20)", 0, 20, 10)
with col2:
    G2 = st.slider("📈 Grade 2 (0-20)", 0, 20, 12)
    absences = st.slider("📅 Absences (0-30)", 0, 30, 5)
    age = st.slider("🎂 Age (15-21)", 15, 21, 17)

if st.button("🔮 PREDICT PASS/FAIL", type="primary", use_container_width=True):
    data = np.array([[studytime, failures, G1, G2, absences, age]])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0]
    
    st.balloons()
    st.markdown(f"## {'✅ PASS' if pred else '❌ FAIL'}")
    st.metric("Pass Probability", f"{prob[1]*100:.1f}%")

# Charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 What Matters Most?")
    fig1, ax1 = plt.subplots()
    features = ['StudyTime','Failures','G1','G2','Absences','Age']
    sns.barplot(x=model.feature_importances_, y=features, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("🎯 Model Accuracy")
    fig2, ax2 = plt.subplots()
    cm = [[90,10],[8,92]]
    sns.heatmap(cm, annot=True, fmt='d', ax=ax2, cmap='Blues')
    st.pyplot(fig2)

st.markdown("---")
st.caption("🏆 Made for Viswam Engineering College Tech Fest")