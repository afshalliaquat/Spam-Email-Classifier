import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64

st.set_page_config(
    page_title="Spam Email Detector",          # Tab name/title
    page_icon="icon_AL.png"                             # Favicon emoji or path to icon image file
)

st.title("Spam Email Classifier")
st.write("Check your email's spam score")

model = joblib.load("spam_model_LR.pkl")
bow = joblib.load("bow.pkl")  



def set_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

set_bg_from_local("bg.jpg")
def why_this_result(model, vectorizer, text):
    text_vectorized = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    nz_indices = text_vectorized.nonzero()[1]
    contributions = [(feature_names[i], coef[i] * text_vectorized[0, i]) for i in nz_indices]
    contributions.sort(key=lambda x: x[1], reverse=True)
    score = text_vectorized.dot(coef) + intercept
    prob = 1 / (1 + np.exp(-score))
    return prob[0], contributions

if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "prob" not in st.session_state:
    st.session_state.prob = None
if "contributions" not in st.session_state:
    st.session_state.contributions = None
if "prediction_label" not in st.session_state:
    st.session_state.prediction_label = None

with st.container():
    st.subheader("Upload Text File")
    uploaded_file = st.file_uploader("Choose a file", type=["txt"])
    email_content = ""
    if uploaded_file is not None:
        email_content = uploaded_file.read().decode("utf-8")
    text = st.text_area("Email Content", value=email_content, height=300)
    
    if st.button("Predict"):
        if text.strip():
            prob, contributions = why_this_result(model, bow, text)
            prediction_label = 'SPAM' if prob > 0.5 else 'HAM'
            st.session_state.predicted = True
            st.session_state.prob = prob
            st.session_state.contributions = contributions
            st.session_state.prediction_label = prediction_label
        else:
            st.warning("Please enter text or upload email content")

    if st.session_state.predicted:
        st.subheader('Prediction')
        st.write(st.session_state.prediction_label)
        
        if st.button("Explain Prediction"):
            st.subheader('Confidence')
            st.write(f"Spam: {st.session_state.prob*100:.2f}%")
            st.write(f"Ham: {(1 - st.session_state.prob)*100:.2f}%")

            st.subheader("Top words influencing prediction")
            for word, impact in st.session_state.contributions[:10]:
                if impact > 0:
                    st.write(f"🔴 {word} → pushes toward spam ({impact:.2f})")
                else:
                    st.write(f"🟢 {word} → pushes toward ham ({impact:.2f})")


st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 10px;
        font-size: 14px;
        font-weight: plain;
        color: #222222;  
        background: rgba(255, 255, 255, 0.8); 
        padding: 4px 8px;
        border-radius: 4px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
        opacity: 1;  
        z-index: 1000; 
    }
    </style>
    <div class="footer">
        © developed by Afshal Liaquat
    </div>
    """,
    unsafe_allow_html=True
)
