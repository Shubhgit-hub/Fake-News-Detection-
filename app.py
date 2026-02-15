import streamlit as st
import pickle
import re
import string

#  Text Cleaning 
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

# Load trained model & vectorizer

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

#  Streamlit Page Setup
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🕵️",
    layout="centered"
)

# 4. App UI
st.title("🕵️ Fake News Detection System")
st.markdown("Paste the news here!")

# 5. User Input
news_text = st.text_area(
    "📝 News Article Text:",
    height=200,
    placeholder="Paste it here"
)

# 6. Prediction Logic
if st.button("🔍 Check News"):
    if news_text.strip() == "":
        st.warning("Please enter news text news.")
    else:
        # Step A: Clean the text
        cleaned_text = wordopt(news_text)
        
        # Step B: Vectorize
        input_vector = vectorizer.transform([cleaned_text])
        
        # Step C: Predict
        prediction = model.predict(input_vector)
        
        # Step D: Display Result
        st.divider()
        if prediction[0] == 1:
            st.error("🚨 **FAKE NEWS DETECTED!**")
            st.caption("Fake.")
        else:
            st.success("✅ **REAL NEWS**")
            st.caption("True.")

