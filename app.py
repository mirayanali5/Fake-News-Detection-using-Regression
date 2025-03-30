import streamlit as st
import pickle
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import base64

# Function to convert a binary file (e.g., an image) to a Base64 string
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Local image path
img_path = r"C:\Users\miray\OneDrive\Desktop\Jupyter Practice\image.jpg"

# Convert the image to a Base64 string
img_base64 = get_base64_of_bin_file(img_path)

# Inject custom CSS to set the background image using the Base64 string
page_bg_img = f"""
<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Stemming function
def stemming(content):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Load the model and vectorizer
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load dataset
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('train_cleaned.csv', engine='python', on_bad_lines='skip')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found.")
        return None

# Enhanced prediction function with confidence score
def predict_news(text, model, vectorizer):
    if model is None or vectorizer is None:
        st.error("Model or vectorizer not loaded.")
        return None, None
    
    stemmed_text = stemming(text)
    vectorized_text = vectorizer.transform([stemmed_text])
    prediction = model.predict(vectorized_text)
    
    prediction_proba = model.predict_proba(vectorized_text)[0]
    confidence = max(prediction_proba) * 100
    
    return "Fake" if prediction[0] == 1 else "Real", confidence

# Dataset statistics and visualization
def show_dataset_stats(df):
    st.subheader("📊 Dataset Statistics")
    
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    
    total_articles = len(df)
    real_news = int(len(df[df['label'] == 0]))
    fake_news = int(len(df[df['label'] == 1]))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Total Articles", f"{total_articles:,}")
    with col2:
        st.metric("✅ Real News", f"{real_news:,}")
    with col3:
        st.metric("❌ Fake News", f"{fake_news:,}")
    
    st.subheader("📈 Distribution of Real vs Fake News")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['#00CC96', '#EF553B']
    ax1.bar(['Real News', 'Fake News'], [real_news, fake_news], color=colors)
    ax1.set_ylabel('Number of Articles')
    ax1.set_title('Distribution (Count)')
    
    if real_news + fake_news > 0:
        sizes = [real_news, fake_news]
        labels = ['Real News', 'Fake News']
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribution (Percentage)')
    else:
        ax2.text(0.5, 0.5, 'No valid data available', horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("📋 Detailed Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Article Distribution:")
        if total_articles > 0:
            dist_df = pd.DataFrame({
                'Category': ['Real News', 'Fake News'],
                'Count': [real_news, fake_news],
                'Percentage': [
                    f"{(real_news / total_articles * 100):.2f}%",
                    f"{(fake_news / total_articles * 100):.2f}%"
                ]
            })
            st.dataframe(dist_df)
        else:
            st.write("No valid data available")
    
    with col2:
        st.write("Key Metrics:")
        text_lengths = df['text'].str.len().dropna()
        if len(text_lengths) > 0:
            st.write(f"• Average article length: {text_lengths.mean():.0f} characters")
            st.write(f"• Median article length: {text_lengths.median():.0f} characters")
        else:
            st.write("No valid text data available")

# Streamlit UI
def main():
    st.title("🔍 Fake News Detection System")
    
    # Create tabs with increased size
    tab1, tab2 = st.tabs(["🎯 **Prediction**", "📊 **Dataset Analysis**"])
    
    model, vectorizer = load_model()
    
    with tab1:
        st.write("Enter a news article to check if it's real or fake")
        
        if model is not None and vectorizer is not None:
            news_text = st.text_area("Enter the news text here:", height=200)
            
            if st.button("🔍 Analyze"):
                if news_text.strip() == "":
                    st.warning("⚠️ Please enter some text to analyze")
                else:
                    with st.spinner("🔄 Analyzing..."):
                        result, confidence = predict_news(news_text, model, vectorizer)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if result == "Real":
                                st.success(f"✅ This news appears to be {result}")
                            else:
                                st.error(f"❌ This news appears to be {result}")
                        
                        with col2:
                            st.info(f"🎯 Confidence: {confidence:.2f}%")
                        
                        st.progress(confidence/100)
                        
                        with st.expander("🔍 See processed text"):
                            st.write("Processed text after stemming:")
                            st.code(stemming(news_text))
    
    with tab2:
        df = load_dataset()
        if df is not None:
            show_dataset_stats(df)
            
            st.subheader("🔍 Dataset Preview")
            num_rows = st.slider("Number of rows to display", 10, 100, 50)
            
            columns = st.multiselect("Select columns to display", df.columns.tolist(), default=df.columns.tolist())
            
            st.dataframe(df[columns].head(num_rows))
            
            with st.expander("ℹ️ Dataset Information"):
                st.write("Dataset Shape:", df.shape)
                st.write("\nMissing Values:")
                st.write(df.isnull().sum())
                st.write("\nBasic Statistics:")
                st.write(df.describe())

if __name__ == "__main__":
    main()
