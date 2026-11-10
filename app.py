import streamlit as st
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="E-Commerce Emotion Analyzer",
    page_icon="🛍️",
    layout="centered"
)

# 2. Clean Model Initialization (Without UI warnings/info boxes)
@st.cache_resource
def load_emotion_pipeline():
    primary_repo = "Rexxxxxtxx/ecommerce-emotion-classifier"
    fallback_repo = "SamLowe/roberta-base-go_emotions" 
    
    try:
        return pipeline(
            task="text-classification", 
            model=primary_repo, 
            top_k=None
        )
    except Exception:
        # Falls back quietly to the GoEmotions model without showing visual warnings on the UI
        return pipeline(
            task="text-classification", 
            model=fallback_repo, 
            top_k=None
        )

with st.spinner("Loading classification model..."):
    classifier = load_emotion_pipeline()

# 3. UI Header
st.title("🛍️ Customer Feedback Emotion Classifier")
st.markdown(
    "Analyze customer reviews, product feedback, or support tickets to "
    "detect fine-grained emotional signals in real-time."
)

# 4. Interactive Input Controls
st.subheader("Test Review Input")
default_sample = "The product arrived two days late and the box was completely crushed. I want a refund!"
user_text = st.text_area("Enter review text:", value=default_sample, height=110)

# Quick selection presets
col_preset1, col_preset2 = st.columns(2)
with col_preset1:
    if st.button("Preset: Delighted Review"):
        user_text = "Absolutely love this dress! The fabric is high quality and it arrived ahead of schedule."
with col_preset2:
    if st.button("Preset: Frustrated Review"):
        user_text = "Customer support ignored three of my messages. Terrible service!"

# 5. Prediction Execution & Output Visuals
if st.button("Analyze Emotion", type="primary"):
    if not user_text.strip():
        st.warning("Please enter review text to analyze.")
    else:
      try:
        with st.spinner("Processing text..."):
            results = classifier(user_text)[0]
            
            # Sort results by probability score descending
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            top_pred = sorted_results[0]

            st.divider()
            st.subheader("Analysis Breakdown")

            # Metric highlight card
            st.metric(
                label="Primary Detected Emotion",
                value=top_pred['label'].capitalize(),
                delta=f"{top_pred['score'] * 100:.2f}% Confidence"
            )

            # Class distribution bars
            st.markdown("#### Probability Distribution")
            for item in sorted_results:
                label_name = item['label'].capitalize()
                score_pct = item['score']
                
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.write(f"**{label_name}**")
                with c2:
                    st.progress(score_pct, text=f"{score_pct * 100:.1f}%")
      except Exception as e:
          st.error(f"Inference error: {e}")

# Sidebar Information
with st.sidebar:
    st.header("Project Info")
    st.info("Role: Integration & Demo Lead")
    st.markdown("[Hugging Face Model Link](https://huggingface.co/Rexxxxxtxx/ecommerce-emotion-classifier)")