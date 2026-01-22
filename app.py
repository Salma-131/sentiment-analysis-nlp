import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(page_title="Sentiment Analysis with BERT", layout="centered")

st.title("🧠 Sentiment Analysis using BERT")
st.write("Enter a sentence and the model will predict its sentiment.")

# ---------------------------
# Load model (cached)
# ---------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert_sentiment_model")
    model = BertForSequenceClassification.from_pretrained("bert_sentiment_model")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ---------------------------
# Prediction function
# ---------------------------
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Positive 😊" if prediction == 1 else "Negative 😞"

# ---------------------------
# UI
# ---------------------------
text_input = st.text_area("✍️ Enter your text here:")

if st.button("Predict Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment = predict_sentiment(text_input)
        st.success(f"**Predicted Sentiment:** {sentiment}")
