from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os
model_name = "MAQSoftware/roberta-sa-e1"
hf_token = os.environ.get("HUGGINGFACE_TOKEN")

def load_model(model_name, hf_token):
    """Load the sentiment analysis model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    return model, tokenizer

def predict_sentiment(input_text, model, tokenizer):
    """Predict the sentiment of the given text."""
    try:
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return pipe(input_text)[0]
    except Exception as e:
        print(f"Error during sentiment prediction: {e}")
        return None

# Load model and tokenizer
model, tokenizer = load_model(model_name, hf_token)

# Example text
input_text = "I love you"

# Way 1: Predict sentiment
sentiment_result = predict_sentiment(input_text, model, tokenizer)
if sentiment_result:
    print(f"Sentiment Result (Way 1): {sentiment_result}")

# Way 2: For more detailed output
try:
    tokenized_input = tokenizer(input_text, return_tensors="pt")
    output = model(**tokenized_input)
    predicted_class = output.logits.argmax(-1).item() 
    print(f"Predicted Class (Way 2): {predicted_class}")
except Exception as e:
    print(f"Error during detailed prediction: {e}")
