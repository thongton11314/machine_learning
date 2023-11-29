# https://mlflow.org/docs/latest/models.html
import mlflow
import os
import logging
import json
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define variables
mlflow_model_path = "./sentiment_analysis_mlflow"
model_name_huggingface = "MAQSoftware/roberta-sa-e1"
model_name_local = "roberta-sa-e1"

# https://huggingface.co/docs/hub/security-tokens
access_token = os.environ.get("HUGGINGFACE_TOKEN")

# Paths for the model and tokenizer
model_path = f"./model_{model_name_local}"
tokenizer_path = f"./model_tokenizer_{model_name_local}"

# Check if the model and tokenizer are already downloaded
if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
    logging.info("Downloading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_huggingface, token=access_token)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_huggingface, use_auth_token=access_token)
    except Exception as e:
        logging.error(f"Error in downloading the model/tokenizer: {e}")
        raise

    # Save the model and tokenizer locally
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)
else:
    logging.info("Model and tokenizer already exist locally.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Wrapper class for the Hugging Face model
class HuggingFaceModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, tokenizer):
        """Wrapper class for the Hugging Face model"""
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, context, model_input):
        """Generates predictions for the input using the Hugging Face model"""
        # Pipeline for text classification
        pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        if not model_input:
            raise ValueError('Input is empty')
        print(f"*Your input*: {model_input}")
        logging.info(f'Input: {model_input}')
        return json.dumps(pipe(str(model_input["input_string"]))[0])

# Run by MLflow model
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="sentiment_analysis_mlflow",
        python_model=HuggingFaceModelWrapper(model, tokenizer),
        artifacts={"model": model_path}
    )
    print("Logged model in run ID:", run.info.run_id)

# Use the printed run ID to load the model
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/sentiment_analysis_mlflow")

# Test prediction
try:
    output = loaded_model.predict({'input_string': np.array('I love you!', dtype='<U11')})
    print(output)
except Exception as e:
    logging.error(f"Error in prediction: {e}")
