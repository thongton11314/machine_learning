# Sentiment Analysis Model with Hugging Face and MLflow Integration

This project integrates a sentiment analysis model from Hugging Face with MLflow, a platform for the machine learning lifecycle, including experimentation, reproducibility, and deployment.

## Requirements

- Python 3.10
- Hugging Face Transformers
- MLflow
- Logging

## Environment Setup

1. **Set Up Environment Variables:**
   - Ensure you have set the `HUGGINGFACE_TOKEN` in your environment variables for secure access to Hugging Face models.

2. **Install Required Libraries:**
   - Use `pip install transformers mlflow logging` to install the necessary libraries.

## Usage

1. **Model Initialization:**
   - The script initializes the model and tokenizer from Hugging Face, specifically using the model `MAQSoftware/roberta-sa-e1`. If the model and tokenizer are not downloaded, they will be fetched and saved locally.

2. **MLflow Integration:**
   - The model is then wrapped using a custom `HuggingFaceModelWrapper` class for compatibility with MLflow.

3. **Saving and Loading the Model:**
   - The model is saved and loaded using MLflow's `pyfunc` format.

4. **Test Prediction:**
   - A test prediction is made using the loaded model to ensure functionality.

## Error Handling

- The script includes error handling for model download, saving, and prediction processes, ensuring smooth execution and debugging.

## Logging

- Logging is utilized throughout the script to keep track of the model's downloading, saving, and prediction processes.

## Note

- The script is an example of integrating Hugging Face models with MLflow and is not intended for production use without further modifications and testing.
