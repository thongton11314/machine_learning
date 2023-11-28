# Sentiment Analysis with RoBERTa

This project demonstrates how to perform sentiment analysis using the RoBERTa model from the Hugging Face Transformers library. The code provided in this repository includes steps to load a pre-trained RoBERTa model, tokenize input text, and predict the sentiment of the given text.

## Overview

The script uses `MAQSoftware/roberta-sa-e1`, a RoBERTa-based model specifically fine-tuned for sentiment analysis. It includes functions to load the model and tokenizer, and predict the sentiment of any input text.

## Prerequisites

To use this script, you need the following:
- Python 3.10
- `transformers` library
- An internet connection to download the model and tokenizer

## Installation

1. Install the `transformers` library:
   ```bash
   pip install transformers
   ```

2. Clone this repository to your local machine.

3. Obtain a Hugging Face API token and set it as an environment variable named `HUGGINGFACE_TOKEN`.

## Usage

### Loading the Model

The `load_model` function loads the pre-trained RoBERTa model and its tokenizer. It requires the model name (`"MAQSoftware/roberta-sa-e1"`) and the Hugging Face API token.

Example:
```python
model, tokenizer = load_model("MAQSoftware/roberta-sa-e1", your_hf_token)
```

### Predicting Sentiment

There are two ways to predict sentiment:

1. **Standard Prediction:** Uses the `predict_sentiment` function. It takes the input text, model, and tokenizer as arguments and returns the sentiment classification.

   Example:
   ```python
   sentiment_result = predict_sentiment("I love you", model, tokenizer)
   if sentiment_result:
       print(f"Sentiment Result: {sentiment_result}")
   ```

2. **Detailed Output:** Provides more detailed output, including the predicted class index.

   Example:
   ```python
   tokenized_input = tokenizer("I love you", return_tensors="pt")
   output = model(**tokenized_input)
   predicted_class = output.logits.argmax(-1).item() 
   print(f"Predicted Class: {predicted_class}")
   ```

## Error Handling

The script includes basic error handling for loading the model and predicting sentiment. In case of an error, it prints an error message.

## Contributing

Contributions to improve this project are welcome. Please adhere to the standard practices for submitting pull requests.

## License

Specify the license under which this project is available.

---

*Note: Replace `your_hf_token` with your actual Hugging Face API token.*
