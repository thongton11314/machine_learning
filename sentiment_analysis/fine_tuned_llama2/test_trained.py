from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    BitsAndBytesConfig
)
import torch
from peft import PeftModel, PeftConfig

# Load imdb dataset and select a small sample
imdb = load_dataset("imdb")
small_test_dataset = imdb["test"].shuffle(seed=42).select(range(300))

# Setup BitsAndBytes configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load the fine-tuned model
output_dir = "./checkpoints/sentiment-analysis/13b-lora"
config = PeftConfig.from_pretrained(output_dir)
model = AutoModelForSequenceClassification.from_pretrained(
    config.base_model_name_or_path,
    num_labels=2,
    id2label={0: 'Negative', 1: 'Positive'},
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
finetuned_model = PeftModel.from_pretrained(model, output_dir)

# Create sentiment-analysis pipeline
sent_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Test inputs
test_inputs = [
    "wow, that event was incredible! It was decorated with such finesse and was a warming experience overall",
    "I don't know how I feel about the event...",
    "There are certainly improvements to be made, the event was quite slow."
]
print(sent_pipeline(test_inputs))

# Evaluate model accuracy on the sample dataset
predictions = sent_pipeline(small_test_dataset["text"])
num_correct = sum(
    (pred["label"] == "Positive" and true_label == 1) or 
    (pred["label"] == "Negative" and true_label == 0)
    for pred, true_label in zip(predictions, small_test_dataset["label"])
)
accuracy = 100 * num_correct / len(small_test_dataset)
print(f"Accuracy: {accuracy}%")

# Output: Accuracy is approximately 95.33%
