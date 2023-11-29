import numpy as np
import torch
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
import evaluate

class SentimentAnalyzer:
    """A class for fine-tuning a sentiment analysis model."""
    
    def __init__(self, model_id, dataset_name, seed=42, output_dir="./checkpoints/sentiment-analysis/tweet-13b-lora"):
        """Initializes the SentimentAnalyzer class with the given model_id, dataset_name, and seed."""
        
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.seed = seed
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.training_arguments = None
        self.ourput_dir = output_dir
    
    # This function is for returning a string representation of the SentimentAnalyzer class
    def __str__(self):
        """Returns a string representation of the SentimentAnalyzer class."""
        
        return f"SentimentAnalyzer(model_id={self.model_id}, dataset_name={self.dataset_name}, seed={self.seed})"
    
    # This function is for running the SentimentAnalyzer class
    def run(self):
        """Runs the SentimentAnalyzer class."""
        
        self.load_data()
        self.__setup_tokenizer()
        self.__preprocess_data()
        self.__setup_model()
        self.__setup_trainer()
        self.__train()
        
    # This function is for loading the data
    def load_data(self):
        """Loads the data from the dataset_name and splits into train and test datasets of size 3000 and 300 respectively."""
        # https://huggingface.co/datasets/mteb/tweet_sentiment_extraction
        tweet = load_dataset("mteb/tweet_sentiment_extraction")
        raw_dataset = tweet["train"].shuffle(seed=self.seed).select([i for i in list(range(3000))])
        raw_dataset = raw_dataset.train_test_split(test_size=0.2, seed=self.seed)
        self.small_train_dataset = raw_dataset["train"]
        self.small_test_dataset = raw_dataset["test"]
           
    # This function is for preprocessing the data 
    def preprocess_data(self):
        """Tokenizes the text in the datasets."""
        
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)
        self.tokenized_train = self.small_train_dataset.map(preprocess_function, batched=True)
        self.tokenized_test = self.small_test_dataset.map(preprocess_function, batched=True)
        
    # This function is for setting up the tokenizer
    def setup_tokenizer(self):
        """Sets up the tokenizer."""
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.bos_token_id = 1
        self.tokenizer.padding_side = 'right'

    # This function is for setting up the model
    def setup_model(self):
        """Sets up the model."""
        
        # Setup the bitsandbytes config for 4bit quantization - optimization for memory
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # LoRA attention dimension
        lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
        
        # Setup lora config, specify save modules to not lose fine-tuned weights later - optimization for memory
        peft_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=lora_target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            modules_to_save=lora_target_modules.append("score"),
        )

        # Sets up the model.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, 
            num_labels=3, 
            id2label={0: 'negative', 1: 'neutral', 2: 'positive'}
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map='balanced'
        )
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
    # This function is for computing the metrics
    def compute_metrics(self, eval_pred):
        """Computes the metrics for the given evaluation prediction."""
        
        # Loads the accuracy metric.
        load_accuracy = evaluate.load("accuracy")
        
        # Computes the metrics for the given evaluation prediction.
        load_f1 = evaluate.load("f1")
        
        # Computes the metrics for the given evaluation prediction.
        logits, labels = eval_pred
        
        # Computes the predictions for the given logits.
        predictions = np.argmax(logits, axis=-1)
        
        # Computes the accuracy for the given predictions and labels.
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        
        # Computes the f1 score for the given predictions and labels.
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}
    
    # This function is used for setting up the trainer
    def setup_trainer(self):
        
        """Sets up the trainer argument"""
        training_arguments = TrainingArguments(
            output_dir=self.ourput_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            save_steps=100,
            logging_steps=10,
            learning_rate=2e-4,
            fp16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            resume_from_checkpoint=True,
            save_strategy="epoch"
        )

        """Sets up the trainer."""
        self.trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_test,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics,
        )
        
    # This function is for training the model
    def train(self):
        """Trains the model."""
        
        self.trainer.train()
    
    # This funtion is for evaluating the model
    def evaluate(self):
        """Evaluates the model."""
        
        return self.trainer.evaluate()
    
    # This function is for saving the model
    def save_model(self, output_dir):
        """Saves the model to the given output directory."""
        
        self.trainer.model.save_pretrained(output_dir)

# This function is for running the SentimentAnalyzer class
if __name__ == "__main__":
    timeStart = time.time()
    analyzer = SentimentAnalyzer(model_id="Llama-2-13b-chat-hf/",
                                dataset_name="tweet",
                                output_dir = "./checkpoints/sentiment-analysis/tweet-13b-lora")
    analyzer.run()
    evaluation_results = analyzer.evaluate()
    print(evaluation_results)
    print("Time taken: ", -timeStart + time.time())
