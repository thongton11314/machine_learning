import os
import logging
import json
import numpy as np
import torch
import mlflow
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline, 
    BitsAndBytesConfig
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("mlflow").setLevel(logging.DEBUG)

# Define variables
mlflow_model_path = "./summarizer_mlflow"
model_name_huggingface = "MAQSoftware/Llama-2-32k-Summarization"
model_name_local = "summarization-32k"
access_token = os.environ.get("HUGGINGFACE_TOKEN")

# Paths for the model and tokenizer
model_path = f"./model_{model_name_local}"
tokenizer_path = f"./model_tokenizer_{model_name_local}"

# Download and save the model and tokenizer
if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
    logging.info("Downloading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_huggingface, token=access_token)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_quant_type="fp4", 
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_huggingface, 
            device_map="auto", 
            config=bnb_config, 
            torch_dtype=torch.bfloat16, 
            use_auth_token=access_token
        )
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(tokenizer_path)
    except Exception as e:
        logging.error(f"Error in downloading the model/tokenizer: {e}")
        raise
else:
    logging.info("Model and tokenizer already exist locally.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

# Hugging Face model wrapper class
class HuggingFaceModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

    def predict(self, context, model_input):
        if not model_input:
            raise ValueError('Input is empty')
        input_strings = np.ndarray.tolist(model_input["input_string"])
        summaries = []
        for input_str in input_strings:
            prompt = f"Write a response that summarizes the following:\n### Instruction:\n###Input:\n{input_str}\n### Response:\n"
            chat = [{"role": "system", "content": prompt}]
            chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            summary = self.summarizer(chat)
            summaries.append(summary)
        return json.dumps(summaries)

# Save the model with MLflow
try:
    mlflow.pyfunc.save_model(
        path=mlflow_model_path,
        python_model=HuggingFaceModelWrapper(model, tokenizer),
        artifacts={"model": model_path}
    )
except Exception as e:
    logging.error(f"Error in saving the model with MLflow: {e}")
    raise

# Load the model in `python_function` format
try:
    loaded_model = mlflow.pyfunc.load_model(mlflow_model_path)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error in loading the model: {e}")
    raise

book_stuff = """
Once upon a time, in a land far, far away, there existed a kingdom known as Eldoria. Eldoria was a place of great beauty, nestled between towering mountains and surrounded by lush forests. Its people were known far and wide for their kindness, wisdom, and love for nature.

At the heart of Eldoria stood the majestic Castle Alaria, a sprawling palace of white marble with shimmering spires that seemed to touch the sky. Within the castle walls lived King Alaric and Queen Elara, rulers beloved by their subjects. They were known for their fairness and compassion, and the kingdom prospered under their rule.

But Eldoria was not just known for its rulers; it was also renowned for its magical creatures. In the depths of the Enchanted Forest, which bordered the kingdom, lived creatures that existed nowhere else in the world. The most famous of these creatures was a magnificent unicorn named Seraphina. Seraphina possessed a coat as white as snow, eyes that sparkled like the stars, and a horn that shimmered with enchantment. She was the guardian of the forest, and her presence brought peace and harmony to Eldoria.

One day, a young man named Elandor arrived at the kingdom's borders. He was a traveler, an adventurer who had heard tales of Eldoria's beauty and Seraphina's magic. Elandor had journeyed from a distant land, and he carried with him a secret. He was on a quest to find a legendary artifact known as the "Heart of Eldoria." This ancient relic was said to hold the power to heal any ailment, mend any broken heart, and bring prosperity to any land.

Elandor's arrival stirred curiosity among the people of Eldoria, and soon, his quest became intertwined with the kingdom's fate. King Alaric and Queen Elara invited him to the castle to learn more about his quest. Elandor revealed his purpose, and the monarchs, recognizing the potential benefit of the Heart of Eldoria, pledged their support. They believed that if the artifact's power could be harnessed, it could bring even greater happiness and prosperity to their kingdom.

Elandor, Seraphina, and the royal court embarked on a grand adventure to find the Heart of Eldoria. They journeyed through dark forests, crossed treacherous rivers, and faced magical challenges that tested their courage and determination. Along the way, they encountered mystical beings, made new friends, and discovered the depths of their own hearts.

As they ventured deeper into the enchanted realms of the world, they also uncovered a hidden threat. A malevolent sorcerer named Malachi sought to seize the Heart of Eldoria for his own dark purposes. Malachi had been lurking in the shadows, watching Elandor's quest with malevolent intent.

The climax of their adventure took place in the heart of the Enchanted Forest, where Seraphina used her magic to guide them to the hidden location of the Heart of Eldoria. But just as they reached the artifact, Malachi appeared, wielding dark magic and a determination to claim the Heart for himself.

A great battle ensued, a clash of light and dark magic that shook the very foundations of the forest. Elandor, Seraphina, and the royal court fought valiantly, using the strength of their unity to combat Malachi's malevolence. In the end, it was the combined power of love, friendship, and the pure intentions of the Eldorians that prevailed.

Malachi was defeated, and the Heart of Eldoria was secured. Its magic was used not for personal gain but to heal the land, mend the hearts of those who suffered, and bring prosperity to the kingdom. Eldoria flourished even more under the newfound power of the artifact, and its people lived in peace and harmony for generations to come.

And so, the tale of Eldoria and the Heart of Eldoria became a legendary story, shared with generations to come. It served as a reminder that true power lay not in the artifact itself but in the hearts of those who wielded it with love and selflessness. And the kingdom of Eldoria remained a place of beauty, kindness, and magic for all time.
"""

# Test prediction
try:
    output = loaded_model.predict({'input_string': np.array([book_stuff, book_stuff], dtype='<U100000')})
    print(output)
except Exception as e:
    logging.error(f"Error in prediction: {e}")
