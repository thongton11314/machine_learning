import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, TextStreamer

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Argument Parser for Preparing Huggingface Dataset")
    parser.add_argument("--model", type=str, required=True, help="The model to be evaluated.")
    parser.add_argument("--input", type=str, required=True, help="The input to the model.")
    parser.add_argument("--prompt", type=str, default="Following is an instruction that describes a task. Write a response that appropriately completes the request.", help="System prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="The maximum number of newly generated tokens for causalLM. Defaults to 512.")
    parser.add_argument("--use_text_streamer", action='store_true', help="Enable text streamer output from pipeline. Repeats the chat template.")
    return parser.parse_args()

def load_model_and_tokenizer(model_name):
    """
    Load the model and tokenizer with the specified configuration.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    return model, tokenizer

def setup_pipeline(model, tokenizer, use_text_streamer, max_new_tokens):
    """
    Set up the text generation pipeline with the provided model and tokenizer.
    """
    text_streamer = TextStreamer(tokenizer) if use_text_streamer else None
    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        streamer=text_streamer,
        temperature=0.2,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.2
    )
    return pipe

def main():
    args = parse_arguments()

    model, tokenizer = load_model_and_tokenizer(args.model)
    pipe = setup_pipeline(model, tokenizer, args.use_text_streamer, args.max_new_tokens)

    chat_template = tokenizer.apply_chat_template([{"role": "system", "content": args.prompt}, 
                                                   {"role": "user", "content": args.input}], 
                                                  tokenize=False, 
                                                  add_generation_prompt=True)

    output = pipe(chat_template)
    generated_text = output[0]["generated_text"][len(chat_template):] if not args.use_text_streamer else output

    print("\nChat template: ")
    print("-----------------------------")
    print(chat_template + '\n')
    print("Generated text:")
    print("-----------------------------")
    print(generated_text)

if __name__ == "__main__":
    main()

