import os
import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


def main(input_file_path, model_name_suffix, system_prompt):
    # Extract base name and remove extension to get file name
    base_name = os.path.basename(input_file_path)
    file_name_without_extension = os.path.splitext(base_name)[0]

    # Create full model name
    full_model_name = f"meta-llama/Llama-2-{model_name_suffix}-chat-hf"

    # Read the prompts CSV file
    df = pd.read_csv(input_file_path)

    promptTemplate = """<s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {user_message} [/INST]"""

    # Define a function to format the prompt
    def format_prompt(row):
        user_message = row["prompt"]
        return promptTemplate.format(
            system_prompt=system_prompt, user_message=user_message
        )

    # Apply the function to the prompt column
    df["formatted_prompt"] = df.apply(format_prompt, axis=1)

    # Initialize the model and tokenizer
    cache_directory = "./model_cache"
    tokenizer = AutoTokenizer.from_pretrained(
        full_model_name, cache_dir=cache_directory
    )
    model = AutoModelForCausalLM.from_pretrained(
        full_model_name, cache_dir=cache_directory
    )

    # Initialize the text generation pipeline
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device=0,
    )

    # Initialize an empty list to store results
    df["answer"] = ""

    # Loop through each row in the DataFrame to generate text
    for index, row in df.iterrows():
        prompt = row["formatted_prompt"]
        generated_text = text_gen_pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
            max_new_tokens=1024,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
        )[0]["generated_text"]

        # Append the ID and generated text to the results list
        df.at[index, "answer"] = generated_text

    # Drop unneeded column
    df = df.drop(columns=["formatted_prompt"])

    # Create output file name based on input file name
    output_file_path = (
        f"{model_name_suffix}_answer_for_{file_name_without_extension}.csv"
    )

    # Save the DataFrame to a CSV file
    df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text based on given prompts and model."
    )
    parser.add_argument(
        "input_file_path",
        type=str,
        help="Path to the input CSV file containing prompts.",
    )
    parser.add_argument(
        "model_name_suffix", type=str, help="Suffix for the model name."
    )
    parser.add_argument(
        "system_prompt",
        type=str,
        help="System prompt text to use in the prompt template.",
    )
    args = parser.parse_args()

    main(args.input_file_path, args.model_name_suffix, args.system_prompt)
