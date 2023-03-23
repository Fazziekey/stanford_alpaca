# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import torch
import fire
import time


from pathlib import Path

from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig

MODEL_PATH = "output"
# MODEL_PATH = "/data/scratch/LLaMa-7B/"
def main(path: str=MODEL_PATH):



    model = LlamaForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(path)

    # prompt = "Tell me five words that rhyme with 'shock'."
    # prompt = "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'."
    # prompt = "Write a Python program that prints the first 10 Fibonacci numbers."
    # prompt = "Write a Python program that calculate Fibonacci numbers."
    # prompt = "List all Canadian provinces in alphabetical order"
    # prompt = "Tell me about the president of Mexico in 2019."
    prompt = "How to play support in legends of league"

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')

    # Generate config

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
    )
    # Generate
    start_time = time.time()
    with torch.no_grad():
        generate_ids = model.generate(input_ids, 
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=512,
        )

    for s in generate_ids.sequences:
    # s = generate_ids.sequences[0]
        result = tokenizer.decode(s)
        print("---------------------------")
        print("### Response:\n")
        print(result)
        print("---------------------------\n")
    
    print(f"Generated in {time.time() - start_time:.2f} seconds")



if __name__ == "__main__":
    fire.Fire(main)
