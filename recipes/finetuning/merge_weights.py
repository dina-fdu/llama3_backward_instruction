# import torch
from transformers import AutoTokenizer
import os
from peft import AutoPeftModelForCausalLM

model_path = '/mnt/data1/zwzhu/models/Meta-Llama-3-8B-Instruct-PEFT-backward'
model = AutoPeftModelForCausalLM.from_pretrained(model_path)
model = model.merge_and_unload()

output_merged_dir = "/mnt/data1/zwzhu/models/Meta-Llama-3-8B-Instruct-PEFT-backward-merged"
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy inference
model_name = "meta-llama/Meta-Llama-3-8B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_merged_dir)