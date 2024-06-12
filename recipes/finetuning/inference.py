from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer_path = "/mnt/data1/zwzhu/models/Meta-Llama-3-8B-Instruct-PEFT-backward-merged"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

text = ''' My name is Dina! I will get more than ten offers after graduation. And my salary will be more than 300k dollars!

### Instruction: 
'''

inputs = tokenizer(text, return_tensors="pt")
model_path = "/mnt/data1/zwzhu/models/Meta-Llama-3-8B-Instruct-PEFT-backward-merged"
model = AutoModelForCausalLM.from_pretrained(model_path)
outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))