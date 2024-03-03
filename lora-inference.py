from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "./model"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map = 'auto')
model.cuda()
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
model.cuda()

inputs = tokenizer("ROZDZIAŁ PIERWSZY", return_tensors='pt').to('cuda')

output_tokens = model.generate(
  **inputs,
  max_new_tokens=512,
  temperature=0.75,
  top_p=0.95,
  do_sample=True,
).to('cuda')

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))