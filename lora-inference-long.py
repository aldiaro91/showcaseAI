from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "./model"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map='auto')
model.cuda()
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
model.cuda()

initial_prompt = "Rozdział pierwszy"
inputs = tokenizer(initial_prompt, return_tensors='pt').to('cuda')

# Ustal ilość tokenów do wygenerowania
total_tokens_to_generate = 4000
tokens_generated = 0
new_tokens_per_iteration = 32
previous_tokens_provided_for_context = 736
full_text = initial_prompt

while tokens_generated < total_tokens_to_generate:
    print('inputs length:', len(inputs[0]))
    print('inputs:', inputs[0])
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=new_tokens_per_iteration,
        temperature=0.75,
        top_p=0.95,
        do_sample=True,
    ).to('cuda')
    
    # Dodaj tylko nowo wygenerowany tekst (bez promptu) do pełnego tekstu
    new_text = tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    full_text += new_text
    tokens_generated += len(tokenizer.encode(new_text))
    print('tokens generated:', tokens_generated)
    
    # Ustaw ostatnie wygenerowane tokeny jako nowe wejście dla kolejnej iteracji
    if tokens_generated < total_tokens_to_generate:
        inputs = tokenizer(full_text, return_tensors='pt').to('cuda')
        inputs["input_ids"] = inputs["input_ids"][:, -previous_tokens_provided_for_context:]
        if "attention_mask" in inputs:
          inputs["attention_mask"] = inputs["attention_mask"][:, -previous_tokens_provided_for_context:]

# Zapisz pełny tekst do pliku tekstowego
with open("wygenerowana_ksiazka.txt", "w", encoding="utf-8") as file:
    file.write(full_text)

print("Książka została zapisana w pliku 'wygenerowana_ksiazka.txt'")

