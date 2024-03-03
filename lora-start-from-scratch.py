from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import os

# Define the model and tokenizer
model_name = "sdadas/polish-gpt2-medium"
output_path = "./model"
print('loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
print('loading model...')
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=['c_attn', 'c_proj', 'c_fc'],
    lora_dropout=0.05,
    bias="all",
    task_type="CAUSAL_LM"
)

print(model)
model = get_peft_model(model, config)
model = model._prepare_model_for_gradient_checkpointing(model)

# Define a function to load and process the data from a directory
def load_and_process_data(data_dir, tokenizer, block_size):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_dir,
        block_size=min(block_size, tokenizer.model_max_length),
    )
    return dataset

# Directory containing your input files
data_directory = "./data-harry-potter"

# Process each file in the directory
for filename in os.listdir(data_directory):
    if filename.endswith(".txt") and "cache" not in filename:
        file_path = os.path.join(data_directory, filename)

        # Load and process the data for this file
        print('loading dataset...')
        dataset = load_and_process_data(file_path, tokenizer, block_size=768)
        print("DATASET:")
        print(dataset)

        # Define the data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Define training arguments
        training_args = TrainingArguments(
            gradient_checkpointing=True,
            output_dir=output_path,  # Output directory for each file
            overwrite_output_dir=True,
            num_train_epochs=12,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=3e-4,
            save_steps=100,
            save_total_limit=2,
            logging_dir="./logs",
            evaluation_strategy="steps",
            eval_steps=1000000,
            logging_steps=10,
        )

        # Create a Trainer instance for this file
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        # Start the fine-tuning process for this file
        trainer.train()

        # Save the fine-tuned model and tokenizer for this file
        trainer.save_model(output_path)

        # Save the tokenizer files for this file
        tokenizer.save_pretrained(output_path)
