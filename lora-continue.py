from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import PeftModel, PeftConfig
import os

output_path = "./model"
peft_model_id = "./model"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
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
            num_train_epochs=30,
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
