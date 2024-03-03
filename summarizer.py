from transformers import BartTokenizer, BartForConditionalGeneration

model_name = "facebook/bart-large-cnn"
max_chunk_length = 3000

def chunk_text(text):
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    for paragraph in paragraphs:
        # Adding 2 to account for newline characters
        new_length = len(current_chunk) + len(paragraph) + 2
        if new_length <= max_chunk_length:
            current_chunk += '\n\n' + paragraph
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph

    chunks.append(current_chunk)
    return chunks

def summarize_book(file_path, output_file):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Move model to GPU
    model.to('cuda')

    with open(file_path, 'r', encoding='utf-8') as file:
        book_text = file.read()

    max_tokens = max_chunk_length
    text_chunks = chunk_text(book_text)

    with open(output_file, 'w', encoding='utf-8') as file:
        for chunk in text_chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_tokens).to('cuda')
            summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, min_length=30)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            print(summary)
            file.write(summary)
            file.write('\n\n')

# Specify the input and output file paths
input_file_path = 'book-summarizer\input4.txt'
output_file_path = 'book-summarizer\summary4.txt'

# Get the summary
summarize_book(input_file_path, output_file_path)
