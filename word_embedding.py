import pandas as pd
import numpy as np
from transformers import MPNetTokenizer, MPNetModel
import torch
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Set device to 'mps' for running on Azure Machine Learning Hardware
device = torch.device('mps')

# Load CSV data
df = pd.read_csv('./processed_description.csv')

# Ensure the salary_bin is of type 'category' to handle it more efficiently
df['salary_bin'] = df['salary_bin'].astype('category')
df['processed_description'] = df['processed_description'].astype(str)  # Ensure all text data is string type

# Initialize the MPNet tokenizer and model
tokenizer = MPNetTokenizer.from_pretrained('microsoft/mpnet-base')
model = MPNetModel.from_pretrained('microsoft/mpnet-base')

# Function to tokenize and get embeddings in batches
def get_embeddings(texts, batch_size=32):
    # Create DataLoader for batch processing
    text_dataloader = DataLoader(texts, batch_size=batch_size)

    # Storage for embeddings
    all_embeddings = []

    # Process in batches
    for batch in tqdm(text_dataloader, desc='Processing text batches'):
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, device=device)
        embeddings = outputs.last_hidden_state.mean(dim=1).to('mps')
        embeddings = embeddings.cpu().numpy()
        all_embeddings.append(embeddings)

    # Concatenate all batch embeddings
    embeddings_concatenated = np.concatenate(all_embeddings, axis=0)
    return embeddings_concatenated

# Vectorize the processed_description column
print("Vectorizing texts...")
embeddings = get_embeddings(df['processed_description'])

# Saving embeddings to a CSV file
print("Saving embeddings to CSV...")
df_embeddings = pd.DataFrame(embeddings)
df_embeddings.to_csv('./text_embeddings.csv', index=False)

# Create mappings from salary bins to ranges
print("Mapping salary bins to ranges...")
bin_to_range = df[['salary_bin', 'salary_range']].drop_duplicates().set_index('salary_bin')['salary_range'].to_dict()

# Count words per salary bin
print("Counting words per salary bin...")
word_counts_per_bin = defaultdict(Counter)
for _, row in df.iterrows():
    bin = row['salary_bin']
    words = row['processed_description'].split()
    word_counts_per_bin[bin].update(words)

# Plotting function
def plot_top_words(counter, salary_bin, title):
    top_words = counter.most_common(10)
    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 8))
    plt.barh(words, counts)
    plt.xlabel('Counts')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.savefig(f'./embeddings_summary/top_words_bin_{salary_bin}.png')
    plt.close()

# Writing top words to a text file and saving plots
print("Writing top words to text file and saving plots...")
with open('./top_words_by_salary_bin.txt', 'w') as file:
    for salary_bin, word_counter in word_counts_per_bin.items():
        top_words = word_counter.most_common(10)
        file.write(f'Salary Bin {salary_bin} - Range {bin_to_range[salary_bin]}\n')
        for word, count in top_words:
            file.write(f'{word}: {count}\n')
        file.write('\n')
        plot_top_words(word_counter, salary_bin, f'Salary Bin {salary_bin} - Range {bin_to_range[salary_bin]}')

print("All operations completed successfully.")
