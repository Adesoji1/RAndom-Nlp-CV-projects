# create a Text Summarization pytorch script

"""
This script uses the torch and transformers libraries to create a text summarization script that uses the BERT model. The script reads in the input text, tokenizes it, and then encodes it using the BERT model. The encoded representation is then used to create a summary by taking the mean of the encoded text. The resulting summary is printed to the console.
"""


import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# move the model to the device
model = model.to(device)

# read in the input text
with open('input_text.txt', 'r') as f:
    text = f.read()

# tokenize the text
tokens = tokenizer.tokenize(text)

# create the token indices
indices = tokenizer.encode(text, max_length=512)

# create the attention masks
masks = [1] * len(indices)

# convert the token indices and attention masks to tensors
tokens_tensor = torch.tensor([indices]).to(device)
masks_tensor = torch.tensor([masks]).to(device)

# set the model to evaluation mode
model.eval()

# get the encoded representation of the input text
with torch.no_grad():
    encoded_text = model(tokens_tensor, masks_tensor)[0]

# create a summary by taking the mean of the encoded representation
summary = torch.mean(encoded_text, dim=1)

# print the summary
print(summary)


