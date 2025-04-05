import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from nltk.metrics.distance import edit_distance
import numpy as np

# Siamese Network Definition
class SiameseNetwork(nn.Module):
    def __init__(self, word_embeddings, numerical_input_size):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(torch.FloatTensor(word_embeddings), freeze=True)
        self.fc_text = nn.Sequential(
            nn.Linear(2 * word_embeddings.shape[1], 32),
            nn.ReLU(inplace=True)
        )
        self.fc_numerical = nn.Sequential(
            nn.Linear(numerical_input_size, 32),
            nn.ReLU(inplace=True)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
        )

    def forward_one(self, text_input, numerical_input):
        text_embedded = self.embedding(text_input)
        text_output = self.fc_text(text_embedded)

        numerical_output = self.fc_numerical(numerical_input)

        combined_output = torch.cat((text_output, numerical_output), 1)
        final_output = self.fc_final(combined_output)
        return final_output

    def forward(self, text_input1, numerical_input1, text_input2, numerical_input2):
        output1 = self.forward_one(text_input1, numerical_input1)
        output2 = self.forward_one(text_input2, numerical_input2)
        return output1, output2

# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

# Custom Dataset
class SiameseDataset(Dataset):
    def __init__(self, names, addresses, ages, incomes, labels, word_embeddings):
        self.names = names
        self.addresses = addresses
        self.ages = ages
        self.incomes = incomes
        self.labels = labels
        self.word_embeddings = word_embeddings

    def __getitem__(self, index):
        name1, name2 = self.names[index]
        address1, address2 = self.addresses[index]
        age1, age2 = self.ages[index]
        income1, income2 = self.incomes[index]
        label = torch.Tensor([self.labels[index]])

        # Apply pre-trained word embeddings (Word2Vec)
        name_embedding1 = torch.tensor([self.word_embeddings[word] for word in name1.split() if word in self.word_embeddings])
        name_embedding2 = torch.tensor([self.word_embeddings[word] for word in name2.split() if word in self.word_embeddings])

        address_embedding1 = torch.tensor([self.word_embeddings[word] for word in address1.split() if word in self.word_embeddings])
        address_embedding2 = torch.tensor([self.word_embeddings[word] for word in address2.split() if word in self.word_embeddings])

        # Calculate string distance features (Levenshtein)
        name_distance = torch.tensor([edit_distance(name1, name2)])
        address_distance = torch.tensor([edit_distance(address1, address2)])

        return (name_embedding1, address_embedding1, age1, income1), \
               (name_embedding2, address_embedding2, age2, income2), label

    def __len__(self):
        return len(self.labels)

# Sample Data (Replace this with your actual data)
names = [("John Doe", "Jane Doe"), ("Alice Smith", "Bob Johnson")]
addresses = [("123 Main St", "456 Oak Ave"), ("789 Pine Ln", "101 Elm Rd")]
ages = [(30, 25), (40, 45)]
incomes = [(50000, 60000), (75000, 80000)]
labels = [0, 1]

# Pre-trained Word Embeddings (Word2Vec, Replace with your own embeddings)
word_embeddings = {"John": np.random.rand(50), "Jane": np.random.rand(50), "Alice": np.random.rand(50),
                   "Bob": np.random.rand(50), "Doe": np.random.rand(50), "Smith": np.random.rand(50)}

# Split data into train and test sets
data_train, data_test, labels_train, labels_test = train_test_split(
    list(zip(names, addresses, ages, incomes)), labels, test_size=0.2, random_state=42
)

# Custom Siamese Dataset
train_dataset = SiameseDataset(
    [pair[0] for pair in data_train], [pair[1] for pair in data_train],
    [pair[2][0] for pair in data_train], [pair[3][0] for pair in data_train], labels_train, np.array(list(word_embeddings.values()))
)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)

# Initialize Siamese Network and Loss Function
siamese_net = SiameseNetwork(np.array(list(word_embeddings.values())), numerical_input_size=2)
criterion = ContrastiveLoss()
optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    for (name1, address1, age1, income1), (name2, address2, age2, income2), label in train_loader:
        optimizer.zero_grad()
        output1, output2 = siamese_net((name1, address1), (age1, income1), (name2, address2), (age2, income2))
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Save or use the trained Siamese network for similarity tasks

import spacy
from spacy.training.example import Example
import random

# Sample training data (replace this with your labeled dataset)
TRAIN_DATA = [
    ("Document reference number: ABC123", {"entities": [(28, 34, "DOCUMENT_REFERENCE")], "category": "General"}),
    ("Ref: XYZ456 is the document identifier", {"entities": [(5, 11, "DOCUMENT_REFERENCE")], "category": "General"}),
    ("Invoice for services rendered. Total amount: $500.00", {"entities": [(27, 36, "DOCUMENT_REFERENCE")], "category": "Invoice", "amount": 500.00}),
    # Add more examples as needed
]

def train_spacy_ner_and_classifier(train_data, iterations=100):
    nlp = spacy.blank("en")
    
    # Named Entity Recognition
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
    ner.add_label("DOCUMENT_REFERENCE")

    # Text Classification
    text_cat = nlp.create_pipe("text_cat", config={"exclusive_classes": True, "architecture": "simple_cnn"})
    nlp.add_pipe(text_cat, last=True)
    
    # Add category labels
    for category in ["General", "Invoice", "Category3", "Category4", "Category5", "Category6", "Category7", "Category8", "Category9", "Category10"]:
        text_cat.add_label(category)

    # Initialize the training loop
    optimizer = nlp.begin_training()

    for itn in range(iterations):
        random.shuffle(train_data)
        losses = {}

        # Update the model with each example in the training data
        for text, annotations in train_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.5, losses=losses)

        print(losses)

    return nlp

# Train the model
trained_model = train_spacy_ner_and_classifier(TRAIN_DATA)

# Save the trained model to disk
trained_model.to_disk("trained_model")

# Load the trained model from disk
loaded_model = spacy.load("trained_model")

# Test the model on new text
test_text = "Invoice for services rendered. Total amount: $800.00"
doc = loaded_model(test_text)

# Extract entities from the document
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

# Classify the document category
print(f"Document Category: {doc.cats}")

# Extract total amount for invoices
if "Invoice" in doc.cats and doc.cats["Invoice"] > 0.5:
    print(f"Total Amount: {doc._.amount}")


yieldlosspergroup = losstrack.apply(lambda x: 0 if x['DEFECT_TYPE']=='Good' else x['DEFECTS_COUNT'] / x['VolumeGroup'] ,axis=1)



