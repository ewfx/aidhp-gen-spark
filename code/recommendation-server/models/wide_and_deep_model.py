import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from google.colab import files

np.random.seed(42)
torch.manual_seed(42)

uploaded = files.upload()

df = pd.read_csv("sample_user_data.csv")
print("Dataset shape:", df.shape)


class DeepModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class WideDeepModel(nn.Module):
    def __init__(self, wide_dim, deep_input_dim, hidden_dim):
        super(WideDeepModel, self).__init__()
        self.wide = nn.Linear(wide_dim, 1)
        self.deep = DeepModel(deep_input_dim, hidden_dim, 1)

    def forward(self, wide_input, deep_input):
        wide_out = self.wide(wide_input)
        deep_out = self.deep(deep_input)
        combined = wide_out + deep_out  
        return torch.sigmoid(combined)  

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.numpy().flatten()

def get_user_bert_embedding(texts):
    embeddings = [get_text_embedding(text) for text in texts if text.strip() != ""]

    if not embeddings:
        return np.zeros(768, dtype=np.float32)
    return np.mean(embeddings, axis=0)


def get_wide_features_from_row(row):
    age = row["age"]
    income = row["income"]
    visits = row["number_of_visits"]

    features = [age / 100.0, income / 200000.0, visits / 10.0]

    features += [0.0] * (10 - len(features))
    return np.array(features, dtype=np.float32)

def get_deep_features_from_row(row):
    try:
        sentiments = json.loads(row["social_media_sentiment"])
    except Exception as e:
        sentiments = []
    texts = [s["content"] for s in sentiments] if sentiments else [""]
    bert_emb = get_user_bert_embedding(texts)  
    
    product_emb = np.random.rand(32).astype(np.float32)
    deep_feature = np.concatenate([bert_emb, product_emb])  
    return deep_feature.astype(np.float32)


def get_label():
    return np.random.randint(0, 2, size=(1,)).astype(np.float32)[0]


wide_features_list = []
deep_features_list = []
labels_list = []

for idx, row in df.iterrows():
    wide_feat = get_wide_features_from_row(row)
    deep_feat = get_deep_features_from_row(row)
    label = get_label()  
    wide_features_list.append(wide_feat)
    deep_features_list.append(deep_feat)
    labels_list.append([label])

wide_features_array = np.array(wide_features_list)
deep_features_array = np.array(deep_features_list)
labels_array = np.array(labels_list)

print("Wide features shape:", wide_features_array.shape)   
print("Deep features shape:", deep_features_array.shape)   
print("Labels shape:", labels_array.shape)

wide_tensor = torch.tensor(wide_features_array)
deep_tensor = torch.tensor(deep_features_array)
labels_tensor = torch.tensor(labels_array)


num_samples = wide_tensor.shape[0]
batch_size = 64
num_epochs = 30
learning_rate = 0.0005
wide_dim = 10
deep_input_dim = 800
hidden_dim = 100

model = WideDeepModel(wide_dim, deep_input_dim, hidden_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_batches = num_samples // batch_size

print("\nStarting Training...\n")
for epoch in range(num_epochs):
    permutation = torch.randperm(num_samples)
    epoch_loss = 0.0
    for i in range(num_batches):
        indices = permutation[i*batch_size:(i+1)*batch_size]
        batch_wide = wide_tensor[indices]
        batch_deep = deep_tensor[indices]
        batch_labels = labels_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_wide, batch_deep)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "wide.pth")
print("\nModel trained and saved as 'wide.pth'")


# Load the trained model (if needed in a separate session, run this cell)
def load_trained_model(wide_dim=10, deep_input_dim=800, hidden_dim=100):
    model = WideDeepModel(wide_dim, deep_input_dim, hidden_dim)
    model.load_state_dict(torch.load("wide.pth"))
    model.eval()
    return model

loaded_model = load_trained_model()


