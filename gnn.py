# import kgbench as kg
# import torch
# import torch.nn as nn
# import torch_geometric
# from torch_geometric.data import Data
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter
# from transformers import CLIPProcessor, CLIPModel
# from transformers import BertTokenizer, BertModel
# import torchvision.models as models
# from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool

# # 1. Data Loading and EDA
# def load_and_analyze_data(data):
#     """
#     Perform Exploratory Data Analysis on the dmg777k dataset
#     """
#     # Extract node types and counts
#     node_types = Counter(data.node_types)
    
#     # Plot node type distribution
#     plt.figure(figsize=(12, 6))
#     plt.bar(node_types.keys(), node_types.values())
#     plt.title('Distribution of Node Types')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     # Extract edge types and counts
#     edge_types = Counter(data.edge_types)
    
#     # Plot edge type distribution
#     plt.figure(figsize=(12, 6))
#     plt.bar(edge_types.keys(), edge_types.values())
#     plt.title('Distribution of Edge Types')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     # Analyze node attributes
#     print("Node attributes available:", data.node_features.keys())
#     print("Number of nodes:", len(data.nodes))
#     print("Number of edges:", len(data.triples))
    
#     return node_types, edge_types

# # 2. Feature Extraction
# class FeatureExtractor:
#     def __init__(self):
#         # Initialize CLIP for vision features
#         self.clip_model = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
#         # Initialize BERT for text features
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
        
#     def extract_image_features(self, image):
#         """Extract features from images using CLIP"""
#         inputs = self.clip_model(images=image, return_tensors="pt")
#         image_features = self.clip.get_image_features(**inputs)
#         return image_features
    
#     def extract_text_features(self, text):
#         """Extract features from text using BERT"""
#         inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         outputs = self.bert(**inputs)
#         return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token

# # 3. GNN Model
# class MultimodalGNN(nn.Module):
#     def __init__(self, node_feature_dim, num_classes, hidden_dim=256):
#         super(MultimodalGNN, self).__init__()
        
#         # GNN layers
#         self.conv1 = GCNConv(node_feature_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
#         # Classification head
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(hidden_dim // 2, num_classes)
#         )
        
#     def forward(self, x, edge_index):
#         # Graph convolutions
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
        
#         # Classification
#         out = self.classifier(x)
#         return out

# # 4. Training Pipeline
# def train_model(model, data, optimizer, criterion, num_epochs=100):
#     model.train()
    
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
        
#         # Forward pass
#         out = model(data.x, data.edge_index)
#         loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
#         # Backward pass
#         loss.backward()
#         optimizer.step()
        
#         # Validation
#         if epoch % 10 == 0:
#             model.eval()
#             with torch.no_grad():
#                 val_out = model(data.x, data.edge_index)
#                 val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
                
#                 # Calculate accuracy
#                 pred = val_out[data.val_mask].argmax(dim=1)
#                 correct = pred.eq(data.y[data.val_mask]).sum().item()
#                 acc = correct / data.val_mask.sum().item()
                
#                 print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}')
#             model.train()

# # 5. Main execution
# def main():
#     # Load dataset
#     data = kg.load('dmg777k')
    
#     # Perform EDA
#     node_types, edge_types = load_and_analyze_data(data)
    
#     # Initialize feature extractor
#     feature_extractor = FeatureExtractor()
    
#     # Process nodes and extract features
#     node_features = []
#     for node in data.nodes:
#         if node['type'] == 'image':
#             features = feature_extractor.extract_image_features(node['image'])
#         else:
#             features = feature_extractor.extract_text_features(node['text'])
#         node_features.append(features)
    
#     # Convert to PyG Data object
#     x = torch.stack(node_features)
#     edge_index = torch.tensor(data.edge_index)
#     y = torch.tensor(data.labels)
    
#     # Create and train model
#     model = MultimodalGNN(
#         node_feature_dim=node_features[0].shape[1],
#         num_classes=len(set(data.labels))
#     )
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     criterion = nn.CrossEntropyLoss()
    
#     # Train the model
#     train_model(model, Data(x=x, edge_index=edge_index, y=y), optimizer, criterion)

# if __name__ == "__main__":
#     main()

import kgbench as kg
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Optional, Union, List
import gc

MAX_ENTITIES = 40000  # Reduced to 4000 entities

class FeatureExtractor:
    def __init__(self, feature_dim: int = 768):
        self.feature_dim = feature_dim
        print("Initializing feature extractors...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
    def extract_batch_features(self, entity_ids: List[int], data, batch_size: int = 32) -> torch.Tensor:
        """Extract features for a batch of entities"""
        entity_ids = entity_ids[:MAX_ENTITIES]
        features = []
        
        for i in range(0, len(entity_ids), batch_size):
            batch_ids = entity_ids[i:i + batch_size]
            batch_features = []
            
            try:
                strings = data.get_strings_batch(batch_ids, dtype=str)
                images = None
                if not any(strings):
                    images = data.get_images_batch(batch_ids)
            except Exception as e:
                print(f"Error fetching batch data for entities {batch_ids[0]}-{batch_ids[-1]}: {e}")
                strings = [None] * len(batch_ids)
                images = None
            
            for idx, entity_id in enumerate(batch_ids):
                try:
                    if strings and strings[idx]:
                        feat = self.extract_text_features(strings[idx])
                    elif images and images[idx] is not None:
                        feat = self.extract_image_features(images[idx])
                    else:
                        feat = torch.zeros(self.feature_dim)
                except Exception as e:
                    print(f"Error processing entity {entity_id}: {e}")
                    feat = torch.zeros(self.feature_dim)
                
                batch_features.append(feat)
            
            del strings
            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            features.extend(batch_features)
            
            if (i + batch_size) % 1000 == 0:
                print(f"Processed {min(i + batch_size, MAX_ENTITIES)}/{MAX_ENTITIES} entities...")
        
        return torch.stack([f.detach() for f in features])
    
    def extract_image_features(self, image) -> torch.Tensor:
        if image is None:
            return torch.zeros(self.feature_dim)
        
        with torch.no_grad():
            try:
                inputs = self.clip_processor(images=image, return_tensors="pt")
                image_features = self.clip_model.get_image_features(**inputs)
                return image_features.mean(dim=0)[:self.feature_dim]
            except Exception as e:
                print(f"Error processing image: {e}")
                return torch.zeros(self.feature_dim)
    
    def extract_text_features(self, text: str) -> torch.Tensor:
        if not text or not isinstance(text, str):
            return torch.zeros(self.feature_dim)
        
        with torch.no_grad():
            try:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = self.bert(**inputs)
                return outputs.last_hidden_state[:, 0, :].squeeze()[:self.feature_dim]
            except Exception as e:
                print(f"Error processing text: {e}")
                return torch.zeros(self.feature_dim)

def process_dataset(data, batch_size=32):
    """Process the dataset and extract features with improved entity handling"""
    print("Analyzing dataset structure...")
    print(f"Total number of entities: {data.num_entities}")
    print(f"Number of relations: {data.num_relations}")
    
    # Analyze triple structure
    min_head, max_head, min_tail, max_tail = analyze_triples(data)
    
    # Adjust MAX_ENTITIES based on the actual entity distribution
    global MAX_ENTITIES
    suggested_max = max(max_head, max_tail) + 1
    if suggested_max > MAX_ENTITIES:
        print(f"\nWarning: Current MAX_ENTITIES ({MAX_ENTITIES}) is too low.")
        print(f"Adjusting MAX_ENTITIES to {suggested_max} to include all entities")
        print("Felfnejbf",MAX_ENTITIES)
        MAX_ENTITIES = suggested_max
    
    print(f"\nProcessing first {MAX_ENTITIES} entities")
    
    feature_extractor = FeatureExtractor()
    entity_ids = list(range(min(data.num_entities, MAX_ENTITIES)))
    
    print("Extracting features...")
    features = feature_extractor.extract_batch_features(entity_ids, data, batch_size=batch_size)
    
    print("Creating graph structure...")
    filtered_triples = [t for t in data.triples if t[0] < MAX_ENTITIES and t[2] < MAX_ENTITIES]
    
    if not filtered_triples:
        raise ValueError("No valid triples found after filtering. Check entity ID distribution in triples.")
    
    print(f"Number of filtered triples: {len(filtered_triples)}")
    print(f"Sample of first 5 triples: {filtered_triples[:5]}")
    
    edge_index = torch.tensor([[t[0], t[2]] for t in filtered_triples]).t().contiguous()
    edge_type = torch.tensor([t[1] for t in filtered_triples])
    
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge type shape: {edge_type.shape}")
    
    print("Creating labels...")
    labels = torch.zeros(MAX_ENTITIES, dtype=torch.long)
    
    # Create train/val masks
    train_mask = torch.zeros(MAX_ENTITIES, dtype=torch.bool)
    val_mask = torch.zeros(MAX_ENTITIES, dtype=torch.bool)
    
    # Split 80/20 for train/val
    num_train = int(0.8 * MAX_ENTITIES)
    train_mask[:num_train] = True
    val_mask[num_train:] = True
    
    print("Preparing PyG data object...")
    pyg_data = Data(
        x=features,
        edge_index=edge_index,
        edge_type=edge_type,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask
    )
    
    print("\nFinal data object validation:")
    print(f"Number of nodes: {pyg_data.num_nodes}")
    print(f"Number of edges: {pyg_data.num_edges}")
    print(f"Feature matrix shape: {pyg_data.x.shape}")
    print(f"Edge index shape: {pyg_data.edge_index.shape}")
    print(f"Edge type shape: {pyg_data.edge_type.shape}")
    
    return pyg_data

class MultimodalGNN(nn.Module):
    def __init__(self, feature_dim, num_relations, num_classes=2, hidden_dim=256):
        super(MultimodalGNN, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_relations = num_relations
        
        # Relation embeddings
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        
        # GNN layers
        self.conv1 = GCNConv(feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        
        edge_embeddings = self.relation_embedding(edge_type)
        x = x + torch.index_select(edge_embeddings, 0, edge_index[1])
        
        x = self.conv2(x, edge_index)
        
        out = self.classifier(x)
        return out

def train_model(model, data, optimizer, criterion, device, num_epochs=100):
    model.train()
    data = data.to(device)
    
    print("Training on device:", device)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_type)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, data.edge_type)
                val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
                
                pred = val_out[data.val_mask].argmax(dim=1)
                correct = pred.eq(data.y[data.val_mask]).sum().item()
                acc = correct / data.val_mask.sum().item()
                
                print(f'Epoch {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}')
            model.train()

def analyze_triples(data):
    """Analyze the distribution of entity IDs in triples"""
    head_entities = [t[0] for t in data.triples]
    tail_entities = [t[2] for t in data.triples]
    
    print("\nTriple Analysis:")
    print(f"Total number of triples: {len(data.triples)}")
    print(f"Head entity ID range: {min(head_entities)} to {max(head_entities)}")
    print(f"Tail entity ID range: {min(tail_entities)} to {max(tail_entities)}")
    
    # Count triples that would be filtered out
    valid_triples = [t for t in data.triples if t[0] < MAX_ENTITIES and t[2] < MAX_ENTITIES]
    print(f"Number of valid triples (with MAX_ENTITIES={MAX_ENTITIES}): {len(valid_triples)}")
    
    return min(head_entities), max(head_entities), min(tail_entities), max(tail_entities)

def main():
    print("Loading dataset...")
    data = kg.load('dmg777k')
    
    batch_size = 16
    
    print(f"Processing dataset...")
    try:
        pyg_data = process_dataset(data, batch_size=batch_size)
    except Exception as e:
        print(f"Error during dataset processing: {e}")
        return
    
    if pyg_data.num_edges == 0:
        print("Error: Graph has no edges. Cannot proceed with training.")
        return
    
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalGNN(
        feature_dim=768,
        num_relations=data.num_relations
    ).to(device)
    
    print("Setting up training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    train_model(model, pyg_data, optimizer, criterion, device)

if __name__ == "__main__":
    main()