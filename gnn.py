# '''
#     EDA of dmg777k dataset
# '''
# import matplotlib.pyplot as plt
# import numpy as np
# from kgbench import load
# from collections import Counter, defaultdict
# import seaborn as sns
# import torch
# from typing import Tuple, List, Dict
# import pandas as pd

# def analyze_splits(dataset):
#     """Analyze available data splits in the dataset"""
#     split_info = {}
    
#     # Check for different possible split attributes
#     split_attributes = ['train_idx', 'val_idx', 'test_idx',
#                        'train_mask', 'val_mask', 'test_mask']
    
#     for attr in split_attributes:
#         if hasattr(dataset, attr):
#             if isinstance(getattr(dataset, attr), torch.Tensor):
#                 split_info[attr] = len(getattr(dataset, attr))
#             elif isinstance(getattr(dataset, attr), np.ndarray):
#                 split_info[attr] = len(getattr(dataset, attr))
#             else:
#                 split_info[attr] = sum(getattr(dataset, attr))
    
#     return split_info

# def analyze_triples(dataset) -> Tuple[Dict, Dict]:
#     """Analyze the triples in the dataset"""
#     if not hasattr(dataset, 'triples'):
#         return {}, {}
    
#     # Analyze subject-predicate-object patterns
#     relation_patterns = defaultdict(int)
#     entity_roles = defaultdict(lambda: {'subject': 0, 'object': 0})
    
#     for s, p, o in dataset.triples:
#         # Convert to integers if they're tensors
#         s = s.item() if isinstance(s, torch.Tensor) else s
#         p = p.item() if isinstance(p, torch.Tensor) else p
#         o = o.item() if isinstance(o, torch.Tensor) else o
        
#         # Get entity types if possible
#         s_type = type(dataset.i2e[s]).__name__ if s < len(dataset.i2e) else 'Unknown'
#         o_type = type(dataset.i2e[o]).__name__ if o < len(dataset.i2e) else 'Unknown'
        
#         pattern = f"{s_type} --[{p}]--> {o_type}"
#         relation_patterns[pattern] += 1
        
#         entity_roles[s]['subject'] += 1
#         entity_roles[o]['object'] += 1
    
#     return dict(relation_patterns), dict(entity_roles)

# def analyze_node_information(dataset) -> Dict:
#     """Analyze information available per node"""
#     node_info = {}
    
#     # Check for node features
#     if hasattr(dataset, 'x') and dataset.x is not None:
#         node_info['features'] = {
#             'shape': dataset.x.shape,
#             'type': dataset.x.dtype,
#             'sparse': isinstance(dataset.x, torch.sparse.Tensor)
#         }
    
#     # Check for node labels
#     if hasattr(dataset, 'y') and dataset.y is not None:
#         node_info['labels'] = {
#             'shape': dataset.y.shape,
#             'unique_labels': len(torch.unique(dataset.y)),
#             'type': dataset.y.dtype
#         }
    
#     return node_info

# def plot_entity_distribution(dataset):
#     """Plot the distribution of entity types"""
#     entity_types = defaultdict(int)
    
#     for entity in dataset.i2e:
#         if isinstance(entity, tuple):
#             entity_type = entity[1] if len(entity) > 1 else 'untyped_literal'
#         else:
#             entity_type = type(entity).__name__
#         entity_types[entity_type] += 1
    
#     plt.figure(figsize=(12, 6))
#     plt.bar(entity_types.keys(), entity_types.values(), color='skyblue')
#     plt.xticks(rotation=45, ha='right')
#     plt.title('Distribution of Entity Types')
#     plt.ylabel('Count')
#     plt.tight_layout()
#     plt.show()

# def plot_edge_distribution(dataset):
#     """Plot the distribution of edge types"""
#     if not hasattr(dataset, 'triples'):
#         print("No triples found in dataset")
#         return
    
#     edge_types = Counter([p.item() if isinstance(p, torch.Tensor) else p 
#                          for _, p, _ in dataset.triples])
    
#     plt.figure(figsize=(12, 6))
#     sns.barplot(x=list(range(len(edge_types))), 
#                y=list(edge_types.values()),
#                color='lightgreen')
#     plt.xlabel('Edge Type ID')
#     plt.ylabel('Count')
#     plt.title('Distribution of Edge Types')
#     plt.tight_layout()
#     plt.show()

# def plot_entity_roles(entity_roles):
#     """Plot the distribution of entity roles (subject vs object)"""
#     if not entity_roles:
#         return
    
#     # Prepare data for plotting
#     subjects = [roles['subject'] for roles in entity_roles.values()]
#     objects = [roles['object'] for roles in entity_roles.values()]
    
#     plt.figure(figsize=(12, 6))
#     plt.scatter(subjects, objects, alpha=0.5)
#     plt.xlabel('Times used as Subject')
#     plt.ylabel('Times used as Object')
#     plt.title('Entity Usage as Subject vs Object')
#     plt.tight_layout()
#     plt.show()

# def main():
#     # Load dataset
#     print("Loading DMG777K dataset...")
#     dataset = load('dmg777k')
    
#     # Print all available attributes
#     print("\n=== Available Dataset Attributes ===")
#     for attr in dir(dataset):
#         # if not attr.startswith('_'):
#         print(f"- {attr}")
    
#     # 1. Entity Analysis
#     print("\n=== Entity Analysis ===")
#     if hasattr(dataset, 'i2e'):
#         print(f"Total number of entities: {len(dataset.i2e)}")
#         plot_entity_distribution(dataset)
    
#     # 2. Edge Analysis
#     print("\n=== Edge Analysis ===")
#     if hasattr(dataset, 'triples'):
#         print(f"Total number of triples: {len(dataset.triples)}")
#         print(f"Number of relation types: {dataset.num_relations}")
#         plot_edge_distribution(dataset)
    
#     # 3. Node Information Analysis
#     print("\n=== Node Information Analysis ===")
#     node_info = analyze_node_information(dataset)
#     for key, value in node_info.items():
#         print(f"\n{key.capitalize()} information:")
#         for k, v in value.items():
#             print(f"- {k}: {v}")
    
#     # 4. Triple Analysis
#     print("\n=== Triple Analysis ===")
#     relation_patterns, entity_roles = analyze_triples(dataset)
    
#     print("\nTop 10 relation patterns:")
#     for pattern, count in sorted(relation_patterns.items(), 
#                                key=lambda x: x[1], reverse=True)[:10]:
#         print(f"- {pattern}: {count}")
    
#     # Plot entity roles
#     if entity_roles:
#         plot_entity_roles(entity_roles)
    
#     # 5. Entity to Index Mapping Analysis
#     print("\n=== Entity-Index Mapping Analysis ===")
#     if hasattr(dataset, 'i2e'):
#         print(f"Number of mappings in i2e: {len(dataset.i2e)}")
#     if hasattr(dataset, 'e2i'):
#         print(f"Number of mappings in e2i: {len(dataset.e2i)}")
    
#     # 6. Data Split Analysis
#     print("\n=== Data Split Information ===")
#     split_info = analyze_splits(dataset)
#     if split_info:
#         for split_name, size in split_info.items():
#             print(f"{split_name}: {size}")
#     else:
#         print("No standard split information found")
        
#         # Check for custom split attributes
#         custom_splits = [attr for attr in dir(dataset) 
#                         if any(x in attr.lower() for x in ['train', 'test', 'val', 'split'])]
#         if custom_splits:
#             print("\nFound custom split attributes:")
#             for split in custom_splits:
#                 value = getattr(dataset, split)
#                 if isinstance(value, (torch.Tensor, np.ndarray, list)):
#                     print(f"- {split}: {len(value)} items")
#                 else:
#                     print(f"- {split}: {type(value)}")
    
#     # 7. GNN Training Considerations
#     print("\n=== GNN Training Considerations ===")
#     print("1. Graph Structure:")
#     if hasattr(dataset, 'triples'):
#         print(f"- Using {len(dataset.triples)} triples to construct the graph")
#         print("- Consider using a relation-aware GNN to leverage edge types")
    
#     print("\n2. Node Features:")
#     if hasattr(dataset, 'x') and dataset.x is not None:
#         print(f"- Input feature dimension: {dataset.x.shape[1]}")
#         print("- Features are available for GNN input")
#     else:
#         print("- No node features found")
#         print("- Consider using:")
#         print("  * One-hot encodings")
#         print("  * Positional encodings")
#         print("  * Pre-trained entity embeddings")
    
#     print("\n3. Training Setup:")
#     if split_info:
#         print("- Use provided data splits for training")
#     else:
#         print("- Need to create custom train/val/test splits")
#         print("- Consider using stratified sampling if task is classification")
    
#     if hasattr(dataset, 'y') and dataset.y is not None:
#         print(f"\n4. Task Type:")
#         print(f"- Classification task with {len(torch.unique(dataset.y))} classes")
#         print("- Consider using:")
#         print("  * Cross entropy loss")
#         print("  * Class weights if distribution is imbalanced")
    
#     # Save summary statistics
#     summary_stats = {
#         'total_entities': len(dataset.i2e) if hasattr(dataset, 'i2e') else 0,
#         'total_relations': dataset.num_relations if hasattr(dataset, 'num_relations') else 0,
#         'total_triples': len(dataset.triples) if hasattr(dataset, 'triples') else 0
#     }
    
#     if split_info:
#         summary_stats.update(split_info)
    
#     pd.DataFrame([summary_stats]).to_csv('dmg777k_summary_stats.csv', index=False)

# if __name__ == "__main__":
#     main()

"""
Multimodal GNN for Node Classification with Embedding Caching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from transformers import CLIPProcessor, CLIPModel
from transformers import DistilBertModel, DistilBertTokenizer
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import numpy as np
import tqdm
import logging
import os
import pickle
import hashlib
from datetime import datetime
from kgbench import load, tic, toc
import fire

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultimodalEncoder(nn.Module):
    def __init__(self, output_dim, device='cuda'):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        
        logger.info("Initializing CLIP model...")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        logger.info("Initializing DistilBERT model...")
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        self.clip_projection = nn.Linear(512, output_dim).to(device)
        self.bert_projection = nn.Linear(768, output_dim).to(device)
        self.iri_embedding = nn.Embedding(10000, output_dim).to(device)
        
        self.pca = PCA(n_components=output_dim)
    
    @torch.no_grad()
    def encode_images(self, images, batch_size=16):
        embeddings = []
        for i in tqdm.tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            batch = images[i:i + batch_size]
            inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.clip.get_image_features(**inputs)
            embeddings.append(self.clip_projection(image_features).cpu())
        return torch.cat(embeddings, dim=0)
    
    @torch.no_grad()
    def encode_text(self, texts, batch_size=32):
        embeddings = []
        for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
            batch = texts[i:i + batch_size]
            inputs = self.bert_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.bert(**inputs).last_hidden_state[:, 0]
            embeddings.append(self.bert_projection(text_features).cpu())
        return torch.cat(embeddings, dim=0)

class MultimodalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        self.output = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        return self.output(x)

class DMGDataProcessor:
    def __init__(self, data, output_dim=64, device='cuda', cache_dir='cached_embeddings'):
        self.data = data
        self.output_dim = output_dim
        self.device = device
        self.cache_dir = cache_dir
        self.encoder = MultimodalEncoder(output_dim, device)
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, datatype):
        hash_input = f"{datatype}_{self.output_dim}"
        filename = hashlib.md5(hash_input.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{filename}.pkl")
    
    def load_cached_embeddings(self, datatype):
        cache_path = self.get_cache_path(datatype)
        if os.path.exists(cache_path):
            logger.info(f"Loading cached embeddings for {datatype}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_embeddings(self, embeddings, datatype):
        cache_path = self.get_cache_path(datatype)
        logger.info(f"Caching embeddings for {datatype}")
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
    
    def process_node_type(self, dtype):
        cached_embeddings = self.load_cached_embeddings(dtype)
        if cached_embeddings is not None:
            return cached_embeddings
        
        if dtype in ['iri', 'blank_node']:
            strings = self.data.get_strings(dtype=dtype)
            if strings:
                logger.info(f"Processing {dtype} nodes...")
                features = torch.randn(len(strings), self.output_dim)
                self.save_embeddings(features, dtype)
                return features
        elif dtype == 'http://kgbench.info/dt#base64Image':
            images = self.data.get_images()
            if images:
                logger.info("Processing image nodes...")
                features = self.encoder.encode_images(images)
                self.save_embeddings(features, dtype)
                return features
        else:
            strings = self.data.get_strings(dtype=dtype)
            if strings:
                logger.info(f"Processing {dtype} nodes...")
                features = self.encoder.encode_text(strings)
                self.save_embeddings(features, dtype)
                return features
        return None
    
    def process_data(self):
        logger.info("Processing DMG777k dataset...")
        node_features = []
        
        datatypes = ['iri', 'blank_node', 'http://kgbench.info/dt#base64Image'] + \
                   [dt for dt in self.data.datatypes() if dt not in ['iri', 'blank_node', 'http://kgbench.info/dt#base64Image']]
        
        for dtype in datatypes:
            features = self.process_node_type(dtype)
            if features is not None:
                node_features.append(features)
        
        node_features = torch.cat(node_features, dim=0)
        edge_index = torch.tensor(self.data.triples[:, [0, 2]], dtype=torch.long).t()
        train_idx = torch.tensor(self.data.training[:, 0], dtype=torch.long)
        val_idx = torch.tensor(self.data.withheld[:, 0], dtype=torch.long)
        
        num_nodes = node_features.size(0)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        
        labels = torch.zeros(num_nodes, dtype=torch.long)
        labels[train_idx] = torch.tensor(self.data.training[:, 1], dtype=torch.long)
        labels[val_idx] = torch.tensor(self.data.withheld[:, 1], dtype=torch.long)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask
        )

def train(model, data, optimizer, device, epochs=100, patience=10):
    model = model.to(device)
    data = data.to(device)
    
    best_val_acc = 0
    best_epoch = 0
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        val_acc = evaluate(model, data, data.val_mask)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            train_acc = evaluate(model, data, data.train_mask)
            logger.info(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    logger.info(f'Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}')
    return model

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    correct = pred.eq(data.y[mask]).sum().item()
    return correct / mask.sum().item()

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def main(
    dataset='dmg777k',
    hidden_dim=128,
    output_dim=64,
    num_heads=4,
    dropout=0.1,
    lr=0.001,
    weight_decay=5e-4,
    epochs=100,
    patience=10,
    save_dir='checkpoints',
    cache_dir='cached_embeddings'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info(f"Loading {dataset} dataset...")
    data = load(dataset)
    
    processor = DMGDataProcessor(data, output_dim=output_dim, device=device, cache_dir=cache_dir)
    processed_data = processor.process_data()
    
    model = MultimodalGNN(
        input_dim=output_dim,
        hidden_dim=hidden_dim,
        num_classes=data.num_classes,
        num_heads=num_heads,
        dropout=dropout
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    logger.info("Starting training...")
    tic()
    model = train(model, processed_data, optimizer, device, epochs, patience)
    training_time = toc()
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"model_{timestamp}.pt")
    save_model(model, save_path)
    
    train_acc = evaluate(model, processed_data, processed_data.train_mask)
    val_acc = evaluate(model, processed_data, processed_data.val_mask)
    logger.info(f"Final results - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

if __name__ == "__main__":
    fire.Fire(main)
