import fire
import sys
import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import json
import hashlib
from pathlib import Path
import os
import gc
from collections import Counter
import pickle
from torch.nn import LayerNorm, Dropout
import transformers as tf
import open_clip
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np

from kgbench import load, tic, toc, d
import kgbench as kg
from pathlib import Path
import json
import pickle
import os

class EmbeddingCache:
    def __init__(self, cache_dir='/content/drive/MyDrive/EmbeddingCache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)  # Ensure directory exists
        self.metadata_file = self.cache_dir / 'metadata.json'
        self.load_metadata()

    def load_metadata(self):
        """Load metadata from the JSON file."""
        if self.metadata_file.exists():
            print("Loading metadata from file")
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self.save_metadata()

    def save_metadata(self):
        """Save metadata to the JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

    def get_embedding_filename(self, dtype, model_name):
        """Generate a filename based on datatype and model name."""
        dtype_clean = dtype.replace('http://kgbench.info/dt#', '').replace('@', '_').replace('/', '_')
        filename = f"embeddings_{dtype_clean}_{model_name}.pkl"
        return self.cache_dir / filename

    def get_metadata_key(self, dtype, model_name):
        """Generate a metadata key based on datatype and model name."""
        return f"{dtype}_{model_name}"

    def get_cached_embeddings(self, data_items, dtype, model_name):
        """Retrieve cached embeddings if they exist."""
        embedding_file = self.get_embedding_filename(dtype, model_name)
        metadata_key = self.get_metadata_key(dtype, model_name)
        print("*" * 199)
        print(f"Checking cache: {embedding_file}, Key: {metadata_key}")
        print("*" * 199)

        if embedding_file.exists() and metadata_key in self.metadata:
            print(f"Loading cached embeddings from {embedding_file}")
            with open(embedding_file, 'rb') as f:
                embeddings = pickle.load(f)
            # Verify the number of items matches
            if self.metadata[metadata_key]['num_items'] == len(data_items):
                print(f"Using cached embeddings for dtype {dtype}")
                return embeddings
            else:
                print(f"Cached embeddings found but item count mismatch: "
                      f"{self.metadata[metadata_key]['num_items']} vs {len(data_items)}")
                return None
        print(f"No cached embeddings found for dtype {dtype}")
        return None

    def cache_embeddings(self, embeddings, data_items, dtype, model_name):
        """Save embeddings to a file and update metadata."""
        embedding_file = self.get_embedding_filename(dtype, model_name)
        metadata_key = self.get_metadata_key(dtype, model_name)

        # Save embeddings
        with open(embedding_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Saved embeddings to {embedding_file}")

        # Update metadata
        self.metadata[metadata_key] = {
            'dtype': dtype,
            'model': model_name,
            'num_items': len(data_items),
            'embedding_shape': list(embeddings.shape),
            'created': str(embedding_file.stat().st_mtime)
        }
        self.save_metadata()
        print(f"Updated metadata for {metadata_key}")

def enrich(triples: torch.Tensor, n: int, r: int):
    cuda = triples.is_cuda

    inverses = torch.cat([
        triples[:, 2:],
        triples[:, 1:2] + r,
        triples[:, :1]
    ], dim=1)

    selfloops = torch.cat([
        torch.arange(n, dtype=torch.long, device=d(cuda))[:, None],
        torch.full((n, 1), fill_value=2*r, device=d(cuda)),
        torch.arange(n, dtype=torch.long, device=d(cuda))[:, None],
    ], dim=1)

    return torch.cat([triples, inverses, selfloops], dim=0)

def sum_sparse(indices, values, size, row=True):
    ST = torch.cuda.sparse.FloatTensor if indices.is_cuda else torch.sparse.FloatTensor

    k, r = indices.size()

    if not row:
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)
        size = size[1], size[0]

    ones = torch.ones((size[1], 1), device=d(indices), dtype=torch.float32)  # Explicitly set dtype
    smatrix = ST(indices.t(), values, size=size)
    sums = torch.mm(smatrix, ones)

    return sums[indices[:, 0]].view(k)

def adj(triples, num_nodes, num_rels, cuda=False, vertical=True):
    r, n = num_rels, num_nodes
    size = (r * n, n) if vertical else (n, r * n)

    from_indices = []
    upto_indices = []

    for fr, rel, to in triples:
        offset = rel.item() * n

        if vertical:
            fr = offset + fr.item()
        else:
            to = offset + to.item()

        from_indices.append(fr)
        upto_indices.append(to)

    indices = torch.tensor([from_indices, upto_indices], dtype=torch.long, device=d(cuda))

    return indices.t(), size
class RGCN(nn.Module):
    def __init__(self, triples, n, r, emb, hidden, numcls,dropout_general=0.2,dropout_feature=0.1, bases=None):
        """
        Fixed RGCN initialization with correct parameter names

        Args:
            triples: Knowledge graph triples
            n: Number of entities
            r: Number of relations
            emb: Embedding size (previously insize)
            hidden: Hidden layer size
            numcls: Number of output classes
            bases: Number of bases for weight decomposition
        """
        super().__init__()

        self.emb = emb  # Changed from insize to emb
        self.hidden = hidden
        self.bases = bases
        self.numcls = numcls

        # Dropout layers
        self.dropout = nn.Dropout(dropout_general)
        self.feature_dropout = nn.Dropout(dropout_feature)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.layer_norm2 = nn.LayerNorm(numcls)

        self.triples = enrich(triples, n, r)

        hor_ind, hor_size = adj(self.triples, n, 2*r+1, vertical=False)
        ver_ind, ver_size = adj(self.triples, n, 2*r+1, vertical=True)

        _, rn = hor_size
        r = rn // n

        vals = torch.ones(ver_ind.size(0), dtype=torch.float32)
        vals = vals / sum_sparse(ver_ind, vals, ver_size)

        hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size)
        self.register_buffer('hor_graph', hor_graph)

        ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)
        self.register_buffer('ver_graph', ver_graph)

        if bases is None:
            self.weights1 = nn.Parameter(torch.FloatTensor(r, emb, hidden))  # Changed insize to emb
            self.weights2 = nn.Parameter(torch.FloatTensor(r, hidden, numcls))
            nn.init.xavier_uniform_(self.weights1, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.weights2, gain=nn.init.calculate_gain('relu'))
            self.bases1 = None
            self.bases2 = None
        else:
            self.comps1 = nn.Parameter(torch.FloatTensor(r, bases))
            self.comps2 = nn.Parameter(torch.FloatTensor(r, bases))
            self.bases1 = nn.Parameter(torch.FloatTensor(bases, emb, hidden))  # Changed insize to emb
            self.bases2 = nn.Parameter(torch.FloatTensor(bases, hidden, numcls))
            nn.init.xavier_uniform_(self.comps1, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.comps2, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.bases1, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.bases2, gain=nn.init.calculate_gain('relu'))

        self.bias1 = nn.Parameter(torch.FloatTensor(hidden).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(numcls).zero_())

    def forward(self, features):
        # Ensure input features are float32
        features = features.float()

        # Apply initial feature dropout
        features = self.feature_dropout(features)

        n, rn = self.hor_graph.size()
        r = rn // n
        b, c = self.bases, self.numcls

        h = torch.mm(self.ver_graph, features)
        h = h.view(r, n, features.size(1))

        if self.bases1 is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps1, self.bases1)
        else:
            weights = self.weights1

        h = torch.bmm(h, weights).sum(dim=0)
        h = self.layer_norm1(h + self.bias1)  # Added layer normalization
        h = F.relu(h)
        h = self.dropout(h)  # Added dropout

        h = torch.mm(self.ver_graph, h)
        h = h.view(r, n, self.hidden)

        if self.bases2 is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps2, self.bases2)
        else:
            weights = self.weights2

        h = torch.bmm(h, weights).sum(dim=0)
        h = self.layer_norm2(h + self.bias2)  # Added layer normalization
        h = self.dropout(h)  # Added final dropout

        return h

    def penalty(self, p=2):
        if self.bases is None:
            return self.weights1.pow(2).sum() + self.weights2.pow(2).sum()  # Added weights2 to penalty
        return self.comps1.pow(p).sum() + self.comps2.pow(p).sum() + \
               self.bases1.pow(p).sum() + self.bases2.pow(p).sum()


def clip_emb(pilimages, bs=8):  # Reduced batch size
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    image_embeddings = []
    nimages = len(pilimages)

    try:
        for i in tqdm.tqdm(range(0, nimages, bs)):
            # Clear GPU cache at the start of each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            batch = pilimages[i:i + bs]
            # Convert and resize images before preprocessing
            batch = [img.convert('RGB').resize((224, 224)) for img in batch]

            try:
                processed_images = torch.stack([preprocess(img) for img in batch])

                if torch.cuda.is_available():
                    processed_images = processed_images.cuda()

                with torch.no_grad():
                    image_features = model.encode_image(processed_images)
                    image_features = F.normalize(image_features, dim=-1)
                    # Move to CPU immediately to free GPU memory
                    image_embeddings.append(image_features.cpu().float())

                # Explicitly delete intermediate tensors
                del processed_images, image_features
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    # If OOM occurs, try processing one image at a time
                    print("OOM error, processing one image at a time...")
                    for single_img in batch:
                        processed_img = preprocess(single_img).unsqueeze(0)
                        if torch.cuda.is_available():
                            processed_img = processed_img.cuda()
                        with torch.no_grad():
                            feat = model.encode_image(processed_img)
                            feat = F.normalize(feat, dim=-1)
                            image_embeddings.append(feat.cpu().float())
                        del processed_img, feat
                        torch.cuda.empty_cache()
                else:
                    raise e

            # Force garbage collection
            gc.collect()

    except Exception as e:
        print(f"Error during image processing: {str(e)}")
        raise

    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return torch.cat(image_embeddings, dim=0)

MNAME = 'distilbert-base-cased'
bmodel = tf.DistilBertModel.from_pretrained(MNAME)
btok = tf.DistilBertTokenizerFast.from_pretrained(MNAME)

def bert_emb(strings, bs_chars=10_000):
    pbar = tqdm.tqdm(total=len(strings))
    outs = []
    fr = 0

    while fr < len(strings):
        to = fr
        bs = 0
        while bs < bs_chars and to < len(strings):
            bs += len(strings[to])
            to += 1

        strbatch = strings[fr:to]

        with torch.no_grad():
            batch = btok(strbatch, padding=True, truncation=True, return_tensors="pt")
            inputs, mask = batch['input_ids'], batch['attention_mask']

            if torch.cuda.is_available():
                inputs, mask = inputs.cuda(), mask.cuda()

            out = bmodel(inputs, mask)
            outs.append(out[0][:, 0, :].cpu().float())
            torch.cuda.empty_cache()

        pbar.update(len(strbatch))
        fr = to

    return torch.cat(outs, dim=0)

def pca(tensor, target_dim):
    device = tensor.device
    n, f = tensor.size()
    if n < 25:
        return tensor[:, :target_dim]

    # Move to CPU for PCA
    if tensor.is_cuda:
        tensor = tensor.cpu()

    tensor = tensor.float()

    if n > 10000:
        ipca = IncrementalPCA(n_components=target_dim)
        batch_size = 1000
        for i in range(0, n, batch_size):
            batch = tensor[i:i + batch_size]
            ipca.partial_fit(batch)

        result = []
        for i in range(0, n, batch_size):
            batch = tensor[i:i + batch_size]
            result.append(torch.from_numpy(ipca.transform(batch)))
        res = torch.cat(result, dim=0)
    else:
        model = PCA(n_components=target_dim, whiten=True)
        res = torch.from_numpy(model.fit_transform(tensor))

    # Move back to original device
    return res.to(device)

def compute_embeddings_with_cache(data, emb_dim, cache_dir='/content/drive/MyDrive/EmbeddingCache'):
    cache = EmbeddingCache(cache_dir)
    embeddings = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for datatype in data.datatypes():
        print(f'Processing datatype: {datatype}')

        try:
            if datatype in ['iri', 'blank_node']:
                print(f'Initializing embedding for datatype {datatype}.')
                n = len(data.get_strings(dtype=datatype))
                nodes = torch.randn(n, emb_dim, device=device)
                embeddings.append(nodes)

            elif datatype == 'http://kgbench.info/dt#base64Image':
                print(f'Computing/loading embeddings for images.')
                images = data.get_images()
                model_name = 'clip-vit-b-32'
                cached_emb = cache.get_cached_embeddings(images, datatype, model_name)

                if cached_emb is not None:
                    image_embeddings = cached_emb.to(device)
                else:
                    print(f"Computing CLIP embeddings for {len(images)} images")
                    image_embeddings = clip_emb(images, bs=8)
                    print("Caching image embeddings")
                    cache.cache_embeddings(image_embeddings.cpu(), images, datatype, model_name)
                    image_embeddings = image_embeddings.to(device)

                image_embeddings = pca(image_embeddings, target_dim=emb_dim)
                embeddings.append(image_embeddings)

            else:
                print(f'Computing/loading embeddings for datatype {datatype}.')
                strings = data.get_strings(dtype=datatype)
                model_name = 'distilbert-base-cased'
                cached_emb = cache.get_cached_embeddings(strings, datatype, model_name)

                if cached_emb is not None:
                    string_embeddings = cached_emb.to(device)
                else:
                    print(f"Computing BERT embeddings for {len(strings)} strings")
                    string_embeddings = bert_emb(strings)
                    print("Caching text embeddings")
                    cache.cache_embeddings(string_embeddings.cpu(), strings, datatype, model_name)
                    string_embeddings = string_embeddings.to(device)

                string_embeddings = pca(string_embeddings, target_dim=emb_dim)
                embeddings.append(string_embeddings)

        except Exception as e:
            print(f"Error processing datatype {datatype}: {str(e)}")
            raise

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print("Verifying device consistency...")
    for i, emb in enumerate(embeddings):
        if emb.device != device:
            print(f"Warning: Moving embedding {i} to {device}")
            embeddings[i] = emb.to(device)

    return torch.cat(embeddings, dim=0)

def go(name='dmg777k',patience=10, lr=0.005, wd=5e-2, l2=5e-2, epochs=500, prune=True,
       optimizer='adam', final=False, emb=16,dropout_general=0.2,dropout_feature=0.1,label_smoothing=0.1, bases=40,
       cache_dir='/content/drive/MyDrive/EmbeddingCache', imagebatch=8,
       stringbatch=5_000,
       printnorms=None):

    # Set default tensor type to float32
    torch.set_default_tensor_type(torch.FloatTensor)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    print('Loading data...')
    data = load(name, torch=True, prune_dist=2 if prune else None, final=final)
    data = kg.group(data)

    # Convert training and withheld data to long tensors
    data.training = data.training.long()
    data.withheld = data.withheld.long()

    print(f'{data.triples.size(0)} triples')
    print(f'{data.num_entities} entities')
    print(f'{data.num_relations} relations')

    if torch.cuda.is_available():
        bmodel.cuda()
        print('CUDA available. Using GPU.')

    tic()
    print('Computing embeddings...')
    with torch.no_grad():
        embeddings = compute_embeddings_with_cache(
            data=data,
            emb_dim=emb,
            cache_dir=cache_dir
        ).float()  # Ensure float32
    print(f'Embeddings computed in {toc():.2f} seconds')

    # Split embeddings into trainable and non-trainable parts
    num_uri = len(data.datatype_l2g('uri'))
    num_bnode = len(data.datatype_l2g('blank_node'))
    numparms = num_uri + num_bnode

    trainable = embeddings[:numparms, :]
    constant = embeddings[numparms:, :]

    trainable = nn.Parameter(trainable)

    print('Initializing RGCN...')
    tic()
    rgcn = RGCN(
        triples=data.triples.long(),  # Ensure triples are long tensors
        n=data.num_entities,
        r=data.num_relations,
        emb=emb,  # Changed from insize to emb
        hidden=emb,
        numcls=data.num_classes,
        dropout_general=dropout_general,
        dropout_feature=dropout_feature,
        bases=bases
    )

    if torch.cuda.is_available():
        rgcn.cuda()
        trainable = trainable.cuda()
        constant = constant.cuda()
        data.training = data.training.cuda()
        data.withheld = data.withheld.cuda()

    print(f'RGCN initialized in {toc():.2f} seconds')

    # Initialize optimizer
    if optimizer == 'adam':
        opt = torch.optim.Adam(lr=lr, weight_decay=wd, params=[*rgcn.parameters(), trainable])
    elif optimizer == 'adamw':
        opt = torch.optim.AdamW(lr=lr, weight_decay=wd, params=[*rgcn.parameters(), trainable])
    else:
        raise ValueError(f'Unknown optimizer: {optimizer}')

    # Add a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode='min',
    factor=0.5,
    patience=patience,
    verbose=True
)

    # Training loop
    best_withheld_acc = 0
    best_state = None

    print('\nStarting training...')
    for e in range(epochs):
        tic()
        opt.zero_grad()

        # Combine trainable and constant embeddings
        features = torch.cat([trainable, constant], dim=0)
        out = rgcn(features)

        # Get training and validation indices and classes
        idxt, clst = data.training[:, 0], data.training[:, 1]
        idxw, clsw = data.withheld[:, 0], data.withheld[:, 1]

        # Compute loss
        out_train = out[idxt, :]
        loss = F.cross_entropy(out_train, clst, reduction='mean',label_smoothing=label_smoothing)
        if l2 != 0.0:
            loss = loss + l2 * rgcn.penalty()

        # Compute metrics
        with torch.no_grad():
            training_acc = (out[idxt, :].argmax(dim=1) == clst).sum().item() / idxt.size(0)
            withheld_acc = (out[idxw, :].argmax(dim=1) == clsw).sum().item() / idxw.size(0)

            # Save best model
            if withheld_acc > best_withheld_acc:
                best_withheld_acc = withheld_acc
                best_state = {
                    'rgcn': rgcn.state_dict(),
                    'trainable': trainable.detach().clone(),
                    'epoch': e,
                    'acc': withheld_acc
                }

        # Backward pass and optimization
        loss.backward()
        opt.step()

        # Step the scheduler after the 500th epoch
        scheduler.step(loss)

        print(f'Epoch {e:02d}: loss {loss:.4f}, train acc {training_acc:.4f}, '
              f'valid acc {withheld_acc:.4f} ({toc():.2f}s)')

        # Clear cache periodically
        if e % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Save best model if requested
    if best_state is not None:
        save_path = Path(cache_dir) / f'best_model_{name}.pt'
        torch.save(best_state, save_path)
        print(f'\nBest model saved (epoch {best_state["epoch"]}, '
              f'acc {best_state["acc"]:.4f})')

    return best_withheld_acc

if __name__=="__main__":
    go(   name='dmg777k',
          patience=5,
          lr=0.005,
          wd=1e-2,
          l2=1e-2,
          epochs=500,
          prune=True,
          optimizer='adamw',
          final=False,
          emb=16,
          dropout_general=0.1,
          dropout_feature=0.1,
          label_smoothing=0.1,
          bases=40,
          cache_dir='/content/drive/MyDrive/EmbeddingCache',
          imagebatch=8,
          stringbatch=5_000,
          printnorms=None
           )
