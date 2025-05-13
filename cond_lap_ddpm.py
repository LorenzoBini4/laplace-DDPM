import torch
import torch.nn as nn
from torch_scatter import scatter_add
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import ChebConv

# Enable CuDNN benchmarking for optimal kernels
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MerfishCellGraphDataset(InMemoryDataset):
    def __init__(self, csv_path, k=7, root='data'):
        self.csv_path = csv_path
        self.k = k
        super().__init__(root)
        df = pd.read_csv(self.csv_path)
        print("Columns in CSV before processing:", df.columns)
        print("Shape of CSV before processing:", df.shape)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.csv_path)
        # Extract gene columns between '1700022I11Rik' and 'Gad1'
        gene_cols = df.columns[
            df.columns.get_loc('1700022I11Rik') : df.columns.get_loc('Gad1') + 1
        ].tolist()
        feats = df[gene_cols].values
        coords = df[['coord_X', 'coord_Y']].values
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(coords)
        _, idx = nbrs.kneighbors(coords)
        edges = [[i, j] for i, nbr in enumerate(idx) for j in nbr[1:]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data = Data(
            x=torch.tensor(feats, dtype=torch.float),
            edge_index=edge_index,
            pos=torch.tensor(coords, dtype=torch.float),
            gene_cols=gene_cols
        )
        
        # Compute and store Laplacian PE
        max_nodes = 500000  # Set based on your RAM capacity
        if data.num_nodes > max_nodes:
            raise ValueError(f"Graph too large ({data.num_nodes} nodes). "
                             f"Max supported: {max_nodes}")
        
        data.lap_pe = compute_lap_pe(data.edge_index, data.num_nodes)
        # Save with lap_pe
        torch.save(self.collate([data]), self.processed_paths[0])

# Laplacian PE computation
def compute_lap_pe(edge_index, num_nodes, k=10):
    device = edge_index.device
    
    # Convert to CPU-based SciPy sparse matrix (memory efficient)
    edge_index_np = edge_index.cpu().numpy()
    data = np.ones(edge_index_np.shape[1])
    
    # Create symmetric adjacency matrix
    adj = sp.coo_matrix((data, (edge_index_np[0], edge_index_np[1])), 
                       shape=(num_nodes, num_nodes))
    adj = adj + adj.T
    adj.data = np.clip(adj.data, 0, 1)  # Ensure binary adjacency
    
    # Compute normalized Laplacian using sparse operations
    deg = adj.sum(1).A1
    deg_inv_sqrt = 1 / np.sqrt(np.maximum(deg, 1e-8))
    deg_inv_sqrt_mat = sp.diags(deg_inv_sqrt)
    L = sp.eye(num_nodes) - deg_inv_sqrt_mat @ adj @ deg_inv_sqrt_mat

    # Compute smallest k+1 eigenvectors using ARPACK
    eigvals, eigvecs = eigsh(L, k=k+1, which='SM', tol=1e-3)
    
    # Return eigenvectors corresponding to smallest non-zero eigenvalues
    pe = torch.from_numpy(eigvecs[:, 1:k+1]).float().to(device)
    return pe

# Adversarial Laplacian Perturbation
class LaplacianPerturb:
    def __init__(self, alpha_min=0.001, alpha_max=0.1):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def sample(self, edge_index, num_nodes):
        alpha = torch.rand(1, device=device) * (self.alpha_max - self.alpha_min) + self.alpha_min
        E = edge_index.size(1)
        signs = torch.randint(0, 2, (E,), device=device) * 2 - 1
        weights = 1.0 + alpha * signs.float()
        return weights
    
    def adversarial(self, model, x, edge_index, weights, xi=1e-6, epsilon=0.1, ip=1):
        # Keep all tensors on GPU
        edge_index = edge_index.to(x.device)
        num_nodes = x.size(0)
        
        # Use PyTorch sparse tensors
        edge_i, edge_j = edge_index
        adj = torch.sparse_coo_tensor(
            edge_index,
            weights,
            (num_nodes, num_nodes)
        )
        
        # Symmetrize using sparse operations
        adj = (adj + adj.t()).coalesce()
        
        # Degree computation using sparse
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        L = -adj.to_dense()
        L[range(num_nodes), range(num_nodes)] += deg
        
        # Eigendecomposition
        with torch.cuda.amp.autocast(enabled=False):
            L = L.float()
            eigvals, eigvecs = torch.linalg.eigh(L)
        
            lap_pe = eigvecs[:, 1:11]  # First k=10 non-trivial

            # Forward encoder
            mu, _ = model(x, edge_index.to(x.device), lap_pe)

            loss = (mu ** 2).mean()
            loss.backward()

            if w.grad is None:
                raise RuntimeError("No gradient on adversarial weights")

            g_norm = F.normalize(w.grad, p=2, dim=0)
            w = (w + xi * g_norm).detach().requires_grad_(True)
            model.zero_grad()

        # Final perturbation projection
        return weights + epsilon * F.normalize(w - weights, p=2, dim=0)

# Spectral Encoder
class SpectralEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, pe_dim):
        super().__init__()
        self.cheb1 = ChebConv(in_dim + pe_dim, hid_dim, K=3)
        self.cheb2 = ChebConv(hid_dim, hid_dim, K=3)
        self.mu = nn.Linear(hid_dim, lat_dim)
        self.logvar = nn.Linear(hid_dim, lat_dim)

    def forward(self, x, edge_index, lap_pe):
        x = torch.cat([x, lap_pe], dim=1)
        x = F.relu(self.cheb1(x, edge_index))
        x = F.relu(self.cheb2(x, edge_index))
        x = x.mean(dim=0, keepdim=True)
        return self.mu(x), self.logvar(x)

# ScoreNet with classifier-free guidance
class ScoreNet(nn.Module):
    def __init__(self, lat_dim, cond_dim):
        super().__init__()
        self.cond_dim = cond_dim  # Store cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(lat_dim + cond_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, lat_dim)
        )
        self.cond_drop_prob = 0.1

    def forward(self, zt, cond=None):
        if cond is not None and torch.rand(1).item() < self.cond_drop_prob:
            cond = None
        
        if cond is None:
            # Create a zero tensor with shape (batch_size, cond_dim)
            cond = torch.zeros(zt.size(0), self.cond_dim, device=zt.device)
        else:
            # Expand cond to match the batch size of zt
            cond = cond.expand(zt.size(0), -1)  # Keeps the last dimension size
        
        zt = torch.cat([zt, cond], dim=1)
        return self.mlp(zt)

# Score-based SDE
class ScoreSDE(nn.Module):
    def __init__(self, score_model, T=1.0, N=1000):
        super().__init__()
        self.score_model = score_model
        self.T = T
        self.N = N
        self.timesteps = torch.linspace(T, 1e-3, N)

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(-2 * t))

    def sample(self, z_shape, cond=None):
        with torch.no_grad():
            z = torch.randn(z_shape).to(next(self.parameters()).device)
            for t in self.timesteps:
                t_tensor = torch.tensor([t], device=z.device)
                sigma = self.marginal_std(t_tensor)
                score = self.score_model(z, cond) / sigma
                z = z + score * t * 0.01 + torch.randn_like(z) * np.sqrt(0.01)
            return z

# Feature Decoder
class FeatureDecoder(nn.Module):
    def __init__(self, lat_dim, hid_dim, out_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(lat_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, z):
        return self.decoder(z)

# Trainer (main training and evaluation loop)
class Trainer:
    def __init__(self, in_dim, hid_dim, lat_dim, cond_dim, timesteps, lr=1e-5, warmup_steps=1000, total_steps=10000):
        self.encoder = SpectralEncoder(in_dim, hid_dim, lat_dim, pe_dim=10).to(device)
        self.denoiser = ScoreNet(lat_dim, cond_dim).to(device)
        self.decoder = FeatureDecoder(lat_dim, hid_dim, in_dim).to(device)
        self.diff = ScoreSDE(self.denoiser, T=1.0, N=timesteps).to(device)
        self.lap_pert = LaplacianPerturb()
        self.optim = torch.optim.Adam(list({id(p): p for p in (list(self.encoder.parameters()) + list(self.denoiser.parameters()) + list(self.decoder.parameters()))}.values()), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler()  # Add grad scaler

        # Learning rate scheduler with warmup and cosine decay
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=lambda step: min((step + 1) / warmup_steps, 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))) if step >= warmup_steps else (step + 1) / warmup_steps
        )

    def train_epoch(self, loader):
        self.encoder.train(); self.diff.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            lap_pe = data.lap_pe.to(device)  # Precomputed, GPU transfer
            self.optim.zero_grad()

            # Mixed precision context
            with torch.cuda.amp.autocast():
                # Forward pass
                mu, logvar = self.encoder(data.x, data.edge_index, lap_pe)
                z = mu + logvar.mul(0.5).exp() * torch.randn_like(mu)
                xt = self.diff.sample(z.shape, cond=mu)
                eps_pred = self.denoiser(xt, cond=mu)
                
                # Loss calculations
                loss_diff = F.mse_loss(eps_pred, torch.randn_like(mu))
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                x_rec = self.decoder(mu)
                loss_rec = F.mse_loss(x_rec, data.x.mean(dim=0, keepdim=True))
                loss = loss_diff*0.1 + kl * 0.5 + loss_rec * 1.0

            # Scaled backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)

            self.scaler.step(self.optim)
            self.scaler.update()
            
            total_loss += loss.item()

        # Update learning rate scheduler
        self.scheduler.step()
        return total_loss / len(loader)

    def evaluate(self, real_list, gen_list):
        def graph_structure_similarity(ei_r, ei_g):
            def to_edge_set(ei):
                rows, cols = ei.cpu().numpy()
                return set(tuple(sorted((int(u), int(v)))) for u, v in zip(rows, cols))
            S_r = to_edge_set(ei_r)
            S_g = to_edge_set(ei_g)
            union = S_r | S_g
            diff = S_r ^ S_g
            return 1.0 - (len(diff) / len(union)) if union else 1.0

        GSS, FMS, ARI = [], [], []
        for real, gen in zip(real_list, gen_list):
            GSS.append(graph_structure_similarity(real.edge_index, gen.edge_index))
            FMS.append(F.mse_loss(real.x.mean(0).to(device), gen.x.mean(0)).item())
            r_lbl = KMeans(n_clusters=5).fit_predict(real.x.cpu().detach().numpy())
            g_lbl = KMeans(n_clusters=5).fit_predict(gen.x.cpu().detach().numpy())
            ARI.append(adjusted_rand_score(r_lbl, g_lbl))
        
        return {
            'GSS': np.mean(GSS),
            'FeatMSE': np.mean(FMS),
            'ARI': np.mean(ARI),
        }

if __name__ == '__main__':
    # Load and prepare dataset
    dataset = MerfishCellGraphDataset('data/merfish_train.csv', k=7, root='data')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Instantiate trainer
    trainer = Trainer(
        in_dim=dataset[0].x.size(1),
        hid_dim=512,
        lat_dim=512,
        cond_dim=512,
        timesteps=500,
        lr=1e-5
    )

    # Training loop
    for epoch in range(1, 10001):
        loss = trainer.train_epoch(loader)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

    # Generation & Evaluation
    real_graphs = [dataset[i].to(device) for i in range(len(dataset))]
    gen_graphs = []
    for g in real_graphs:
        # Encode real graph to get conditioning mu
        with torch.no_grad():
            lap_pe = compute_lap_pe(g.edge_index, g.num_nodes, k=10).to(device)
            mu, _ = trainer.encoder(g.x, g.edge_index, lap_pe)
        
        # Generate latent with correct shape and conditioning
        z_shape = (g.num_nodes, trainer.encoder.mu.out_features)  # (num_nodes, lat_dim)
        z_gen = trainer.diff.sample(z_shape, cond=mu)
        
        # Decode to feature space
        x_gen = trainer.decoder(z_gen)
        gen_graphs.append(Data(x=x_gen, edge_index=g.edge_index))

    results = trainer.evaluate(real_graphs, gen_graphs)
    print('Evaluation metrics:', results)
    # Print shapes of real and generated graphs
    print(f"Number of real graphs: {len(real_graphs)}")
    print(f"Number of generated graphs: {len(gen_graphs)}")
    for i, (real, gen) in enumerate(zip(real_graphs, gen_graphs)):
        print(f"Graph {i}: Real x shape: {real.x.shape}, Gen x shape: {gen.x.shape}, Gen edge_index shape: {gen.edge_index.shape}")
    print("Training and evaluation completed.")