import torch
from torch_scatter import scatter_add
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader, InMemoryDataset, Data
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

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
        torch.save(self.collate([data]), self.processed_paths[0])


def compute_laplacian(edge_index, num_nodes, weights):
    if weights is None:
        weights = torch.ones(edge_index.size(1), device=edge_index.device)
    row, col = edge_index
    deg = scatter_add(weights, row, dim=0, dim_size=num_nodes)
    D_index = torch.stack([torch.arange(num_nodes, device=edge_index.device)] * 2)
    D_values = deg
    L_index = torch.cat([D_index, edge_index], dim=1)
    L_values = torch.cat([D_values, -weights])
    L = torch.sparse_coo_tensor(L_index, L_values, (num_nodes, num_nodes))
    return L.coalesce()

class LaplacianPerturb:
    def __init__(self, alpha_min=0.001, alpha_max=0.1):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def sample(self, edge_index, num_nodes):
        alpha = torch.rand(1, device=device) * (self.alpha_max - self.alpha_min) + self.alpha_min
        E = edge_index.size(1)
        signs = torch.randint(0, 2, (E,), device=device) * 2 - 1
        weights = 1.0 + alpha * signs.float()
        return compute_laplacian(edge_index, num_nodes, weights), weights

    def adversarial(self, model, x, edge_index, weights, xi=1e-6, epsilon=0.1, ip=1):
        w = weights.clone().detach().requires_grad_(True)
        for _ in range(ip):
            mu, _ = model(x, edge_index, w)
            loss = (mu**2).mean()
            loss.backward()
            g_norm = F.normalize(w.grad, p=2, dim=0)
            w = (w + xi * g_norm).detach().requires_grad_(True)
            model.zero_grad()
        return weights + epsilon * F.normalize(w - weights, p=2, dim=0)

class SpectralGNNEncoder(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.lin_mu = torch.nn.Linear(hid_dim, lat_dim)
        self.lin_logvar = torch.nn.Linear(hid_dim, lat_dim)

    def forward(self, x, edge_index, weights):
        h = F.relu(self.conv1(x, edge_index, weights))
        h = self.conv2(h, edge_index, weights)
        mu = self.lin_mu(h)
        logvar = self.lin_logvar(h)
        return mu, logvar

class Denoiser(torch.nn.Module):
    def __init__(self, lat_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(lat_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, lat_dim)
        )

    def forward(self, xt):
        return self.mlp(xt)

class Decoder(torch.nn.Module):
    def __init__(self, lat_dim, hid_dim, out_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(lat_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, out_dim)
        )

    def forward(self, z):
        return self.net(z)

class GraphDiffusion(torch.nn.Module):
    def __init__(self, denoiser, lat_dim, timesteps):
        super().__init__()
        self.denoiser = denoiser
        self.lat_dim = lat_dim
        self.T = timesteps
        betas = self.cosine_beta_schedule(self.T)
        self.register_buffer('betas', betas)
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)

    @staticmethod
    def cosine_beta_schedule(T, s=0.008):
        steps = T + 1
        x = torch.linspace(0, T, steps, device=device)
        cp = torch.cos(((x / T) + s) / (1 + s) * np.pi * 0.5)**2
        cp = cp / cp[0]
        betas = 1 - (cp[1:] / cp[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def q_sample(self, x0, t, noise):
        acp = self.alphas_cumprod[t].view(-1, 1)
        return acp.sqrt() * x0 + (1 - acp).sqrt() * noise

    @torch.no_grad()
    def p_sample(self, xt, t, edge_index, weights):
        eps = self.denoiser(xt)
        beta = self.betas[t]
        alpha = self.alphas[t]
        acp = self.alphas_cumprod[t]
        mean = (1 / alpha.sqrt()) * (xt - beta / (1 - acp).sqrt() * eps)
        if t > 0:
            return mean + beta.sqrt() * torch.randn_like(mean)
        return mean

    @torch.no_grad()
    def sample(self, num_nodes, edge_index, weights):
        torch.cuda.empty_cache()
        x = torch.randn(num_nodes, self.lat_dim, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(x, t, edge_index, weights)
        return x

class Trainer:
    def __init__(self, in_dim, hid_dim, lat_dim, timesteps, lr=1e-5):
        self.encoder = SpectralGNNEncoder(in_dim, hid_dim, lat_dim).to(device)
        self.denoiser = Denoiser(lat_dim).to(device)
        self.decoder = Decoder(lat_dim, hid_dim, in_dim).to(device)
        self.diff = GraphDiffusion(self.denoiser, lat_dim, timesteps).to(device)
        self.lap_pert = LaplacianPerturb()
        params = list(self.encoder.parameters()) + \
                 list(self.denoiser.parameters()) + \
                 list(self.diff.parameters()) + \
                 list(self.decoder.parameters())
        self.optim = torch.optim.Adam(params, lr=lr)
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, loader):
        self.encoder.train(); self.diff.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            L, weights = self.lap_pert.sample(data.edge_index, data.num_nodes)
            with torch.cuda.amp.autocast():
                mu, logvar = self.encoder(data.x, data.edge_index, weights)
                adv_w = self.lap_pert.adversarial(self.encoder, data.x, data.edge_index, weights)
                mu_adv, _ = self.encoder(data.x, data.edge_index, adv_w)
                # Diffusion forward
                eps = torch.randn_like(mu)
                t = torch.randint(0, self.diff.T, (1,), device=device).item()
                z = mu + logvar.mul(0.5).exp() * torch.randn_like(mu)
                xt = self.diff.q_sample(z, t, eps)
                eps_pred = self.diff.denoiser(xt)
                loss_diff = F.mse_loss(eps_pred, eps)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss_adv = F.mse_loss(mu_adv, mu.detach())
                # Reconstruction
                x_rec = self.decoder(mu)
                loss_rec = F.mse_loss(x_rec, data.x)
                loss = loss_diff + kl + loss_adv + loss_rec
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            total_loss += loss.item()
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

        def spectral_mse(ei_r, ei_g, num_nodes, k=10):
            rows_r, cols_r = ei_r.cpu().numpy()
            A_r = sp.coo_matrix((np.ones(len(rows_r)), (rows_r, cols_r)), shape=(num_nodes, num_nodes))
            A_r = (A_r + A_r.T).tocsr()
            rows_g, cols_g = ei_g.cpu().numpy()
            A_g = sp.coo_matrix((np.ones(len(rows_g)), (rows_g, cols_g)), shape=(num_nodes, num_nodes))
            A_g = (A_g + A_g.T).tocsr()
            k_eff = min(k, num_nodes - 1) if num_nodes > 1 else 1
            er, _ = eigsh(A_r, k=k_eff, which='LM')
            eg, _ = eigsh(A_g, k=k_eff, which='LM')
            return float(np.mean((np.sort(er) - np.sort(eg))**2))

        GSS, FMS, ARI, SMS = [], [], [], []
        for real, gen in zip(real_list, gen_list):
            GSS.append(graph_structure_similarity(real.edge_index, gen.edge_index))
            FMS.append(F.mse_loss(real.x.mean(0).to(device), gen.x.mean(0)).item())
            r_lbl = KMeans(n_clusters=5).fit_predict(real.x.cpu().detach().numpy())
            g_lbl = KMeans(n_clusters=5).fit_predict(gen.x.cpu().detach().numpy())
            ARI.append(adjusted_rand_score(r_lbl, g_lbl))
            SMS.append(spectral_mse(real.edge_index, gen.edge_index, real.num_nodes))
        return {
            'GSS': np.mean(GSS),
            'FeatMSE': np.mean(FMS),
            'ARI': np.mean(ARI),
            'SpecMSE': np.mean(SMS)
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
        timesteps=500,
        lr=1e-5
    )

    # Training loop
    for epoch in range(1, 5001):
        loss = trainer.train_epoch(loader)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

    # Generation & Evaluation
    real_graphs = [dataset[i] for i in range(len(dataset))]
    gen_graphs = []
    for g in real_graphs:
        num_nodes = g.x.size(0)
        weights = torch.ones(g.edge_index.size(1), device=device)
        z_gen = trainer.diff.sample(num_nodes, g.edge_index, weights)
        x_gen = trainer.decoder(z_gen)
        gen_graphs.append(Data(x=x_gen, edge_index=g.edge_index))

    results = trainer.evaluate(real_graphs, gen_graphs)
    print('Evaluation metrics:', results)
