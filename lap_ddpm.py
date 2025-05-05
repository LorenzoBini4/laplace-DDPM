import torch
from torch_scatter import scatter_add
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader, InMemoryDataset, Data
from torch_geometric.utils import to_dense_adj
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MerfishCellGraphDataset(InMemoryDataset):
    def __init__(self, csv_path, k=5, root='data'):
        self.csv_path = csv_path
        self.k = k
        super().__init__(root)
        df = pd.read_csv(self.csv_path)
        print("Columns in CSV:", df.columns)
        print("Shape of CSV:", df.shape)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return ['data.pt']
    def download(self): pass
    def process(self):
        df = pd.read_csv(self.csv_path)
        print("Columns in CSV:", df.columns)
        # extract coordinates (for graph construction)
        coords = df[['coord_X', 'coord_Y']].values
        # get the gene expression columns only (excluding Unnamed: 0)
        gene_cols = df.columns[
            df.columns.get_loc('1700022I11Rik') : df.columns.get_loc('Gad1') + 1
        ]
        feats = df[gene_cols].values
        # build k-NN graph (excluding self-loop)
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(coords)
        _, idx = nbrs.kneighbors(coords)
        edges = [[i, j] for i, nbr in enumerate(idx) for j in nbr[1:]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # PyG Data object
        data = Data(
            x=torch.tensor(feats, dtype=torch.float),
            edge_index=edge_index,
            pos=torch.tensor(coords, dtype=torch.float)
        )
        torch.save(self.collate([data]), self.processed_paths[0])

def compute_laplacian(edge_index, num_nodes, weights):
    """
    Compute Laplacian in sparse format (no .to_dense())
    """
    # Edge weights default to 1 if None
    if weights is None:
        weights = torch.ones(edge_index.size(1), device=edge_index.device)

    # Degree: scatter sum over rows
    row, col = edge_index
    deg = scatter_add(weights, row, dim=0, dim_size=num_nodes)

    # Create sparse diagonal Degree matrix D
    D_index = torch.stack([torch.arange(num_nodes, device=edge_index.device)] * 2)
    D_values = deg

    # Laplacian: L = D - A
    # D is diagonal, A is sparse adjacency
    L_index = torch.cat([D_index, edge_index], dim=1)
    L_values = torch.cat([D_values, -weights])

    L = torch.sparse_coo_tensor(L_index, L_values, (num_nodes, num_nodes))
    return L.coalesce()  # Keep sparse!

class LaplacianPerturb:
    def __init__(self, alpha_min=0.01, alpha_max=0.1):
        self.alpha_min, self.alpha_max = alpha_min, alpha_max
    def sample(self, edge_index, num_nodes):
        alpha = torch.rand(1, device=device)*(self.alpha_max-self.alpha_min)+self.alpha_min
        E = edge_index.size(1)
        signs = torch.randint(0,2,(E,),device=device)*2-1
        weights = 1.0 + alpha*signs.float()
        return compute_laplacian(edge_index, num_nodes, weights), weights

    def adversarial(self, model, x, edge_index, weights, xi=1e-6, epsilon=0.1, ip=1):
        # Make a copy of weights that requires grad
        w = weights.clone().detach().requires_grad_(True)

        for _ in range(ip):
            # Directly call your encoder with sparse weights
            mu, _ = model(x, edge_index, w)

            # Adversarial objective: push mu away from zero (or any proxy)
            loss = (mu**2).mean()
            loss.backward()

            # Compute the gradient direction in weight‐space
            g = w.grad
            g_norm = F.normalize(g, p=2, dim=0)

            # Small ascent step
            w = (w + xi * g_norm).detach().requires_grad_(True)

            # Zero grads for next iteration
            model.zero_grad()
        # Return the perturbed weights within your epsilon budget
        return weights + epsilon * F.normalize(w - weights, p=2, dim=0)

class SpectralGNNEncoder(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.lin_mu = torch.nn.Linear(hid_dim, lat_dim)
        self.lin_logvar = torch.nn.Linear(hid_dim, lat_dim)
    def forward(self, x, edge_index, weights):
        # apply weighted adjacency via weights
        h = F.relu(self.conv1(x, edge_index, weights))
        h = self.conv2(h, edge_index, weights)
        g = h.mean(0, keepdim=True)
        mu = self.lin_mu(g)
        logvar = self.lin_logvar(g)
        return mu, logvar

class GraphDiffusion(torch.nn.Module):
    def __init__(self, denoiser, lat_dim, timesteps=200):
        super().__init__()
        self.denoiser = denoiser
        self.T = timesteps
        betas = self.cosine_beta_schedule(self.T)
        self.register_buffer('betas', betas)
        self.alphas = 1-betas
        self.alphas_cumprod = torch.cumprod(self.alphas,0)

    @staticmethod
    def cosine_beta_schedule(T, s=0.008):
        steps = T+1
        x = torch.linspace(0,T,steps,device=device)
        cp = torch.cos(((x/T)+s)/(1+s)*np.pi*0.5)**2
        cp = cp/cp[0]
        betas = 1-(cp[1:]/cp[:-1])
        return torch.clamp(betas,0.0001,0.9999)

    def q_sample(self, x0, t, noise):
        acp = self.alphas_cumprod[t].view(-1,1)
        return acp.sqrt()*x0 + (1-acp).sqrt()*noise

    def p_sample(self, xt, t, edge_index, weights):
        eps = self.denoiser(xt, edge_index, weights)
        beta, alpha = self.betas[t], self.alphas[t]
        acp = self.alphas_cumprod[t]
        mean = (1/alpha.sqrt())*(xt - beta/(1-acp).sqrt()*eps)
        if t>0:
            return mean + beta.sqrt()*torch.randn_like(mean)
        return mean

    def sample(self, z, edge_index, weights):
        x = torch.randn_like(z)
        for t in reversed(range(self.T)):
            x = self.p_sample(x, t, edge_index, weights)
        return x

class Trainer:
    def __init__(self, in_dim, hid_dim, lat_dim, timesteps):
        self.encoder = SpectralGNNEncoder(in_dim, hid_dim, lat_dim).to(device)
        self.diff = GraphDiffusion(self.encoder, lat_dim, timesteps).to(device)
        self.lap_pert = LaplacianPerturb()
        self.optim = torch.optim.Adam(list(self.encoder.parameters())+list(self.diff.parameters()), lr=1e-3)

    def train_epoch(self, loader):
        self.encoder.train(); self.diff.train()
        total=0
        for data in loader:
            data = data.to(device)
            # standard laplacian
            L, weights = self.lap_pert.sample(data.edge_index, data.num_nodes)
            mu, logvar = self.encoder(data.x, data.edge_index, weights)
            # adversarial laplacian
            adv_weights = self.lap_pert.adversarial(self.encoder, data.x, data.edge_index, weights)
            mu_adv, _ = self.encoder(data.x, data.edge_index, adv_weights)
            # diffusion
            eps = torch.randn_like(mu)
            t = torch.randint(0,self.diff.T,(1,),device=device)
            z = mu + logvar.mul(0.5).exp()*torch.randn_like(mu)
            xt = self.diff.q_sample(z, t, eps)
            eps_pred = self.diff.denoiser(xt, data.edge_index, weights)
            loss_diff = F.mse_loss(eps_pred, eps)
            kl = -0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
            loss_adv = F.mse_loss(mu_adv, mu.detach())
            loss = loss_diff + kl*1e-3 + loss_adv*0.5
            self.optim.zero_grad(); loss.backward(); self.optim.step()
            total+=loss.item()
        return total/len(loader)

    def evaluate(self, real_list, gen_list):
        metrics={'GSS':[],'FeatMSE':[],'ARI':[],'SpecMSE':[]}
        for real, gen in zip(real_list, gen_list):
            A_r=to_dense_adj(real.edge_index)[0]; A_g=to_dense_adj(gen.edge_index)[0]
            metrics['GSS'].append(1-(A_r-A_g).abs().mean().item())
            metrics['FeatMSE'].append(F.mse_loss(real.x.mean(0),gen.x.mean(0)).item())
            rlbl=KMeans(5).fit_predict(real.x.cpu()); glbl=KMeans(5).fit_predict(gen.x.cpu())
            metrics['ARI'].append(adjusted_rand_score(rlbl,glbl))
            er=torch.linalg.eigvalsh(A_r); eg=torch.linalg.eigvalsh(A_g)
            metrics['SpecMSE'].append(F.mse_loss(er,eg).item())
        return {k:np.mean(v) for k,v in metrics.items()}

# TRAINING
dataset = MerfishCellGraphDataset('data/merfish_train.csv', k=5, root='data')
loader = DataLoader(dataset,batch_size=16,shuffle=True)
trainer=Trainer(in_dim=dataset.num_node_features,hid_dim=128,lat_dim=64,timesteps=200)
for epoch in range(500): print('Epoch',epoch,trainer.train_epoch(loader))

# EVALUATION
real_graphs=[dataset[i] for i in range(len(dataset))]
gen=[trainer.diff.sample(trainer.diff.alphas_cumprod, g.edge_index, torch.ones(g.edge_index.size(1),device=device)) for g in real_graphs]
print(trainer.evaluate(real_graphs,gen))