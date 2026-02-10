import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import traceback
from torch_scatter import scatter_add

# Helper function for Laplacian PE
def compute_lap_pe(edge_index, num_nodes, k=10):
    """
    Computes Laplacian Positional Encoding for a graph.
    Args:
        edge_index (torch.Tensor): Graph connectivity in COO format.
        num_nodes (int): Number of nodes in the graph.
        k (int): Number of eigenvectors to compute for the PE.
    Returns:
        torch.Tensor: Laplacian PE of shape (num_nodes, k).
    """
    target_device = edge_index.device if edge_index.numel() > 0 else torch.device('cpu')
    if num_nodes == 0: return torch.zeros((0, k), device=target_device, dtype=torch.float)
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    edge_index_np = edge_index.cpu().numpy()
    if edge_index_np.max() >= num_nodes or edge_index_np.min() < 0:
        print(f"Warning: Edge index out of bounds ({edge_index_np.min()},{edge_index_np.max()}). Num nodes: {num_nodes}. Clamping.")
        edge_index_np = np.clip(edge_index_np, 0, num_nodes - 1)
        valid_edges_mask = edge_index_np[0] != edge_index_np[1]
        edge_index_np = edge_index_np[:, valid_edges_mask]
        if edge_index_np.size == 0:
            print("Warning: All edges removed after clamping. Returning zero PEs.")
            return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    data_np = np.ones(edge_index_np.shape[1])
    try:
        row, col = edge_index_np
        adj = sp.coo_matrix((data_np, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    except Exception as e:
        print(f"Error creating sparse adj matrix for PE: {e}"); return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    adj = adj + adj.T
    adj.data[adj.data > 1] = 1
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    deg_inv_sqrt_mat = sp.diags(deg_inv_sqrt)
    L = sp.eye(num_nodes, dtype=np.float32) - deg_inv_sqrt_mat @ adj @ deg_inv_sqrt_mat

    num_eigenvectors_to_compute = min(k + 1, num_nodes)
    if num_eigenvectors_to_compute <= 1:
        return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    try:
        eigvals, eigvecs = eigsh(L, k=num_eigenvectors_to_compute, which='SM', tol=1e-4,
                                ncv=min(num_nodes, max(2 * num_eigenvectors_to_compute + 1, 20)))
        sorted_indices = np.argsort(eigvals)
        eigvecs = eigvecs[:, sorted_indices]
    except Exception as e:
        print(f"Eigenvalue computation failed for PE ({num_nodes} nodes, k={num_eigenvectors_to_compute}): {e}. Returning zero PEs.");
        traceback.print_exc()
        return torch.zeros((num_nodes, k), device=target_device, dtype=torch.float)

    start_idx = 1 if eigvecs.shape[1] > 1 else 0
    actual_k_to_use = min(k, eigvecs.shape[1] - start_idx)

    if actual_k_to_use <= 0:
        pe = torch.zeros((num_nodes, k), dtype=torch.float)
    else:
        pe = torch.from_numpy(eigvecs[:, start_idx : start_idx + actual_k_to_use]).float()
        if pe.shape[1] < k:
            padding = torch.zeros((num_nodes, k - pe.shape[1]), dtype=torch.float)
            pe = torch.cat((pe, padding), dim=1)
    return pe.to(target_device)

def compute_lap_pe_batch(edge_index, num_nodes, k=10):
    """
    Computes Laplacian Positional Encoding for a batch of graphs.
    Args:
        edge_index (torch.Tensor): Graph connectivity in COO format.
        num_nodes (int): Number of nodes in the graph.
        k (int): Number of eigenvectors to compute for the PE.
    Returns:
        torch.Tensor: Laplacian PE of shape (num_nodes, k).
    """
    if edge_index.numel() == 0 or num_nodes == 0:
        return torch.zeros((num_nodes, k), dtype=torch.float, device=edge_index.device)

    return compute_lap_pe(edge_index, num_nodes, k)

def compute_lap_pe_batch_list(edge_index_list, num_nodes_list, k=10):
    """
    Computes Laplacian Positional Encoding for a list of graphs.
    Args:
        edge_index_list (list of torch.Tensor): List of graph connectivity in COO format.
        num_nodes_list (list of int): List of number of nodes in each graph.
        k (int): Number of eigenvectors to compute for the PE.
    Returns:
        torch.Tensor: Laplacian PE of shape (sum(num_nodes_list), k).
    """
    if not edge_index_list or not num_nodes_list or len(edge_index_list) != len(num_nodes_list):
        return torch.zeros((0, k), dtype=torch.float, device=edge_index_list[0].device)

    pe_list = [compute_lap_pe_batch(edge_index, num_nodes, k) for edge_index, num_nodes in zip(edge_index_list, num_nodes_list)]
    return torch.cat(pe_list, dim=0)    

# ---------------------------------------------
# Graph utility for spectral alignment losses
# ---------------------------------------------
def laplacian_smooth(node_features, edge_index, edge_weight=None, eps: float = 1e-6):
    """
    One-step symmetric Laplacian smoothing used to align representations across graphs.
    Args:
        node_features (Tensor): [N, D]
        edge_index (LongTensor): [2, E]
        edge_weight (Tensor or None): [E]
        eps (float): numerical stability
    Returns:
        Tensor: Smoothed features with same shape as node_features.
    """
    if node_features.numel() == 0 or edge_index.numel() == 0:
        return torch.zeros_like(node_features)

    num_nodes = node_features.size(0)
    device = node_features.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device, dtype=node_features.dtype)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    messages = node_features[col] * norm.unsqueeze(-1)
    smoothed = scatter_add(messages, row, dim=0, dim_size=num_nodes)
    return smoothed
