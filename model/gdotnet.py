import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.init as init
import sys
sys.path.insert(0, 'rational_kat_cu')
from functools import partial
from timm.models.layers import to_2tuple
from kat_rational import KAT_Group
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

    
    

class KAN_rational(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            out_features=None,
            act_layer=KAT_Group,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            act_init="gelu",
            device=None
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        out_features = out_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, out_features, bias=bias[0])
        self.act1 = KAT_Group(mode="identity", device=device)
        self.drop1 = nn.Dropout(drop_probs[0])

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        return x
    


class FinalGraphDiffusionTransformer(nn.Module):
    def __init__(self, num_nodes=264, num_classes=2, trans_layers=1, heads=2, T=3, embed_dim = 1024, alpha = 0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.T = T
        self.step_transformer = Transformer(dim=num_nodes, depth=1, heads=3, dim_head=64, dropout=0., ista=0.1)

        # Class embedding
        self.class_emb = nn.Embedding(num_classes, num_nodes*num_nodes)
        self.encoder =nn.Linear(in_features=num_nodes*num_nodes, out_features=embed_dim)
        self.final_transformer = Transformer(dim=embed_dim, depth=1, heads=3, dim_head=64, dropout=0., ista=0.1)
        self.decoder = nn.Linear(in_features=embed_dim, out_features=num_nodes*num_nodes)
        self.alpha = alpha
        self.A = None



    
    def get_S(self, A):
        D = A.sum(dim=-1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        D_mat_inv_sqrt = torch.diag_embed(D_inv_sqrt)
        self.A = A.clone()
        return torch.bmm(torch.bmm(D_mat_inv_sqrt, A), D_mat_inv_sqrt)
    


    def single_view_diffusion(self, S, A):
        return self.alpha*torch.bmm(torch.bmm(S, A), S.transpose(1, 2)) +(1-self.alpha)*self.A  

    def forward(self, A, class_idx):
        B, N, _ = A.shape
        A_history = []
        S = self.get_S(A)
        if class_idx==None:
            class_scores = torch.matmul(A.view(B,-1), self.class_emb.weight.T)  # [B, num_classes]
            prob = F.softmax(class_scores, dim=-1)
            class_emb = torch.matmul(prob, self.class_emb.weight)  # [B, d]
        else:
            class_emb = self.class_emb(class_idx)
        class_emb = class_emb.view(B,1,-1)
        for _ in range(self.T):
            A = self.single_view_diffusion(S, A)               # Step 1: Diffusion
            A = self.step_transformer(A)                    # Step 2: Transformer
            A = 0.5 * (A + A.transpose(1, 2))               # Step 3: Symmetrize
            A_history.append(A)
        A_stack = torch.stack(A_history, dim=1)
        A_flat = A_stack.view(B, -1, N*N)  # [B, T, N*N]
        class_emb = class_emb.view(B,1, N*N)
        fusion_input = torch.cat([class_emb, A_flat], dim=1)

        fusion_output = self.decoder(self.final_transformer(self.encoder(fusion_input))[:,-1]).view(B,N,N) 
        A_final = 0.5 * (fusion_output + fusion_output.transpose(1, 2))

        return self.single_view_diffusion(S, A_final)  # [B, N, N]
    
def Sinkhorn_log_exp_sum(C, mu, nu, epsilon):
    
    def _log_boltzmann_kernel(u, v, epsilon, C=None):
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= epsilon
        return kernel
  
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    thresh = 1e-6
    max_iter = 100
            
    for i in range(max_iter):
       
        u0 = u  # useful to check the update
        K = _log_boltzmann_kernel(u, v, epsilon, C)
        u_ = torch.log(mu + 1e-8) - torch.logsumexp(K, dim=2)
        u = epsilon * u_ + u
        
        K_t = _log_boltzmann_kernel(u, v, epsilon, C).permute(0, 2, 1).contiguous()
        v_ = torch.log(nu + 1e-8) - torch.logsumexp(K_t, dim=2)
        v = epsilon * v_ + v
        
        err = (u - u0).abs().mean()
        if err.item() < thresh:
            break
    
    K = _log_boltzmann_kernel(u, v, epsilon, C)
    T = torch.exp(K)

    return T


class OTAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 0.05

    def forward(self, q,k,V):
        B, N, D = q.shape
        # Normalize Q, K
        Q = F.normalize(q, dim=-1)
        K = F.normalize(k, dim=-1)

        # Cosine distance
        sim = torch.einsum("bnd,bmd->bnm", Q, K)  # (B, H, N, N)
        wdist = 1. - sim

        # Uniform marginals
        mu = torch.full((B, N), 1./N, device=q.device)
        nu = torch.full((B, N), 1./N, device=k.device)

        # Flatten batch and head dims for OT
        C = wdist.view(B, N, N)
        T = Sinkhorn_log_exp_sum(C, mu, nu, self.eps)  # (B*H, N, N)
        T = T.view(B, N, N)

        # Weighted sum of V
        out = torch.bmm(T,V) +V  # (B, N, N)
        return out
    

# The transformer components is implemented by https://github.com/Ma-Lab-Berkeley/CRATE
'''
@article{yu2024white,
  title={White-Box Transformers via Sparse Rate Reduction},
  author={Yu, Yaodong and Buchanan, Sam and Pai, Druv and Chu, Tianzhe and Wu, Ziyang and Tong, Shengbang and Haeffele, Benjamin and Ma, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
'''

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., step_size=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        # compute D^T * D * x
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., ista=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, dim, dropout=dropout, step_size=ista))
                    ]
                )
            )

    def forward(self, x):
        depth = 0
        for attn, ff in self.layers:
            grad_x = attn(x) + x

            x = ff(grad_x)
        return x

    
class Transformer_edge(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., ista=0.1, dim_edge = 264):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        self.edge_layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, dim, dropout=dropout, step_size=ista))
                    ]
                )
            )
        for _ in range(depth):
            self.edge_layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim_edge, Attention(dim_edge, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim_edge, FeedForward(dim_edge, dim_edge, dropout=dropout, step_size=ista))
                    ]
                )
            )
        self.OT = OTAttention()
        

    def forward(self, adj, x):
        for i in range(len(self.layers)):
            attn_edge, ff_edge = self.edge_layers[i]
            grad_adj = attn_edge(adj) + adj
            adj = ff_edge(grad_adj)
            attn, ff = self.layers[i]
            x = attn(x)
            x_fuse = torch.bmm(adj, x)
            grad_x = x_fuse + x
            x = ff(grad_x)
            adj_x = torch.bmm(x, x.transpose(1,2))
            x = self.OT(adj_x,adj,x)
        return x


class CRATE_edgev3(nn.Module):
    def __init__(
            self, *,  num_classes=2, dim=2640, depth=2, heads=3, dim_head=512,
            dropout=0.2, emb_dropout=0., ista=0.1,T = 4, alpha = 0.3
            ):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_edge(dim, depth, heads, dim_head, dropout, ista=ista)

        self.lin_in = KAN_rational(in_features=264,out_features=10)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, num_classes)
        )
        self.aggregator = KAN_rational(in_features=264,out_features=1)
        self.graphar = FinalGraphDiffusionTransformer(T=T)
    def forward(self, batch_graph, X_concat,class_idx):
        batch,_,_ = X_concat.size()
        x = self.lin_in(X_concat)
        adj_mask = (batch_graph != 0).float()
        broadcasted_features = x.unsqueeze(2).expand(-1, -1, 264, -1)
        # 使用邻接矩阵掩码作为掩码，只保留邻居节点的特征
        x = broadcasted_features * adj_mask.unsqueeze(-1)
        x = x.view(batch, 264, -1)
        batch_graph = self.graphar(batch_graph,class_idx)
        x = rearrange(self.transformer(batch_graph,x), ' b h (w d) -> b (h w) d',w=10,h=264)
        

        x = self.aggregator(x).view(batch, 10, -1) # kan readout
        x = torch.mean(x, dim = 2) 
        return self.mlp_head(x),_

    





