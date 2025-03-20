import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (GATv2Conv,
                                SAGPooling, 
                                LayerNorm, 
                                global_mean_pool, 
                                global_add_pool
)

from layers import CoAttentionLayer, RESCAL


class SSI_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.kge_dim = kge_dim
        self.rel_total = rel_total
        
        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = ModuleList()
        self.net_norms = ModuleList()
        
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = SSI_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads
        
        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples):
        h_data, t_data, rels = triples
        
        h_data.x = self.initial_norm(h_data.x)
        t_data.x = self.initial_norm(t_data.x)

        repr_h, repr_t = [], []
        
        for i, block in enumerate(self.blocks):
            h_data, r_h = block(h_data)
            t_data, r_t = block(t_data)
            
            repr_h.append(r_h)
            repr_t.append(r_t)
            
            h_data.x = F.gelu(self.net_norms[i](h_data.x))
            t_data.x = F.gelu(self.net_norms[i](t_data.x))
        
        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        attentions = self.co_attention(repr_h, repr_t)
        scores = self.KGE(repr_h, repr_t, rels, attentions)
        
        return scores


class SSI_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.conv = GATv2Conv(in_features, head_out_feats, n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
    
    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        pooled_x, att_edge_index, _, att_batch, _, _ = self.readout(data.x, data.edge_index, batch=data.batch)
        
        global_emb = global_add_pool(pooled_x, att_batch) + global_mean_pool(pooled_x, att_batch)
        
        return data, global_emb