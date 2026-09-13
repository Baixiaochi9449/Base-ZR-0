"""Dense flow prediction from only the trailing Difference Query states."""

import torch
from torch import nn
from torch.nn import functional as F


class FlowDecoderLayer(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, grid, queries):
        grid = self.norm(grid + self.attention(grid, queries, queries, need_weights=False)[0])
        return self.output_norm(grid + self.mlp(grid))


class DenseRegressionFlowHead(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config.validate()
        hidden = config.flow_head_hidden_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden)
        self.grid = nn.Parameter(torch.empty(1, config.flow_grid_size ** 2, hidden))
        nn.init.normal_(self.grid, std=0.02)
        self.layers = nn.ModuleList([
            FlowDecoderLayer(hidden, config.flow_head_num_heads)
            for _ in range(config.flow_head_num_layers)
        ])
        self.refinement = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1),
                                        nn.GroupNorm(8, hidden), nn.GELU())
        self.output = nn.Conv2d(hidden, 2, 3, padding=1)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, query_states):
        if query_states.ndim != 3 or query_states.shape[1] < self.config.num_flow_queries:
            raise ValueError("flow head requires [B,Nq,H] with Nq >= num_flow_queries")
        queries = query_states[:, -self.config.num_flow_queries:, :]
        # A single differentiable boundary cast supports mixed backbone/head dtypes
        # without autocast. Parameters retain their construction or explicit .to dtype.
        queries = queries.to(dtype=self.input_projection.weight.dtype)
        queries = self.input_projection(self.input_norm(queries))
        grid = self.grid.expand(queries.shape[0], -1, -1)
        for layer in self.layers:
            grid = layer(grid, queries)
        grid = grid.transpose(1, 2).reshape(queries.shape[0], -1, self.config.flow_grid_size, self.config.flow_grid_size)
        grid = F.interpolate(grid, size=(56, 56), mode="bilinear", align_corners=False)
        return self.output(self.refinement(grid))

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "submodules": {name: sum(p.numel() for p in module.parameters())
                               for name, module in self.named_children()},
                "grid": self.grid.numel()}


class LatentFlowDecoderLayer(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio):
        super().__init__()
        self.q_norm = nn.LayerNorm(hidden)
        self.kv_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, mlp_ratio * hidden), nn.GELU(),
                                 nn.Linear(mlp_ratio * hidden, hidden))

    def forward(self, spatial, memory):
        q = self.q_norm(spatial)
        kv = self.kv_norm(memory)
        spatial = spatial + self.attention(q, kv, kv, need_weights=False)[0]
        return spatial + self.mlp(self.mlp_norm(spatial))


class WanVAELatentFlowHead(nn.Module):
    def __init__(self, input_dim, config, latent_shape):
        super().__init__()
        self.config = config.validate()
        self.latent_shape = tuple(int(v) for v in latent_shape)
        channels, height, width = self.latent_shape
        hidden = config.flow_v2_hidden_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden)
        self.spatial_tokens = nn.Parameter(torch.empty(1, height * width, hidden))
        nn.init.normal_(self.spatial_tokens, std=0.02)
        self.layers = nn.ModuleList([LatentFlowDecoderLayer(hidden, config.flow_v2_num_heads,
                                                             config.flow_v2_mlp_ratio)
                                     for _ in range(config.flow_v2_num_layers)])
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, channels)

    def forward(self, query_states):
        if query_states.ndim != 3 or query_states.shape[1] < self.config.num_flow_queries:
            raise ValueError("flow head requires [B,Nq,H] with Nq >= num_flow_queries")
        queries = query_states[:, -self.config.num_flow_queries:, :]
        queries = queries.to(dtype=self.input_projection.weight.dtype)
        memory = self.input_projection(self.input_norm(queries))
        spatial = self.spatial_tokens.expand(queries.shape[0], -1, -1)
        for layer in self.layers:
            spatial = layer(spatial, memory)
        output = self.output(self.output_norm(spatial))
        return output.transpose(1, 2).reshape(query_states.shape[0], *self.latent_shape)

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "latent_shape": self.latent_shape}


OPTICAL_FLOW_AUX_REGISTRY = {"none": None, "dense_regression_v1": DenseRegressionFlowHead,
                             "wan_vae_latent_v2": WanVAELatentFlowHead}


def build_optical_flow_head(input_dim, config, latent_shape=None):
    config.validate()
    factory = OPTICAL_FLOW_AUX_REGISTRY[config.optical_flow_aux_type]
    if factory is None:
        return None
    if config.optical_flow_aux_type == "wan_vae_latent_v2" and latent_shape is None:
        raise ValueError("V2 head construction requires the probed latent shape")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.flow_init_seed)
        if config.optical_flow_aux_type == "wan_vae_latent_v2":
            return factory(input_dim, config, latent_shape)
        return factory(input_dim, config)
