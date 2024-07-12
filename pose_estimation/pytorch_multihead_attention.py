import torch

class PyTorchMultiHeadAttentionWrapper(torch.nn.Module):
    def __init__(self, ray_fea_size, img_fea_size, embed_dim, num_heads=1):
        super().__init__()

        self.attention = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            kdim=ray_fea_size,
            vdim=ray_fea_size,
            batch_first=True,
        )

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_proj = torch.nn.Linear(img_fea_size, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)

    def forward(self, img_features, ray_features, mask=None):
        query = self.q_proj(img_features)  # [Head, SeqLen, Dims]

        _, attn_output_weights = self.attention(
            query,
            ray_features,
            torch.broadcast_to(
                torch.tensor(1.0, dtype=query.dtype, device=query.device),
                ray_features.shape,
            ),
        )

        return attn_output_weights
