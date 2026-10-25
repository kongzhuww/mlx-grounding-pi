"""GroundingPI connector; see upstream license attribution in THIRD_PARTY_NOTICES.md."""

from mlx import nn

from .config import VisionConfig


class Projector(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.patch_dim = cfg.hidden_size
        self.merge_area = cfg.spatial_merge_size**2
        self.input_size = self.patch_dim * self.merge_area
        self.pre_norm = nn.LayerNorm(self.patch_dim, eps=cfg.projector_ln_eps)
        self.fc1 = nn.Linear(self.input_size, cfg.projector_hidden_size, bias=False)
        self.fc2 = nn.Linear(cfg.projector_hidden_size, cfg.out_hidden_size, bias=False)
        self.post_norm = nn.RMSNorm(cfg.out_hidden_size, eps=cfg.projector_ln_eps)

    def __call__(self, features):
        outputs = []
        for item in features:
            if item.ndim != 3 or item.shape[1:] != (self.merge_area, self.patch_dim):
                raise ValueError("expected merged [tokens, merge_area, hidden_size] features")
            x = self.pre_norm(item).reshape(item.shape[0], self.input_size)
            outputs.append(self.post_norm(self.fc2(nn.gelu(self.fc1(x)))))
        return outputs
