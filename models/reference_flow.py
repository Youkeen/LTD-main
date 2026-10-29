"""Reference-supervised feature flow. Banks and hidden mixup are training-only."""
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FlowConfig:
    clip_name: str = 'ViT-L/14'
    feature_layer: int = 19  # legacy single-layer configuration; zero-based
    hidden_dim: int = 512
    steps: int = 2
    mix_prob: float = 0.5
    mix_alpha: float = 0.2
    flow_weight: float = 1.0
    mix_weight: float = 1.0
    consistency_weight: float = 0.1
    feature_layers: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.feature_layers is not None:
            self.feature_layers = tuple(self.feature_layers)
        layers = self.selected_layers
        if (not layers or any(type(layer) is not int or layer < 0 for layer in layers)
                or tuple(sorted(set(layers))) != layers):
            raise ValueError('Feature layers must be nonempty, unique, increasing zero-based integers.')

    @property
    def selected_layers(self):
        return self.feature_layers if self.feature_layers is not None else (self.feature_layer,)


def feature_bank_metadata(config):
    """Describe the frozen representation; retain compatibility with old single-layer banks."""
    layers = config.selected_layers
    if len(layers) == 1:
        return dict(clip_name=config.clip_name, feature_layer=layers[0])
    return dict(clip_name=config.clip_name, feature_layers=list(layers),
                feature_normalization='per_layer_layer_norm', feature_layout='concat')


class FrozenCLIP(nn.Module):
    def __init__(self, name, layer, state_dict=None, *, layers=None):
        super().__init__()
        self.feature_layers = FlowConfig(feature_layer=layer, feature_layers=layers).selected_layers
        from .clip import clip
        if state_dict is None:
            self.backbone, self.preprocess = clip.load(name, device='cpu')
        else:
            from .clip.model import build_model
            self.backbone = build_model(dict(state_dict)).float().eval()
            self.preprocess = clip._transform(self.backbone.visual.input_resolution)
        visual = self.backbone.visual
        if not hasattr(visual, 'transformer'):
            raise ValueError('Reference flow requires a CLIP ViT backbone.')
        if any(index >= len(visual.transformer.resblocks) for index in self.feature_layers):
            raise ValueError('Selected feature layer is outside the CLIP transformer.')
        self.layer_dim = visual.ln_post.normalized_shape[0]
        self.feature_dim = self.layer_dim * len(self.feature_layers)
        self._features = {}
        from functools import partial
        for index in self.feature_layers:
            visual.transformer.resblocks[index].register_forward_hook(partial(self._capture, index))
        self.requires_grad_(False)
        self.eval()

    def _capture(self, index, module, inputs, output):
        # CLIP residual blocks return [tokens, batch, channels]; select CLS.
        self._features[index] = output[0].detach().float()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        self._features.clear()
        try:
            semantic = F.normalize(self.backbone.encode_image(images).float(), dim=-1)
            # Normalize each layer independently, then concatenate in depth order.
            # No layer differences or extra positional embeddings are introduced.
            feature = torch.cat([F.layer_norm(self._features[index], (self.layer_dim,))
                                 for index in self.feature_layers], dim=-1)
        finally:
            self._features.clear()
        return feature, semantic


class VelocityField(nn.Module):
    def __init__(self, dim, hidden, probability, alpha):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(2 * dim + 1, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.probability, self.alpha = probability, alpha

    def forward(self, state, original, time, donor=None):
        h = self.hidden(torch.cat((state, original, time), dim=-1))
        if self.training and donor is not None and self.probability > 0:
            donor_state, donor_original = donor
            # Recompute donor hidden features with current weights, without gradients.
            with torch.no_grad():
                clean = self.hidden(torch.cat((donor_state, donor_original, time), dim=-1))
            gate = (torch.rand(h.shape[0], 1, device=h.device) < self.probability)
            alpha = torch.rand_like(h) * self.alpha * gate
            h = (1 - alpha) * h + alpha * clean
        return self.output(h)


class ReferenceFlow(nn.Module):
    def __init__(self, config=None, encoder=None, backbone_state=None):
        super().__init__()
        self.config = config or FlowConfig()
        c = self.config
        if c.steps < 1 or c.hidden_dim < 1 or not 0 <= c.mix_prob <= 1 or not 0 <= c.mix_alpha <= 1:
            raise ValueError('Invalid flow steps, hidden size or mixup parameters.')
        if min(c.flow_weight, c.mix_weight, c.consistency_weight) < 0:
            raise ValueError('Loss weights must be nonnegative.')
        self.encoder = encoder if encoder is not None else FrozenCLIP(
            c.clip_name, c.feature_layer, backbone_state, layers=c.feature_layers)
        self.encoder.requires_grad_(False)
        dim = self.encoder.feature_dim
        self.velocity = VelocityField(dim, c.hidden_dim, c.mix_prob, c.mix_alpha)
        # Only predicted velocities and net correction, no direct original-feature shortcut.
        self.classifier = nn.Sequential(nn.LayerNorm((c.steps + 1) * dim),
                                        nn.Linear((c.steps + 1) * dim, 1))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def extract(self, images):
        return self.encoder(images)

    def rollout(self, original, donor=None):
        state, velocities = original, []
        donor_state = donor
        for step in range(self.config.steps):
            time = original.new_full((len(original), 1), 1 - step / self.config.steps)
            pair = None if donor is None else (donor_state, donor)
            velocity = self.velocity(state, original, time, pair)
            velocities.append(velocity)
            # z(t)=(1-t)*reference+t*input: integrate from t=1 down to 0.
            state = state - velocity / self.config.steps
            if donor is not None:
                with torch.no_grad():
                    donor_state = donor_state - self.velocity(donor_state, donor, time) / self.config.steps
        representation = torch.cat(velocities + [original - state], dim=-1)
        return self.classifier(representation).flatten()

    def forward(self, images):
        features, _ = self.extract(images)
        return self.rollout(features).unsqueeze(1)

    def training_losses(self, original, anchor, donor, donor_anchor, labels):
        time = torch.rand(len(original), 1, device=original.device)
        state = (1 - time) * anchor + time * original
        target = original - anchor
        logits = self.rollout(original)
        classification = F.binary_cross_entropy_with_logits(logits, labels.float())
        flow = F.mse_loss(self.velocity(state, original, time), target)
        mixed_flow = mixed_cls = consistency = original.new_zeros(())
        c = self.config
        if donor is not None and donor_anchor is not None and c.mix_weight > 0 and c.mix_prob > 0 and c.mix_alpha > 0:
            donor_state = (1 - time) * donor_anchor + time * donor
            mixed_flow = F.mse_loss(self.velocity(state, original, time, (donor_state, donor)), target)
            mixed_logits = self.rollout(original, donor)
            mixed_cls = F.binary_cross_entropy_with_logits(mixed_logits, labels.float())
            consistency = F.mse_loss(mixed_logits.sigmoid(), logits.detach().sigmoid())
        total = classification + c.flow_weight * flow + c.mix_weight * (
            mixed_cls + c.flow_weight * mixed_flow + c.consistency_weight * consistency)
        return total, logits, dict(classification=classification, flow=flow,
                                  mixed_classification=mixed_cls, mixed_flow=mixed_flow,
                                  consistency=consistency)

    def checkpoint_config(self):
        return asdict(self.config)


class FrozenFeatureLinear(nn.Module):
    """Linear probe of the exact normalized intermediate feature used by flow."""
    def __init__(self, config=None, encoder=None, backbone_state=None):
        super().__init__()
        self.config = config or FlowConfig()
        self.encoder = encoder if encoder is not None else FrozenCLIP(
            self.config.clip_name, self.config.feature_layer, backbone_state,
            layers=self.config.feature_layers)
        self.encoder.requires_grad_(False)
        self.classifier = nn.Linear(self.encoder.feature_dim, 1)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            features, _ = self.encoder(images)
        return self.classifier(features)

    def checkpoint_config(self):
        return asdict(self.config)


def load_flow_checkpoint(checkpoint):
    """Restore entirely from the saved checkpoint, without downloading CLIP."""
    prefix = 'encoder.backbone.'
    backbone = {k[len(prefix):]: v for k, v in checkpoint['model'].items() if k.startswith(prefix)}
    model_class = FrozenFeatureLinear if checkpoint.get('method') == 'clip_linear' else ReferenceFlow
    model = model_class(FlowConfig(**checkpoint['flow_config']), backbone_state=backbone)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model
