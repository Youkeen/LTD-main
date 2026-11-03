"""Two-stage reference flow: fit v(z,t), freeze it, then fit a velocity readout."""
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FlowConfig:
    clip_name: str = 'ViT-L/14'
    feature_layer: Optional[int] = None  # explicit single-layer override; zero-based
    hidden_dim: int = 512
    steps: int = 4
    mix_prob: float = 0.0  # legacy checkpoint fields; disabled in version 2
    mix_alpha: float = 0.2
    flow_weight: float = 1.0
    mix_weight: float = 0.0
    consistency_weight: float = 0.0
    feature_layers: Optional[Tuple[int, ...]] = None
    start_time_prob: float = 0.1
    architecture_version: int = 2

    def __post_init__(self):
        if self.feature_layers is not None:
            self.feature_layers = tuple(self.feature_layers)
        layers = self.selected_layers
        if (not layers or any(type(layer) is not int or layer < 0 for layer in layers)
                or tuple(sorted(set(layers))) != layers):
            raise ValueError('Feature layers must be nonempty, unique, increasing zero-based integers.')
        if self.architecture_version not in (1, 2):
            raise ValueError('Unsupported reference flow architecture version.')
        if not 0 <= self.start_time_prob <= 1:
            raise ValueError('start_time_prob must be in [0, 1].')

    @property
    def selected_layers(self):
        if self.feature_layers is not None:
            return self.feature_layers
        return (self.feature_layer,) if self.feature_layer is not None else (11, 12, 13, 14, 15)


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
    def __init__(self, dim, hidden):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(dim + 1, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, state, time):
        return self.output(self.hidden(torch.cat((state, time), dim=-1)))


class ReferenceFlow(nn.Module):
    def __init__(self, config=None, encoder=None, backbone_state=None):
        super().__init__()
        self.config = config or FlowConfig()
        c = self.config
        if c.architecture_version != 2:
            raise ValueError('Use load_flow_checkpoint() to restore an old conditional flow.')
        if c.steps < 1 or c.hidden_dim < 1 or c.flow_weight <= 0:
            raise ValueError('Flow steps, hidden size and flow_weight must be positive.')
        if c.mix_prob != 0 or c.mix_weight != 0 or c.consistency_weight != 0:
            raise ValueError('Two-stage reference flow does not use hidden mixup or consistency losses.')
        self.encoder = encoder if encoder is not None else FrozenCLIP(
            c.clip_name, c.feature_layer, backbone_state, layers=c.feature_layers)
        self.encoder.requires_grad_(False)
        dim = self.encoder.feature_dim
        self.velocity = VelocityField(dim, c.hidden_dim)
        # Full velocity vectors only; total correction is already their scaled sum.
        # Keep magnitudes available to the readout instead of normalizing each sample.
        self.classifier = nn.Linear(c.steps * dim, 1)
        self.set_stage('flow')

    def set_stage(self, stage):
        if stage not in ('flow', 'classifier'):
            raise ValueError('Stage must be flow or classifier.')
        self.training_stage = stage
        self.encoder.requires_grad_(False)
        self.velocity.requires_grad_(stage == 'flow')
        self.classifier.requires_grad_(stage == 'classifier')
        self.zero_grad(set_to_none=True)
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.velocity.train(mode and self.training_stage == 'flow')
        self.classifier.train(mode and self.training_stage == 'classifier')
        return self

    @torch.no_grad()
    def extract(self, images):
        return self.encoder(images)

    def integrate(self, original):
        """Actual predicted reverse Euler path, with no reference/bank inputs."""
        state, velocities = original, []
        for step in range(self.config.steps):
            time = original.new_full((len(original), 1), 1 - step / self.config.steps)
            velocity = self.velocity(state, time)
            velocities.append(velocity)
            # z(t)=(1-t)*reference+t*input: integrate from t=1 down to 0.
            state = state - velocity / self.config.steps
        return torch.stack(velocities, dim=1), state

    def rollout_features(self, original):
        velocities, _ = self.integrate(original)
        return velocities.flatten(1)

    def rollout(self, original):
        return self.classifier(self.rollout_features(original)).flatten()

    def forward(self, images):
        features, _ = self.extract(images)
        return self.rollout(features).unsqueeze(1)

    def flow_loss(self, original, anchor, time=None):
        """Stage 1: labels and classifier are deliberately absent from this API."""
        if self.training_stage != 'flow':
            raise RuntimeError('Flow training requires stage=flow.')
        original, anchor = original.detach(), anchor.detach()
        if time is None:
            time = torch.rand(len(original), 1, device=original.device, dtype=original.dtype)
            # The exact t=1 start used by inference receives explicit supervision.
            at_start = torch.rand_like(time) < self.config.start_time_prob
            time = torch.where(at_start, torch.ones_like(time), time)
        state = (1 - time) * anchor + time * original
        target = original - anchor
        return F.mse_loss(self.velocity(state, time), target)

    def classification_loss(self, original, labels):
        """Stage 2: predicted trajectories are detached; only the readout learns."""
        if self.training_stage != 'classifier':
            raise RuntimeError('Classification training requires stage=classifier.')
        with torch.no_grad():
            representation = self.rollout_features(original)
        logits = self.classifier(representation).flatten()
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return loss, logits

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
    config = dict(checkpoint['flow_config'])
    config.setdefault('architecture_version', 1)
    if checkpoint.get('method') == 'clip_linear':
        model_class = FrozenFeatureLinear
    elif config['architecture_version'] == 1:
        from .reference_flow_legacy import LegacyReferenceFlow
        model_class = LegacyReferenceFlow
    else:
        model_class = ReferenceFlow
    model = model_class(FlowConfig(**config), backbone_state=backbone)
    model.load_state_dict(checkpoint['model'], strict=True)
    if isinstance(model, ReferenceFlow):
        model.set_stage(checkpoint.get('training_stage', 'classifier'))
    return model
