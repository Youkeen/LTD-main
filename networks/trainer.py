import torch
import torch.nn as nn
from networks.base_model import BaseModel
from models import get_model


def create_model(opt):
    """Use the same frozen encoder configuration for preparation and training."""
    config = None
    if opt.method in ('reference_flow', 'clip_linear'):
        if not opt.arch.startswith('CLIP:ViT-'):
            raise ValueError('reference_flow and clip_linear require a CLIP ViT architecture')
        config = dict(clip_name=opt.arch[5:], feature_layer=opt.feature_layer,
                      feature_layers=getattr(opt, 'feature_layers', None),
                      hidden_dim=opt.flow_hidden, steps=opt.flow_steps,
                      mix_prob=opt.mix_prob, mix_alpha=opt.mix_alpha,
                      flow_weight=opt.flow_weight, mix_weight=opt.mix_weight,
                      consistency_weight=opt.consistency_weight,
                      start_time_prob=getattr(opt, 'start_time_prob', 0.1))
    backbone = None
    if config is not None and opt.clip_checkpoint:
        saved = torch.load(opt.clip_checkpoint, map_location='cpu', weights_only=True)['model']
        backbone = {k[len('model.'):]: v for k, v in saved.items() if k.startswith('model.')}
        if not backbone:
            raise ValueError('clip_checkpoint must be an original LTD checkpoint containing model.* CLIP weights.')
    if opt.method == 'clip_linear':
        from models.reference_flow import FrozenFeatureLinear, FlowConfig
        return FrozenFeatureLinear(FlowConfig(**config), backbone_state=backbone)
    return get_model(opt.arch, 1, opt.select_k, True, flow_config=config, backbone_state=backbone)


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        if opt.method == 'reference_flow' and not opt.real_bank_path:
            raise ValueError('--real_bank_path is required; run prepare_feature_banks.py before training.')
        self.model = create_model(opt)
        if opt.method in ('reference_flow', 'clip_linear'):
            print(f'Frozen CLIP layers (zero-based): {self.model.config.selected_layers}; '
                  f'joint feature dimension: {self.model.encoder.feature_dim}')
        self.bank = None
        self.training_stage = 'flow' if opt.method == 'reference_flow' else 'classifier'
        self.stage_epoch = 0
        self.model.to(self.device)
        self._make_optimizer()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _make_optimizer(self):
        opt = self.opt
        params = [p for p in self.model.parameters() if p.requires_grad]
        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

    def set_stage(self, stage):
        if self.opt.method != 'reference_flow':
            raise ValueError('Explicit stages are only used by reference_flow.')
        self.model.set_stage(stage)
        self.training_stage, self.stage_epoch = stage, 0
        if stage == 'classifier':
            self.bank = None  # Neither classifier training nor inference uses retrieval.
        self._make_optimizer()  # No velocity parameters/state in the classifier optimizer.

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True


    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()
        self.paths = input[2] if len(input) > 2 else None


    def forward(self):
        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)


    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        if self.opt.method == 'reference_flow':
            self.loss, logits = self.reference_losses(self.input, self.paths, self.label)
            self.output = None if logits is None else logits.unsqueeze(1)
            self.optimizer.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.opt.grad_clip)
            self.optimizer.step()
            return
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def reference_losses(self, images, paths=None, labels=None):
        """One FM query per batch in stage 1; bank-free predicted rollout in stage 2."""
        original, semantic = self.model.extract(images)
        if self.training_stage == 'flow':
            if self.bank is None or paths is None:
                raise RuntimeError('Initialize the real training bank and supply sample paths first.')
            anchor, _ = self.bank.retrieve(semantic, paths, 0, self.opt.anchor_topk,
                                          random_choice=True,
                                          duplicate_threshold=self.opt.duplicate_threshold)
            flow = self.model.flow_loss(original, anchor)
            self.losses = dict(flow=flow)
            return self.model.config.flow_weight * flow, None
        classification, logits = self.model.classification_loss(original, labels)
        self.losses = dict(classification=classification)
        return classification, logits

    def initialize_bank(self, dataset):
        if self.opt.method != 'reference_flow':
            return
        from networks.feature_bank import FeatureBank, path_id
        from models.reference_flow import feature_bank_metadata
        if not self.opt.real_bank_path:
            raise ValueError('--real_bank_path is required; run prepare_feature_banks.py before training.')
        expected = feature_bank_metadata(self.model.config)
        training_paths = {path_id(p): dataset.labels_dict[p] for p in dataset.total_list}
        if self.opt.fake_bank_path:
            raise ValueError('Two-stage reference_flow uses only --real_bank_path; remove --fake_bank_path.')
        bank = FeatureBank.load(self.opt.real_bank_path)
        if bank.metadata != expected:
            raise ValueError(f'Bank representation does not match this model: '
                             f'expected {expected}, got {bank.metadata}. '
                             'Rebuild banks using the same --feature_layers/--feature_layer settings.')
        if not (bank.labels == 0).all():
            raise ValueError('The real reference bank must contain only label 0 samples.')
        if bank.features.shape[1] != self.model.encoder.feature_dim:
            raise ValueError('Bank feature dimensions do not match this model.')
        if any(p not in training_paths or training_paths[p] != 0 for p in bank.paths):
            raise ValueError('Bank contains samples outside the current training split or wrong labels.')
        self.bank = bank
        print(f'Loaded {len(bank.paths)} real training references; sampling one from semantic Top-{self.opt.anchor_topk}.')

