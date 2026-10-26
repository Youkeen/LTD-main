import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model


def create_model(opt):
    """Use the same frozen encoder configuration for preparation and training."""
    config = None
    if opt.method in ('reference_flow', 'clip_linear'):
        if not opt.arch.startswith('CLIP:ViT-'):
            raise ValueError('reference_flow and clip_linear require a CLIP ViT architecture')
        config = dict(clip_name=opt.arch[5:], feature_layer=opt.feature_layer,
                      hidden_dim=opt.flow_hidden, steps=opt.flow_steps,
                      mix_prob=opt.mix_prob, mix_alpha=opt.mix_alpha,
                      flow_weight=opt.flow_weight, mix_weight=opt.mix_weight,
                      consistency_weight=opt.consistency_weight)
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
        self.bank = None
        self.use_interference = False
        # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if opt.fix_backbone or opt.method in ('reference_flow', 'clip_linear'):
            params = []
            for _, p in self.model.named_parameters():
                if p.requires_grad: 
                    params.append(p) 
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()

        

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.model.to(self.device)
        


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
            if self.bank is None or self.paths is None:
                raise RuntimeError('Initialize the training feature bank and supply sample paths first.')
            original, semantic = self.model.extract(self.input)
            anchor, _ = self.bank.retrieve(semantic, self.paths, 0, self.opt.anchor_topk,
                                          duplicate_threshold=self.opt.duplicate_threshold)
            donor = donor_anchor = None
            if self.use_interference:
                donor, ids = self.bank.retrieve(semantic, self.paths, 1, self.opt.donor_topk,
                                               random_choice=True, duplicate_threshold=self.opt.duplicate_threshold)
                donor_anchor, _ = self.bank.retrieve(self.bank.semantics[ids].to(self.device),
                    [self.bank.paths[i] for i in ids.tolist()], 0, self.opt.anchor_topk,
                    duplicate_threshold=self.opt.duplicate_threshold)
            self.loss, logits, self.losses = self.model.training_losses(
                original, anchor, donor, donor_anchor, self.label)
            self.output = logits.unsqueeze(1)
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

    def initialize_bank(self, dataset):
        if self.opt.method != 'reference_flow':
            return
        from networks.feature_bank import FeatureBank, path_id
        if not self.opt.real_bank_path:
            raise ValueError('--real_bank_path is required; run prepare_feature_banks.py before training.')
        expected = dict(clip_name=self.model.config.clip_name, feature_layer=self.model.config.feature_layer)
        training_paths = {path_id(p): dataset.labels_dict[p] for p in dataset.total_list}
        banks = []
        for label, path in ((0, self.opt.real_bank_path), (1, self.opt.fake_bank_path)):
            if not path:
                continue
            bank = FeatureBank.load(path)
            if bank.metadata != expected:
                raise ValueError('Bank backbone/layer does not match this model.')
            if not (bank.labels == label).all():
                raise ValueError(f'Bank at {path} must contain only label {label} samples.')
            if bank.features.shape[1] != self.model.encoder.feature_dim:
                raise ValueError('Bank feature dimensions do not match this model.')
            if any(p not in training_paths or training_paths[p] != int(y)
                   for p, y in zip(bank.paths, bank.labels)):
                raise ValueError('Bank contains samples outside the current training split or wrong labels.')
            banks.append(bank)
        if len(banks) == 2 and banks[0].semantics.shape[1] != banks[1].semantics.shape[1]:
            raise ValueError('Real and fake bank semantic dimensions do not match.')
        # Merge in memory to preserve cross-class hash exclusion during retrieval.
        self.bank = banks[0] if len(banks) == 1 else FeatureBank(
            torch.cat([b.features for b in banks]), torch.cat([b.semantics for b in banks]),
            torch.cat([b.labels for b in banks]), sum([b.paths for b in banks], []),
            sum([b.hashes for b in banks], []), expected)
        c = self.model.config
        self.use_interference = bool(self.opt.fake_bank_path) and c.mix_weight > 0 and c.mix_prob > 0 and c.mix_alpha > 0
        print(f'Loaded prepared feature banks. Interference: {"enabled" if self.use_interference else "disabled"}.')

