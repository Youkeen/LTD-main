import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        config = None
        if opt.method == 'reference_flow':
            if not opt.arch.startswith('CLIP:ViT-'):
                raise ValueError('reference_flow requires a CLIP ViT architecture')
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
        self.model = get_model(opt.arch, 1, opt.select_k, True, flow_config=config, backbone_state=backbone)
        self.bank = None
        # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if opt.fix_backbone or opt.method == 'reference_flow':
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
        from networks.feature_bank import FeatureBank, build_bank, path_id
        import os
        if self.opt.bank_size < 2:
            raise ValueError('bank_size must be at least 2 per class.')
        if self.opt.bank_path:
            self.bank = FeatureBank.load(self.opt.bank_path)
            expected = dict(clip_name=self.model.config.clip_name, feature_layer=self.model.config.feature_layer)
            if self.bank.metadata != expected:
                raise ValueError('Bank backbone/layer does not match this model.')
            training_paths = {path_id(p): dataset.labels_dict[p] for p in dataset.total_list}
            if any(p not in training_paths or training_paths[p] != int(y)
                   for p, y in zip(self.bank.paths, self.bank.labels)):
                raise ValueError('Bank contains samples outside the current training split or wrong labels.')
        else:
            self.bank = build_bank(self.model, dataset, self.opt)
            os.makedirs(self.save_dir, exist_ok=True)
            self.bank.save(os.path.join(self.save_dir, 'training_bank.pt'))

