"""CPU-resident training banks, chunked semantic retrieval, no validation data."""
import hashlib
import os
from copy import deepcopy

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


def path_id(path):
    return os.path.normcase(os.path.realpath(path))


class FeatureBank:
    def __init__(self, features, semantics, labels, paths, hashes, metadata=None):
        self.features = features.float().cpu()
        self.semantics = F.normalize(semantics.float().cpu(), dim=-1)
        self.labels = labels.long().cpu()
        self.paths, self.hashes = list(paths), list(hashes)
        self.metadata = metadata or {}
        n = len(self.labels)
        if (self.features.ndim != 2 or self.semantics.ndim != 2 or
                len(self.features) != n or len(self.semantics) != n or
                len(paths) != n or len(hashes) != n):
            raise ValueError('Bank tensor and path dimensions do not match.')
        if not torch.isfinite(self.features).all() or not torch.isfinite(self.semantics).all():
            raise ValueError('Bank features must be finite.')
        if not ((self.labels == 0) | (self.labels == 1)).all():
            raise ValueError('Bank labels must be real=0 or fake=1.')
        self.lookup = {p: i for i, p in enumerate(paths)}
        if (self.labels == 0).sum() < 2 or (self.labels == 1).sum() < 2:
            raise ValueError('Feature bank needs at least two real and two fake samples.')

    @torch.no_grad()
    def retrieve(self, queries, paths, label, topk=4, random_choice=False,
                 duplicate_threshold=0.9999, chunk_size=2048):
        if topk < 1 or chunk_size < 1:
            raise ValueError('topk and chunk_size must be positive.')
        queries = F.normalize(queries.float(), dim=-1)
        device = queries.device
        indices = torch.where(self.labels == label)[0]
        k = min(topk, len(indices))
        scores = queries.new_full((len(queries), k), -torch.inf)
        selected = torch.full((len(queries), k), -1, dtype=torch.long, device=device)
        query_hashes = [self.hashes[self.lookup[p]] if p in self.lookup else None for p in paths]
        for start in range(0, len(indices), chunk_size):
            ids = indices[start:start + chunk_size]
            sims = queries @ self.semantics[ids].to(device).T
            # Exclude same file, byte-identical copies, and near-identical semantics.
            excluded = [[self.paths[i] == p or (qh is not None and self.hashes[i] == qh)
                         for i in ids.tolist()] for p, qh in zip(paths, query_hashes)]
            sims.masked_fill_(torch.tensor(excluded, device=device), -torch.inf)
            sims.masked_fill_(sims >= duplicate_threshold, -torch.inf)
            merged = torch.cat((scores, sims), dim=1)
            merged_ids = torch.cat((selected, ids.to(device).expand(len(queries), -1)), dim=1)
            scores, positions = merged.topk(k, dim=1)
            selected = merged_ids.gather(1, positions)
        valid = scores.isfinite()
        selected = selected.masked_fill(~valid, -1)
        if not valid.any(dim=1).all():
            raise ValueError('No eligible bank neighbor; add diverse training samples or adjust duplicate threshold.')
        safe_ids = selected.clamp_min(0).cpu()
        if random_choice:
            rank = torch.multinomial(valid.float(), 1)
            ids = selected.gather(1, rank).flatten().cpu()
            return self.features[ids].to(device), ids
        weights = (scores / 0.07).softmax(dim=1).cpu()
        reference = (self.features[safe_ids] * weights.unsqueeze(-1)).sum(dim=1)
        return reference.to(device), selected.cpu()

    def save(self, path):
        torch.save(vars(self) | {'lookup': None}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        data.pop('lookup', None)
        return cls(**data)


def build_bank(model, dataset, opt):
    # Use official deterministic CLIP preprocessing for stable bank references.
    clean = deepcopy(dataset)
    clean.transform = model.encoder.preprocess
    clean.return_path = True
    generator = torch.Generator().manual_seed(opt.bank_seed)
    indices = []
    for label in (0, 1):
        candidates = [i for i, p in enumerate(clean.total_list) if clean.labels_dict[p] == label]
        order = torch.randperm(len(candidates), generator=generator).tolist()
        indices.extend(candidates[i] for i in order[:opt.bank_size])
    loader = DataLoader(Subset(clean, indices), batch_size=opt.batch_size,
                        num_workers=opt.num_threads, shuffle=False)
    features, semantics, labels, paths, hashes = [], [], [], [], []
    device = next(model.parameters()).device
    for step, (images, target, batch_paths) in enumerate(loader):
        f, s = model.extract(images.to(device))
        features.append(f.cpu())
        semantics.append(s.cpu())
        labels.append(target)
        for p in batch_paths:
            paths.append(path_id(p))
            with open(p, 'rb') as handle:
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
                hashes.append(digest.hexdigest())
        print(f'\rBuilding training feature bank: {step + 1}/{len(loader)}', end='', flush=True)
    print()
    if not features:
        raise ValueError('Training dataset is empty.')
    metadata = dict(clip_name=model.config.clip_name, feature_layer=model.config.feature_layer)
    return FeatureBank(torch.cat(features), torch.cat(semantics), torch.cat(labels), paths, hashes, metadata)
