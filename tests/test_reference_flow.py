"""CPU tests; no pretrained weights, datasets or downloads required."""
import os
import tempfile
import unittest
import importlib.util
import sys
import types
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from models.reference_flow import ReferenceFlow, FlowConfig, VelocityField, load_flow_checkpoint
from networks.feature_bank import FeatureBank, build_bank, path_id
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.bank_options import BankOptions
from prepare_feature_banks import prepare_feature_banks


def identity(x):
    return x


class TinyEncoder(nn.Module):
    feature_dim = 8
    preprocess = staticmethod(identity)

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x), F.normalize(x, dim=-1)


class ToyDataset(Dataset):
    def __init__(self, root):
        self.total_list = [os.path.join(root, f'{i}.bin') for i in range(12)]
        self.labels_dict = {p: i % 2 for i, p in enumerate(self.total_list)}
        for i, p in enumerate(self.total_list):
            with open(p, 'wb') as handle:
                handle.write(bytes([i]))
        self.images = torch.randn(12, 8)
        self.transform, self.return_path = identity, True

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, index):
        p = self.total_list[index]
        return self.transform(self.images[index]), self.labels_dict[p], path_id(p)


class FlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def model(self, **kwargs):
        return ReferenceFlow(FlowConfig(hidden_dim=16, **kwargs), encoder=TinyEncoder())

    def test_gradients_and_inference_without_bank(self):
        model = self.model(mix_prob=1.0)
        features, _ = model.extract(torch.randn(4, 8))
        donor = torch.randn(4, 8, requires_grad=True)
        loss, logits, parts = model.training_losses(features, torch.randn(4, 8),
            donor, torch.randn(4, 8), torch.tensor([0., 1., 0., 1.]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(parts['mixed_flow'].item(), 0)
        self.assertIsNone(donor.grad)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.velocity.parameters()))
        model.eval()
        x = torch.randn(3, 8)
        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
        model.train()
        self.assertFalse(model.encoder.training)

    def test_reverse_time_integration(self):
        model = self.model()
        class ConstantVelocity(nn.Module):
            def forward(self, state, original, time, donor=None):
                return torch.ones_like(state) * 3
        model.velocity = ConstantVelocity()
        model.classifier = nn.Identity()
        representation = model.rollout(torch.randn(2, 8)).view(2, 24)
        torch.testing.assert_close(representation[:, -8:], torch.ones(2, 8) * 3)

    def test_mixup_only_in_training_hidden_layer(self):
        field = VelocityField(8, 16, 1.0, 0.9)
        x, other, t = torch.randn(4, 8), torch.randn(4, 8), torch.rand(4, 1)
        field.eval()
        torch.testing.assert_close(field(x, x, t), field(x, x, t, (other, other)))
        field.train()
        self.assertFalse(torch.equal(field(x, x, t), field(x, x, t, (other, other))))

    def test_mixup_disabled_ablation(self):
        model = self.model(mix_weight=0)
        x = torch.randn(4, 8)
        loss, _, parts = model.training_losses(x, x + 1, x, x, torch.zeros(4))
        torch.testing.assert_close(loss, parts['classification'] + parts['flow'])
        self.assertEqual(parts['mixed_flow'].item(), 0)

    def test_no_donor_disables_all_interference_losses(self):
        model = self.model(mix_prob=1.0)
        x = torch.randn(4, 8)
        loss, _, parts = model.training_losses(x, x + 1, None, None, torch.zeros(4))
        torch.testing.assert_close(loss, parts['classification'] + parts['flow'])
        for name in ('mixed_flow', 'mixed_classification', 'consistency'):
            self.assertEqual(parts[name].item(), 0)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.velocity.parameters()))

    def test_chunked_retrieval_self_duplicates_and_donor_class(self):
        semantics = F.normalize(torch.randn(12, 8), dim=-1)
        semantics[2] = semantics[0]
        bank = FeatureBank(torch.randn(12, 8), semantics, torch.arange(12) % 2,
                           [str(i) for i in range(12)], [str(i) for i in range(12)])
        a, ids = bank.retrieve(semantics[:2], ['0', '1'], 0, topk=10, chunk_size=2)
        b, _ = bank.retrieve(semantics[:2], ['0', '1'], 0, topk=10, chunk_size=100)
        torch.testing.assert_close(a, b)
        # Invalid padding is -1; neither the query nor its identical embedding can be selected.
        self.assertNotIn(0, ids[0].tolist())
        self.assertNotIn(2, ids[0].tolist())
        _, fake_ids = bank.retrieve(semantics[:2], ['0', '1'], 1, random_choice=True)
        self.assertTrue((bank.labels[fake_ids] == 1).all())
        self.assertNotEqual(fake_ids[1].item(), 1)

    def test_empty_neighbor_rejected(self):
        bank = FeatureBank(torch.randn(4, 8), torch.ones(4, 8), torch.tensor([0, 0, 1, 1]),
                           list('abcd'), list('abcd'))
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            bank.retrieve(torch.ones(1, 8), ['a'], 0)

    def test_bank_trainer_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            opt = SimpleNamespace(method='reference_flow', arch='CLIP:ViT-L/14', select_k=5,
                clip_checkpoint=None, feature_layer=19, flow_hidden=16, flow_steps=2,
                mix_prob=0.5, mix_alpha=0.2, flow_weight=1., mix_weight=1., consistency_weight=0.1,
                fix_backbone=True, optim='adam', lr=0.001, beta1=0.9, weight_decay=0.,
                gpu_ids=[], checkpoints_dir=root, name='test', bank_size=6, bank_seed=2,
                real_bank_path=os.path.join(root, 'real.pt'), fake_bank_path=os.path.join(root, 'fake.pt'),
                batch_size=4, num_threads=0, anchor_topk=2, donor_topk=3,
                duplicate_threshold=0.9999, grad_clip=1.)
            with patch('networks.trainer.get_model', return_value=self.model()):
                trainer = Trainer(opt)
            dataset = ToyDataset(root)
            real_bank = build_bank(trainer.model, dataset, opt, labels=(0,))
            fake_bank = build_bank(trainer.model, dataset, opt, labels=(1,))
            real_bank.save(opt.real_bank_path)
            fake_bank.save(opt.fake_bank_path)
            # Loading for training must never fall back to extraction.
            with patch('networks.feature_bank.build_bank', side_effect=AssertionError('No training-time build')):
                trainer.initialize_bank(dataset)
            self.assertTrue(trainer.use_interference)
            self.assertEqual(len(trainer.bank.paths), 12)
            trainer.set_input(next(iter(DataLoader(dataset, batch_size=4))))
            before = trainer.model.classifier[1].weight.detach().clone()
            trainer.optimize_parameters()
            self.assertGreater(trainer.losses['mixed_flow'].item(), 0)
            self.assertFalse(torch.equal(before, trainer.model.classifier[1].weight))
            trainer.save_networks('test.pt')
            saved = torch.load(os.path.join(root, 'test', 'test.pt'), weights_only=True)
            self.assertIn('flow_config', saved)
            self.assertFalse(any('bank' in k for k in saved['model']))
            clone = self.model()
            clone.load_state_dict(saved['model'])
            trainer.model.eval()
            clone.eval()
            torch.testing.assert_close(trainer.model(dataset.images), clone(dataset.images))
            trainer.initialize_bank(dataset)
            full_paths = dataset.total_list[:]
            dataset.total_list = dataset.total_list[:-1]
            with self.assertRaisesRegex(ValueError, 'outside'):
                trainer.initialize_bank(dataset)
            dataset.total_list = full_paths

            # Real-only training performs only anchor retrieval and still updates weights.
            opt.fake_bank_path = None
            trainer.initialize_bank(dataset)
            self.assertFalse(trainer.use_interference)
            self.assertTrue((trainer.bank.labels == 0).all())
            trainer.model.train()
            before = trainer.model.classifier[1].weight.detach().clone()
            with patch.object(trainer.bank, 'retrieve', wraps=trainer.bank.retrieve) as retrieve:
                trainer.optimize_parameters()
                self.assertEqual(retrieve.call_count, 1)
                self.assertEqual(retrieve.call_args.args[2], 0)
            self.assertFalse(torch.equal(before, trainer.model.classifier[1].weight))
            for name in ('mixed_flow', 'mixed_classification', 'consistency'):
                self.assertEqual(trainer.losses[name].item(), 0)
            self.assertFalse(os.path.exists(os.path.join(root, 'test', 'training_bank.pt')))

            opt.fake_bank_path = os.path.join(root, 'missing.pt')
            with self.assertRaises(FileNotFoundError):
                trainer.initialize_bank(dataset)
            opt.fake_bank_path = opt.real_bank_path
            with self.assertRaisesRegex(ValueError, 'only label 1'):
                trainer.initialize_bank(dataset)
            opt.fake_bank_path = None
            real_bank.metadata['feature_layer'] = 0
            real_bank.save(opt.real_bank_path)
            with self.assertRaisesRegex(ValueError, 'backbone/layer'):
                trainer.initialize_bank(dataset)
            opt.real_bank_path = None
            with self.assertRaisesRegex(ValueError, 'real_bank_path'):
                trainer.initialize_bank(dataset)
            with patch('networks.trainer.get_model') as get_model:
                with self.assertRaisesRegex(ValueError, 'real_bank_path'):
                    Trainer(opt)
                get_model.assert_not_called()

    def test_single_class_bank_preparation_and_reuse(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = ToyDataset(root)
            opt = SimpleNamespace(bank_size=3, bank_seed=2, batch_size=4, num_threads=0)
            model = self.model()
            bank = build_bank(model, dataset, opt, labels=(0,))
            self.assertEqual(len(bank.paths), 3)
            self.assertTrue((bank.labels == 0).all())
            # File enumeration order must not change the seeded sample selection.
            dataset.total_list.reverse()
            repeated = build_bank(model, dataset, opt, labels=(0,))
            self.assertEqual(bank.paths, repeated.paths)
            filename = os.path.join(root, 'real.pt')
            bank.save(filename)
            restored = FeatureBank.load(filename)
            torch.testing.assert_close(bank.features, restored.features)
            with self.assertRaisesRegex(ValueError, 'no samples for label 1'):
                restored.retrieve(torch.randn(1, 8), ['other'], 1)
            dataset.data_label = 'val'
            with self.assertRaisesRegex(ValueError, 'training split'):
                build_bank(model, dataset, opt, labels=(0,))

    def test_preparation_entry_point(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = ToyDataset(root)
            dataset_module = types.ModuleType('data.datasets')
            dataset_module.RealFakeDataset = lambda opt: dataset
            opt = SimpleNamespace(method='reference_flow', data_label='train', bank_size=3,
                bank_seed=2, batch_size=4, num_threads=0, gpu_ids=[],
                real_bank_output=os.path.join(root, 'banks', 'real.pt'), fake_bank_output=None)
            with patch.dict(sys.modules, {'data.datasets': dataset_module}), patch(
                    'prepare_feature_banks.create_model', return_value=self.model()):
                prepare_feature_banks(opt)
                self.assertEqual(os.listdir(os.path.dirname(opt.real_bank_output)), ['real.pt'])
                self.assertTrue((FeatureBank.load(opt.real_bank_output).labels == 0).all())
                opt.fake_bank_output = os.path.join(root, 'banks', 'fake.pt')
                prepare_feature_banks(opt)
                self.assertTrue((FeatureBank.load(opt.fake_bank_output).labels == 1).all())
                opt.fake_bank_output = opt.real_bank_output
                with self.assertRaisesRegex(ValueError, 'different output paths'):
                    prepare_feature_banks(opt)

    def test_bank_command_line_contract(self):
        with patch.object(sys, 'argv', ['train.py', '--real_bank_path', 'real.pt']):
            opt = TrainOptions().gather_options()
            self.assertEqual(opt.real_bank_path, 'real.pt')
            self.assertIsNone(opt.fake_bank_path)
        with patch.object(sys, 'argv', ['train.py', '--real_bank_path', 'real.pt', '--fake_bank_path', 'fake.pt']):
            self.assertEqual(TrainOptions().gather_options().fake_bank_path, 'fake.pt')
        with patch.object(sys, 'argv', ['train.py', '--method', 'ltd']):
            self.assertIsNone(TrainOptions().gather_options().real_bank_path)
        for args in ([], ['--fake_bank_path', 'fake.pt']):
            with patch.object(sys, 'argv', ['train.py'] + args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    TrainOptions().gather_options()
        with patch.object(sys, 'argv', ['prepare_feature_banks.py', '--real_bank_output', 'real.pt']):
            opt = BankOptions().gather_options()
            self.assertEqual(opt.real_bank_output, 'real.pt')
            self.assertIsNone(opt.fake_bank_output)
        for args in (['--data_label', 'val'], ['--method', 'ltd'], ['--bank_size', '1']):
            with patch.object(sys, 'argv', ['prepare_feature_banks.py', '--real_bank_output', 'real.pt'] + args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    BankOptions().gather_options()

    def test_actual_clip_hook_and_self_contained_checkpoint(self):
        # Load the repository's actual CLIP architecture without optional tokenizer dependencies.
        filename = Path(__file__).resolve().parents[1] / 'models' / 'clip' / 'model.py'
        spec = importlib.util.spec_from_file_location('test_clip_architecture', filename)
        architecture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(architecture)
        backbone = architecture.CLIP(embed_dim=8, image_resolution=16, vision_layers=2,
            vision_width=64, vision_patch_size=8, context_length=4, vocab_size=16,
            transformer_width=64, transformer_heads=1, transformer_layers=1).float()
        # Match clip.load's half-weight conversion followed by CPU float conversion.
        architecture.convert_weights(backbone)
        backbone.float()
        loader = types.ModuleType('models.clip.clip')
        loader.load = lambda *a, **k: (backbone, identity)
        loader._transform = lambda resolution: identity
        package = types.ModuleType('models.clip')
        package.clip = loader
        with patch.dict(sys.modules, {'models.clip': package, 'models.clip.model': architecture}):
            model = ReferenceFlow(FlowConfig(feature_layer=1, hidden_dim=16))
            model.eval()
            images = torch.randn(2, 3, 16, 16)
            features, semantics = model.extract(images)
            self.assertEqual(features.shape, (2, 64))
            self.assertEqual(semantics.shape, (2, 8))
            expected = model(images)
            loader.load = lambda *a, **k: self.fail('Checkpoint loading must not download CLIP')
            restored = load_flow_checkpoint(dict(model=model.state_dict(), flow_config=model.checkpoint_config()))
            restored.eval()
            torch.testing.assert_close(restored(images), expected)


if __name__ == '__main__':
    unittest.main()
