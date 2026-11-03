from .base_options import BaseOptions


class BankOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--real_bank_output', required=True, help='output path for the real feature bank')
        parser.add_argument('--fake_bank_output', default=None, help='optional output path for the fake feature bank')
        parser.add_argument('--bank_size', type=int, default=0, help='maximum samples per class; 0 includes all training samples')
        parser.add_argument('--bank_seed', type=int, default=3000)
        self.isTrain = True
        return parser

    def gather_options(self):
        opt = super().gather_options()
        if opt.method != 'reference_flow' or opt.data_label != 'train':
            self.parser.error('Feature bank preparation requires --method reference_flow and --data_label train.')
        if opt.bank_size < 0 or opt.bank_size == 1:
            self.parser.error('--bank_size must be 0 (all samples) or at least 2 per class.')
        return opt
