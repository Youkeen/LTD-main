from .base_options import BaseOptions


class BankOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--real_bank_output', required=True, help='output path for the real feature bank')
        parser.add_argument('--fake_bank_output', default=None, help='optional output path for the fake feature bank')
        parser.add_argument('--bank_size', type=int, default=4096, help='maximum samples per class')
        parser.add_argument('--bank_seed', type=int, default=3000)
        self.isTrain = True
        return parser

    def gather_options(self):
        opt = super().gather_options()
        if opt.method != 'reference_flow' or opt.data_label != 'train':
            self.parser.error('Feature bank preparation requires --method reference_flow and --data_label train.')
        if opt.bank_size < 2:
            self.parser.error('--bank_size must be at least 2 per class.')
        return opt
