import os
import time
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
import numpy as np
import torch
import random


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def train_epoch(model, data_loader, epoch, epochs, writer):
    model.train()
    model.stage_epoch = epoch
    start = time.time()
    total_loss, samples = 0.0, 0
    for i, data in enumerate(data_loader):
        model.total_steps += 1
        model.set_input(data)
        model.optimize_parameters()
        loss = model.loss.item()
        batch_size = len(data[0])
        total_loss += loss * batch_size
        samples += batch_size
        stage = model.training_stage
        print(f'\r[{stage} | Epoch {epoch:02d}/{epochs:02d} | '
              f'Batch {i + 1:04d}/{len(data_loader):04d} | {time.time() - start:.1f}s] '
              f'loss: {loss:.4f}', end='', flush=True)
        writer.add_scalar(f'{stage}/loss', loss, model.total_steps)
        for name, value in getattr(model, 'losses', {}).items():
            writer.add_scalar(f'{stage}/{name}', value.item(), model.total_steps)
    print()
    if samples == 0:
        raise ValueError('Training data loader is empty.')
    writer.add_scalar(f'{model.training_stage}/epoch_loss', total_loss / samples, epoch)


def train_velocity_stage(model, data_loader, epochs, writer):
    """Fixed-epoch FM training; no classifier/label-based checkpoint selection."""
    # Make FM sampling independent of how many parameters the readout has.
    set_seed(3000)
    for epoch in range(1, epochs + 1):
        train_epoch(model, data_loader, epoch, epochs, writer)
    model.save_networks('flow_final.pth')
    model.set_stage('classifier')
    print('Velocity training complete. Frozen CLIP and velocity field; released reference bank. '
          'Only the linear classifier will now be optimized.')


def main():
    opt = TrainOptions().parse()
    from tensorboardX import SummaryWriter
    from validate import validate
    from data import create_dataloader

    val_opt = get_val_opt()
    set_seed(3000)

    model = Trainer(opt)
    
    data_loader = create_dataloader(opt, None)
    val_loader = create_dataloader(val_opt)
    if opt.method in ('reference_flow', 'clip_linear'):
        from networks.feature_bank import path_id
        train_paths = {path_id(p) for p in data_loader.dataset.total_list}
        val_paths = {path_id(p) for p in val_loader.dataset.total_list}
        if train_paths & val_paths:
            raise ValueError('Training and validation paths overlap. Use separate splits before training.')
    if len(data_loader) == 0 or len(val_loader) == 0:
        raise ValueError('Training and source validation loaders must both be nonempty.')
    model.initialize_bank(data_loader.dataset)


    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    if opt.method == 'reference_flow':
        train_velocity_stage(model, data_loader, opt.flow_epochs, train_writer)

    delta = 0 if opt.method == 'reference_flow' else -0.001
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=delta, verbose=True)
    print ("Length of data loader: %d" %(len(data_loader)))
    
    for epoch in range(1, opt.niter + 1):
        train_epoch(model, data_loader, epoch, opt.niter, train_writer)
    
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks( f'model_epoch_{epoch}.pth')

        # Validation
        model.eval()
        
        ap, r_acc, f_acc, acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                # Preserve the best source score when changing LR; never replace best
                # with the first, potentially worse, score after a scheduler reset.
                early_stopping.counter = 0
                early_stopping.early_stop = False
            else:
                print("Early stopping.")
                break
        model.train()
    train_writer.close()
    val_writer.close()


if __name__ == '__main__':
    main()
