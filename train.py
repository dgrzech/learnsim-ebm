import argparse
import json
import os
from datetime import datetime

import itertools
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import wandb

import utils
from utils import LCC, Learn2RegDataLoader, MI, OasisDataset, SGLD, SSD, UNet, calc_dsc, init_grid_im, write_json

DEVICE = torch.device('cuda:0')
GLOBAL_STEP = 0
separator = '----------------------------------------'


def write_hparams(writer, config):
    hparams = ['epochs_pretrain_model', 'loss_init', 'reg_weight', 'lr', 'tau',
               'batch_size', 'no_samples_per_epoch', 'no_samples_SGLD', 'dims', 'cps']
    hparam_dict = dict(zip(hparams, [config[hparam] for hparam in hparams]))

    for k, v in hparam_dict.items():
        if type(v) == list:
            hparam_dict[k] = torch.tensor(v)

    writer.add_hparams(hparam_dict, metric_dict={'dummy_metric': 0.0}, run_name='.')


def set_up_model_and_preprocessing(args):
    global DEVICE

    with open(args.config) as f:
        config = json.load(f)

    print(f'config: {config}')
    config['start_epoch'] = 1

    if config['loss_init'] == 'ssd':
        loss_init = lambda x: SSD(x[:, 1:2], x[:, 0:1])
    elif config['loss_init'] == 'lcc':
        lcc_module = LCC().to(DEVICE, non_blocking=True)
        loss_init = lambda x: lcc_module(x[:, 1:2], x[:, 0:1])
    elif config['loss_init'] == 'mi':
        mi_module = MI().to(DEVICE, non_blocking=True)
        loss_init = lambda x: mi_module(x[:, 1:2], x[:, 0:1])
    else:
        raise NotImplementedError(f'Loss {args.loss} not supported')

    # model
    if config['cps'] == "none":
        config['cps'] = None

    model = UNet(config['dims'], config['cps'], enable_spectral_norm=config['spectral_norm']).to(DEVICE, non_blocking=True)
    no_params_sim, no_params_enc, no_params_dec = model.no_params
    print(f'NO. PARAMETERS OF THE SIMILARITY METRIC: {no_params_sim}, ENCODER: {no_params_enc}, DECODER: {no_params_dec}')

    # optimisers
    optimizer_enc = torch.optim.Adam(list(model.submodules['enc'].parameters()), lr=config['lr'])
    optimizer_dec = torch.optim.Adam(list(model.submodules['dec'].parameters()), lr=config['lr'])
    optimizer_sim_pretraining = torch.optim.Adam(list(model.submodules['sim'].parameters()), lr=config['lr_sim'])
    optimizer_sim = torch.optim.Adam(list(model.submodules['sim'].parameters()), lr=config['lr_sim'])

    # lr schedulers
    scheduler_sim_pretraining = torch.optim.lr_scheduler.StepLR(optimizer_sim_pretraining, config['sim_step_size_pretraining'], config['sim_gamma_pretraining'])
    scheduler_sim = torch.optim.lr_scheduler.StepLR(optimizer_sim_pretraining, config['sim_step_size'], config['sim_gamma'])

    # resuming training
    if args.resume is not None:
        global GLOBAL_STEP
        checkpoint = torch.load(args.resume)
        GLOBAL_STEP, config['start_epoch'] = checkpoint['step'] + 1, checkpoint['epoch'] + 1

        model.load_state_dict(checkpoint['model'])
        optimizer_enc.load_state_dict(checkpoint['optimizer_enc'])
        optimizer_dec.load_state_dict(checkpoint['optimizer_dec'])
        optimizer_sim_pretraining.load_state_dict(checkpoint['optimizer_sim_pretraining'])
        optimizer_sim.load_state_dict(checkpoint['optimizer_sim'])

        scheduler_sim_pretraining.load_state_dict(checkpoint['scheduler_sim_pretraining'])
        scheduler_sim.load_state_dict(checkpoint['scheduler_sim'])

    config_dict = {'config': config, 'loss_init': loss_init, 'model': model,
                   'optimizer_enc': optimizer_enc, 'optimizer_dec': optimizer_dec,
                   'optimizer_sim_pretraining': optimizer_sim_pretraining, 'optimizer_sim': optimizer_sim,
                   'scheduler_sim_pretraining': scheduler_sim_pretraining, 'scheduler_sim': scheduler_sim}
    print(config_dict)

    return config_dict


def save_model(args, epoch, model, optimizer_enc, optimizer_dec, optimizer_sim_pretraining, optimizer_sim,
               scheduler_sim_pretraining, scheduler_sim):
    global GLOBAL_STEP

    path = os.path.join(args.model_dir, f'checkpoint_{epoch}.pt')
    state_dict = {'epoch': epoch, 'step': GLOBAL_STEP, 'model': model.state_dict(),
                  'optimizer_enc': optimizer_enc.state_dict(), 'optimizer_dec': optimizer_dec.state_dict(),
                  'optimizer_sim_pretraining': optimizer_sim_pretraining.state_dict(), 'optimizer_sim': optimizer_sim.state_dict(),
                  'scheduler_sim_pretraining': scheduler_sim_pretraining.state_dict(), 'scheduler_sim': scheduler_sim.state_dict()}

    torch.save(state_dict, path)


def generate_samples_from_EBM(config, enc, sim, fixed, moving_warped, writer):
    global GLOBAL_STEP

    no_samples_SGLD = config['no_samples_SGLD']
    
    def init_optimizers_LD(config, sample_minus):
        optimizer_LD_minus = torch.optim.SGD([sample_minus], lr=config['tau'])
        return optimizer_LD_minus
    
    sample_minus = torch.clamp(torch.randn_like(fixed['im']), min=0.0, max=1.0)
    sample_minus.requires_grad_(True)

    optimizer_minus = init_optimizers_LD(config, sample_minus)
    sigma_minus, tau = torch.ones_like(sample_minus), config['tau']

    for _ in trange(1, no_samples_SGLD + 1, desc=f'sampling from EBM', colour='#808080', dynamic_ncols=True, leave=False, unit='sample'):
        sample_minus_noise = SGLD.apply(sample_minus, sigma_minus, tau)
        input_minus = torch.cat((moving_warped, sample_minus_noise), dim=1)
        loss_minus = sim(enc(input_minus), reduction='sum')
        loss_minus = loss_minus.sum()

        optimizer_minus.zero_grad(set_to_none=True)
        loss_minus.backward()
        optimizer_minus.step()

        with torch.no_grad():
            writer.add_scalar('train/sample_minus_energy', loss_minus.item(), GLOBAL_STEP)

        GLOBAL_STEP += 1

    with torch.no_grad():
        writer.add_images('train/sample_minus', sample_minus_noise[:, :, sample_minus_noise.size(2) // 2, ...], GLOBAL_STEP)

    return torch.clamp(sample_minus_noise, min=0.0, max=1.0).detach()


def train(args):
    global GLOBAL_STEP

    config_dict = set_up_model_and_preprocessing(args)

    config = config_dict['config']
    loss_init, model = config_dict['loss_init'], config_dict['model']
    enc, dec = model.submodules['enc'], model.submodules['dec']
    sim = model.submodules['sim']

    optimizer_enc, optimizer_dec = config_dict['optimizer_enc'], config_dict['optimizer_dec']
    optimizer_sim_pretraining, optimizer_sim = config_dict['optimizer_sim_pretraining'], config_dict['optimizer_sim']
    scheduler_sim_pretraining, scheduler_sim = config_dict['scheduler_sim_pretraining'], config_dict['scheduler_sim']

    # create output directories
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')

    out_dir = os.path.join(args.out, timestamp)
    log_dir, model_dir = os.path.join(out_dir, 'log'), os.path.join(out_dir, 'checkpoints')

    if args.baseline:
        log_dir = f'{log_dir}_baseline'
    if args.exp_name is not None:
        log_dir = f'{log_dir}_{args.exp_name}'

    os.makedirs(out_dir), os.makedirs(log_dir), os.makedirs(model_dir)
    args.out_dir, args.log_dir, args.model_dir = out_dir, log_dir, model_dir

    # save the config file
    write_json(config, os.path.join(out_dir, 'config.json'))

    # tensorboard writer
    writer = SummaryWriter(log_dir)
    print(separator)

    write_hparams(writer, config)

    config_dict['args'] = vars(args)
    wandb.init(project='learsim-ebm', config=config_dict, entity=args.wandb_entity)

    # dataset
    dims = config['dims']
    batch_size, no_workers, no_samples_per_epoch = config['batch_size'], config['no_workers'], config['no_samples_per_epoch']
    save_paths_dict = {'run_dir': args.out}

    dataset_train = OasisDataset(save_paths_dict, config['im_pairs_train'], dims, config['data_path'])
    dataloader_train = Learn2RegDataLoader(dataset_train, batch_size, no_workers, no_samples_per_epoch)

    dataset_val = OasisDataset(save_paths_dict, config['im_pairs_val'], dims, config['data_path'])
    dataloader_val = Learn2RegDataLoader(dataset_val, 1, no_workers)

    structures_dict = dataset_train.structures_dict

    grid = init_grid_im(config['dims'], spacing=3).to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
    grid = torch.cat(batch_size * [grid])

    reg_weight = config['reg_weight']

    """
    TRAIN THE REGISTRATION NETWORK AND PRE-TRAIN THE SIMILARITY METRIC
    """

    start_epoch, end_epoch = config['start_epoch'], config['epochs_pretrain_model']

    for epoch in trange(start_epoch, end_epoch + 1, desc='MODEL PRE-TRAINING'):
        for fixed, moving in tqdm(dataloader_train, desc=f'Epoch {epoch}', unit='batch', leave=False):
            for key in fixed:
                fixed[key] = fixed[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
                moving[key] = moving[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)

            # registration network
            enc.train()
            dec.train()
            sim.eval()

            input = torch.cat((moving['im'], fixed['im']), dim=1)
            moving_warped = model(input)
            input_warped = torch.cat((moving_warped, fixed['im']), dim=1)

            data_term = loss_init(input_warped)
            reg_term = model.regularizer()
            loss_registration = data_term + reg_weight * reg_term

            optimizer_enc.zero_grad(set_to_none=True)
            optimizer_dec.zero_grad(set_to_none=True)
            loss_registration.mean().backward()
            optimizer_enc.step()
            optimizer_dec.step()

            # similarity metric
            if not args.baseline:
                enc.eval()
                dec.eval()
                sim.train()

                with torch.no_grad():
                    moving_warped = model(input)

                cartesian_prod = list(itertools.product([fixed['im'], moving['im'], moving_warped], [fixed['im'], moving['im'], moving_warped]))
                loss_similarity = 0.0

                # actual input images
                for el in cartesian_prod:
                    input = torch.cat((el[0], el[1]), dim=1)
                    data_term_sim_pred = sim(enc(input))
                    data_term_sim = loss_init(input)

                    loss_similarity += F.l1_loss(data_term_sim_pred, data_term_sim)

                # random images
                for _ in range(config['no_samples_pretrain_sim']):
                    im_fixed, im_moving = torch.randn_like(fixed['im']), torch.randn_like(moving['im'])
                    input = torch.cat((im_fixed, im_moving), dim=1)
                    data_term_sim_pred = sim(enc(input))
                    data_term_sim = loss_init(input)

                    loss_similarity += F.l1_loss(data_term_sim_pred, data_term_sim)

                loss_similarity /= (len(cartesian_prod) + config['no_samples_pretrain_sim'])

                optimizer_sim_pretraining.zero_grad(set_to_none=True)
                loss_similarity.backward()
                optimizer_sim_pretraining.step()

            # tensorboard
            if GLOBAL_STEP % config['log_period'] == 0:
                with torch.no_grad():
                    data_term, reg_term, loss_registration = data_term.mean(), reg_term.mean(), loss_registration.mean()
                    writer.add_scalar('pretrain_model/train/loss_data', data_term.item(), GLOBAL_STEP)
                    writer.add_scalar('pretrain_model/train/loss_regularisation', reg_weight * reg_term.item(), GLOBAL_STEP)
                    writer.add_scalar('pretrain_model/train/loss_registration', loss_registration.item(), GLOBAL_STEP)

                    if not args.baseline:
                        loss_similarity = loss_similarity.mean()
                        writer.add_scalar('pretrain_model/train/loss_similarity', loss_similarity.item(), GLOBAL_STEP)

                    grid_warped = model.warp_image(grid)

                    writer.add_images('pretrain_model/fixed', fixed['im'][:, :, fixed['im'].size(2) // 2, ...], GLOBAL_STEP)
                    writer.add_images('pretrain_model/moving', moving['im'][:, :, moving['im'].size(2) // 2, ...], GLOBAL_STEP)
                    writer.add_images('pretrain_model/moving_warped', moving_warped[:, :, moving_warped.size(2) // 2, ...], GLOBAL_STEP)
                    writer.add_images('pretrain_model/transformation', grid_warped[:, 0:1, grid_warped.size(2) // 2, ...], GLOBAL_STEP)

                    wandb_data = {'pretrain_model': {'loss_data': data_term.item(),
                                                     'loss_regularisation': reg_weight * reg_term.item(),
                                                     'loss_registration': loss_registration.item(),
                                                     'fixed': utils.plot_tensor(fixed['im']),
                                                     'moving': utils.plot_tensor(moving['im']),
                                                     'moving_warped': utils.plot_tensor(moving_warped),
                                                     'transformation': utils.plot_tensor(grid_warped)}}

                    if not args.baseline:
                        wandb_data['pretrain_model']['loss_similarity'] = loss_similarity.item()

                    wandb.log(wandb_data)

            GLOBAL_STEP += 1

        scheduler_sim_pretraining.step()

        # VALIDATION
        if epoch == start_epoch or epoch % config['val_period'] == 0:
            with torch.no_grad():
                dsc = torch.zeros(len(dataloader_val), len(structures_dict))
                loss_val_registration, loss_val_similarity = 0.0, 0.0

                enc.eval()
                dec.eval()
                sim.eval()

                for idx, (fixed, moving) in enumerate(dataloader_val):
                    for key in fixed:
                        fixed[key] = fixed[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
                        moving[key] = moving[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)

                    input = torch.cat((moving['im'], fixed['im']), dim=1)
                    moving_warped = model(input)
                    seg_moving_warped = model.warp_image(moving['seg'].float(), interpolation='nearest').long()

                    dsc[idx, :] = calc_dsc(fixed['seg'], seg_moving_warped, structures_dict)

                    input_warped = torch.cat((moving_warped, fixed['im']), dim=1)
                    data_term = loss_init(input_warped)
                    reg_term = model.regularizer()
                    loss_val_registration += data_term + reg_weight * reg_term

                    if not args.baseline:
                        data_term_pred = sim(enc(input_warped))
                        loss_val_similarity += F.l1_loss(data_term_pred, data_term)


                # tensorboard
                writer.add_scalar('pretrain_model/val/loss_registration', loss_val_registration.item() / len(dataloader_val), GLOBAL_STEP)

                dsc_mean = torch.mean(dsc, dim=0)
                writer.add_scalar('pretrain_model/val/metric_dsc_avg', torch.mean(dsc_mean).item(), GLOBAL_STEP)

                wandb_data = {'pretrain_val': {
                    'loss_registration': loss_val_registration.item(),
                    'metric_dsc_avg': torch.mean(dsc_mean).item()
                }}

                if not args.baseline:
                    writer.add_scalar('pretrain_model/val/loss_similarity', loss_val_similarity.item() / len(dataloader_val), GLOBAL_STEP)
                    wandb_data['pretrain_val']['loss_similarity'] = loss_val_similarity.item()

                for structure_idx, structure_name in enumerate(structures_dict):
                    writer.add_scalar(f'pretrain_model/val/metric_dsc_{structure_name}', dsc_mean[structure_idx].item(), GLOBAL_STEP)
                    wandb_data['pretrain_val'][f'metric_dsc_{structure_name}'] = dsc_mean[structure_idx].item()

        save_model(args, epoch, model, optimizer_enc, optimizer_dec, optimizer_sim_pretraining, optimizer_sim,
                   scheduler_sim_pretraining, scheduler_sim)

    if args.baseline:
        return

    """
    TRAIN THE REGISTRATION NETWORK AND THE SIMILARITY METRIC
    """

    alpha = config['alpha']  # L2 regularisation weight

    start_epoch = end_epoch
    end_epoch = start_epoch + config['epochs'] - 1

    for epoch in trange(start_epoch, end_epoch + 1, desc='MODEL TRAINING'):
        for fixed, moving in tqdm(dataloader_train, desc=f'Epoch {epoch}', unit='batch', leave=False):
            for key in fixed:
                fixed[key] = fixed[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
                moving[key] = moving[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)

            # registration network
            enc.train()
            dec.train()
            sim.eval()

            input = torch.cat((moving['im'], fixed['im']), dim=1)
            moving_warped = model(input)
            input_warped = torch.cat((moving_warped, fixed['im']), dim=1)

            data_term = sim(enc(input_warped))
            reg_term = model.regularizer()
            loss_registration = data_term + reg_weight * reg_term

            optimizer_enc.zero_grad(set_to_none=True)
            optimizer_dec.zero_grad(set_to_none=True)
            loss_registration.mean().backward()
            optimizer_enc.step()
            optimizer_dec.step()

            # similarity metric
            enc.eval()
            dec.eval()
            sim.train()

            with torch.no_grad():
                input = torch.cat((moving['im'], fixed['im']), dim=1)
                moving_warped = model(input)

            sample_minus = generate_samples_from_EBM(config, enc, sim, fixed, moving_warped, writer)

            input_plus = torch.cat((moving_warped, fixed['im']), dim=1)
            loss_plus = sim(enc(input_plus))
            input_minus = torch.cat((moving_warped, sample_minus), dim=1)
            loss_minus = sim(enc(input_minus))

            loss_sim = loss_plus.mean() - loss_minus.mean()

            if config['reg_energy_type'] == 'exp':
                exponent = 1
                reg_energy = torch.exp(-1.0 * (loss_minus.mean() - loss_plus.mean()) ** exponent)
            elif config['reg_energy_type'] == 'tikhonov':
                reg_energy = (loss_plus.mean() + loss_minus.mean()) ** 2
            else:
                raise NotImplementedError

            loss_sim += alpha * reg_energy

            optimizer_sim.zero_grad(set_to_none=True)
            loss_sim.backward()
            optimizer_sim.step()

            # tensorboard
            if GLOBAL_STEP % config['log_period'] == 0:
                with torch.no_grad():
                    data_term, reg_term, loss_registration, loss_sim = data_term.mean(), reg_term.mean(), loss_registration.mean(), loss_sim.mean()
                    writer.add_scalar('train/loss_data', data_term.item(), GLOBAL_STEP)
                    writer.add_scalar('train/loss_regularisation', reg_weight * reg_term.item(), GLOBAL_STEP)
                    writer.add_scalar('train/loss_registration', loss_registration.item(), GLOBAL_STEP)
                    writer.add_scalar('train/loss_similarity', loss_sim.item(), GLOBAL_STEP)

                    grid_warped = model.warp_image(grid)

                    writer.add_images('train/fixed', fixed['im'][:, :, fixed['im'].size(2) // 2, ...], GLOBAL_STEP)
                    writer.add_images('train/moving', moving['im'][:, :, moving['im'].size(2) // 2, ...], GLOBAL_STEP)
                    writer.add_images('train/moving_warped', moving_warped[:, :, moving_warped.size(2) // 2, ...], GLOBAL_STEP)
                    writer.add_images('train/transformation', grid_warped[:, 0:1, grid_warped.size(2) // 2, ...], GLOBAL_STEP)

                    wandb_data = {'train': {'loss_data': data_term.item(),
                                              'loss_regularisation': reg_weight * reg_term.item(),
                                              'loss_registration': loss_registration.item(),
                                              'loss_similarity': loss_sim.item(),
                                              'fixed': utils.plot_tensor(fixed['im']),
                                              'moving': utils.plot_tensor(moving['im']),
                                              'moving_warped': utils.plot_tensor(moving_warped),
                                              'transformation': utils.plot_tensor(grid_warped)
                                            }
                                  }
                    wandb.log(wandb_data)

            GLOBAL_STEP += 1

        scheduler_sim.step()

        # VALIDATION
        if epoch == start_epoch or epoch % config['val_period'] == 0:
            with torch.no_grad():
                dsc = torch.zeros(len(dataloader_val), len(structures_dict))

                for idx, (fixed, moving) in enumerate(dataloader_val):
                    for key in fixed:
                        fixed[key] = fixed[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
                        moving[key] = moving[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)

                    input = torch.cat((moving['im'], fixed['im']), dim=1)
                    model(input)
                    seg_moving_warped = model.warp_image(moving['seg'].float(), interpolation='nearest').long()
                    dsc[idx, :] = calc_dsc(fixed['seg'], seg_moving_warped, structures_dict)

                # tensorboard
                dsc_mean = torch.mean(dsc, dim=0)
                writer.add_scalar('val/metric_dsc_avg', torch.mean(dsc_mean).item(), GLOBAL_STEP)

                wandb_data = {'validation': {
                    'metric_dsc_avg': torch.mean(dsc_mean).item()
                }}

                for structure_idx, structure_name in enumerate(structures_dict):
                    writer.add_scalar(f'val/metric_dsc_{structure_name}', dsc_mean[structure_idx].item(), GLOBAL_STEP)
                    wandb_data['validation'][f'metric_dsc_{structure_name}'] = dsc_mean[structure_idx].item()


        save_model(args, epoch, model, optimizer_enc, optimizer_dec, optimizer_sim_pretraining, optimizer_sim,
                   scheduler_sim_pretraining, scheduler_sim)


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='learnsim-EBM')
    parser.add_argument('--config', default=None, required=True, help='config file')
    parser.add_argument('--baseline', action='store_true', default=False, help='')
    parser.add_argument('--exp-name', default=None, help='experiment name')
    parser.add_argument('--resume', default=None, help='path to a model checkpoint')
    parser.add_argument('--wandb-key', type=str, required=True, help='key to login to your wandb account')
    parser.add_argument('--wandb-entity', type=str, default='ebm', choices=['ebm', 'mfazampour', 'dgrzech'],
                        help='wandb entity, the team one or personal one')

    # logging args
    parser.add_argument('--out', default='saved', help='output root directory')

    args = parser.parse_args()

    wandb.login(key=args.wandb_key)

    train(args)
