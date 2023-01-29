import argparse
import itertools
import json
import numpy as np
import os
import torch
import torch.nn.functional as F
import utils
import wandb

from datetime import datetime
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils import LCC, MI, SSD, UNet, Learn2RegDataLoader, OasisDataset, SGLD,\
    calc_dsc, calc_no_non_diffeomorphic_voxels, init_grid_im, log_images, rescale_im_intensity, save_model, to_device, write_hparams, write_json


DEVICE = torch.device('cuda:0')
GLOBAL_STEP = 0
separator = '----------------------------------------'


def set_up_model_and_preprocessing(args):
    global DEVICE

    with open(args.config) as f:
        config = json.load(f)

    print(f'config: {config}')
    config['start_epoch'] = 1

    if config['loss_init'] == 'ssd':
        loss_init = lambda x, y=None: SSD(x[:, 1:2], x[:, 0:1], mask=y)
    elif config['loss_init'] == 'lcc':
        lcc_module = LCC().to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
        loss_init = lambda x, y=None: lcc_module(x[:, 1:2], x[:, 0:1])
    elif config['loss_init'] == 'mi':
        mi_module = MI().to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
        loss_init = lambda x, y=None: mi_module(x[:, 1:2], x[:, 0:1])
    else:
        raise NotImplementedError(f'Loss {args.loss} not supported')

    # model
    model = UNet(config).to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
    # model = torch.compile(model)
    no_params_sim, no_params_enc, no_params_dec = model.no_params
    print(f'NO. PARAMETERS OF THE SIMILARITY METRIC: {no_params_sim}, ENCODER: {no_params_enc}, DECODER: {no_params_dec}')

    # optimisers
    optimizer_enc = torch.optim.Adam(list(model.submodules['enc'].parameters()), lr=config['lr'])
    optimizer_dec = torch.optim.Adam(list(model.submodules['dec'].parameters()), lr=config['lr'])
    optimizer_sim_pretrain = torch.optim.Adam(list(model.submodules['sim'].parameters()), lr=config['lr_sim_pretrain'])
    optimizer_sim = torch.optim.Adam(list(model.submodules['sim'].parameters()), lr=config['lr_sim'])

    # lr schedulers
    scheduler_sim_pretrain = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_sim_pretrain, factor=config['sim_pretrain_schedule_factor'], patience=config['sim_pretrain_schedule_patience'])
    scheduler_sim = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_sim, factor=config['sim_schedule_factor'], patience=config['sim_schedule_patience'])

    # resuming training
    if args.pretrained_stn is not None:
        with torch.no_grad():
            curr_state = model.state_dict()
            checkpoint = torch.load(args.pretrained_stn)

            for name, param in checkpoint.items():
                if name not in curr_state:
                    continue

                curr_state[name].copy_(param)

        print('Pre-trained STN successfully loaded..')

    if args.resume is not None:
        global GLOBAL_STEP
        checkpoint = torch.load(args.resume)
        GLOBAL_STEP, config['start_epoch'] = checkpoint['step'] + 1, checkpoint['epoch'] + 1

        model.load_state_dict(checkpoint['model'])
        optimizer_enc.load_state_dict(checkpoint['optimizer_enc'])
        optimizer_dec.load_state_dict(checkpoint['optimizer_dec'])
        optimizer_sim_pretrain.load_state_dict(checkpoint['optimizer_sim_pretrain'])
        optimizer_sim.load_state_dict(checkpoint['optimizer_sim'])

        scheduler_sim_pretrain.load_state_dict(checkpoint['scheduler_sim_pretrain'])
        scheduler_sim.load_state_dict(checkpoint['scheduler_sim'])

    config_dict = {'config': config, 'loss_init': loss_init, 'model': model,
                   'optimizer_enc': optimizer_enc, 'optimizer_dec': optimizer_dec,
                   'optimizer_sim_pretrain': optimizer_sim_pretrain, 'optimizer_sim': optimizer_sim,
                   'scheduler_sim_pretrain': scheduler_sim_pretrain, 'scheduler_sim': scheduler_sim}
    print(config_dict)

    return config_dict


def generate_samples_from_EBM(config, epoch, sim, fixed, moving, moving_warped, writer, wandb=False):
    global GLOBAL_STEP

    no_samples_SGLD = config['no_samples_SGLD']
    
    def init_optimizer_LD(config, sample_plus):
        return torch.optim.Adam([sample_plus], lr=config['tau'])

    def init_sample(config, fixed):
        if config['init_sample'] == 'rand':
            sample_plus = torch.rand_like(fixed['im']).detach()
        elif config['init_sample'] == 'fixed':
            sample_plus = fixed['im'].clone().detach()
        else:
            raise NotImplementedError

        sample_plus.requires_grad_(True)
        return sample_plus

    sample_plus = init_sample(config, fixed)
    optimizer_LD = init_optimizer_LD(config, sample_plus)

    with torch.no_grad():
        input_minus = torch.cat((moving['im'], fixed['im']), dim=1)

        sigma, tau = torch.ones_like(sample_plus), config['tau']
        sigma.requires_grad_(False)

    for _ in trange(1, no_samples_SGLD + 1, desc=f'sampling from EBM', colour='#808080', dynamic_ncols=True, leave=False, unit='sample'):
        sample_plus_noise = SGLD.apply(sample_plus, sigma, tau)
        input_plus = torch.cat((moving_warped, sample_plus_noise), dim=1)

        loss_plus = sim(input_plus)
        loss_minus = sim(input_minus)

        loss = (loss_plus - loss_minus) * fixed['mask'].sum()

        optimizer_LD.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_LD.step()

        with torch.no_grad():
            writer.add_scalar(f'train/epoch_{epoch}/sample_plus_energy', loss_plus.item(), GLOBAL_STEP)

        GLOBAL_STEP += 1

    with torch.no_grad():
        writer.add_images('train/sample_plus/sagittal', sample_plus[:, :, sample_plus.size(2) // 2, ...], GLOBAL_STEP)
        writer.add_images('train/sample_plus/axial', sample_plus[:, :, :, sample_plus.size(3) // 2, ...], GLOBAL_STEP)
        writer.add_images('train/sample_plus/coronal', sample_plus[..., sample_plus.size(4) // 2], GLOBAL_STEP)
    
    if wandb:
        wandb_data = {'train/sample_plus': utils.plot_tensor(sample_plus)}
        wandb.log(wandb_data)

    plt.close('all')

    return torch.clamp(sample_plus_noise, min=0.0, max=1.0).detach()


def train(args):
    global GLOBAL_STEP

    # create output directories
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')

    out_dir = os.path.join(args.out, timestamp)
    log_dir, model_dir = os.path.join(out_dir, 'log'), os.path.join(out_dir, 'checkpoints')

    if args.exp_name is not None:
        log_dir = f'{log_dir}_{args.exp_name}'

    os.makedirs(out_dir), os.makedirs(log_dir), os.makedirs(model_dir)
    args.out_dir, args.log_dir, args.model_dir = out_dir, log_dir, model_dir

    # config
    config_dict = set_up_model_and_preprocessing(args)
    config = config_dict['config']

    loss_init, model = config_dict['loss_init'], config_dict['model']
    enc, dec = model.submodules['enc'], model.submodules['dec']
    sim = model.submodules['sim']

    optimizer_enc, optimizer_dec = config_dict['optimizer_enc'], config_dict['optimizer_dec']
    optimizer_sim_pretrain, optimizer_sim = config_dict['optimizer_sim_pretrain'], config_dict['optimizer_sim']
    scheduler_sim_pretrain, scheduler_sim = config_dict['scheduler_sim_pretrain'], config_dict['scheduler_sim']

    write_json(config, os.path.join(out_dir, 'config.json'))

    # tensorboard writer
    writer = SummaryWriter(log_dir)
    write_hparams(writer, config)
    print(separator)

    config_dict['args'] = vars(args)
    
    if args.wandb:
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

    alpha, reg_weight = config['alpha'], config['reg_weight']

    """
    PRE-TRAINING
    """

    start_epoch, end_epoch = config['start_epoch'], config['epochs_pretrain_model']

    if args.pretrained_stn is None or args.pretrain_sim:
        for epoch in trange(start_epoch, end_epoch + 1, desc='MODEL PRE-TRAINING'):
            for fixed, moving in tqdm(dataloader_train, desc=f'Epoch {epoch}', unit='batch', leave=False):
                fixed, moving = to_device(DEVICE, fixed, moving)
                input = torch.cat((moving['im'], fixed['im']), dim=1)
                log_dict = {}

                if args.wandb:
                    wandb_data = {'pretrain': {}}

                # REGISTRATION NETWORK
                if args.pretrained_stn is None:
                    enc.train(), enc.enable_grads()
                    dec.train(), dec.enable_grads()
                    sim.eval(), sim.disable_grads()

                    moving_warped = model(input)
                    input_warped = torch.cat((moving_warped, fixed['im']), dim=1)

                    data_term = loss_init(input_warped, fixed['mask'])
                    reg_term = model.regularizer()
                    loss_registration = data_term + reg_weight * reg_term

                    optimizer_enc.zero_grad(set_to_none=True)
                    optimizer_dec.zero_grad(set_to_none=True)
                    loss_registration.mean().backward()
                    optimizer_enc.step()
                    optimizer_dec.step()

                    # tensorboard
                    if GLOBAL_STEP % config['log_period'] == 0:
                        with torch.no_grad():
                            data_term, reg_term, loss_registration = data_term.mean(), reg_term.mean(), loss_registration.mean()
                            non_diffeomorphic_voxels = calc_no_non_diffeomorphic_voxels(model.get_T2())
                            non_diffeomorphic_voxels_pct = non_diffeomorphic_voxels / np.prod(fixed['im'].shape)

                            log_dict.update({'pretrain/train/loss_data': data_term.item(),
                                             'pretrain/train/loss_regularisation': reg_weight * reg_term.item(),
                                             'pretrain/train/loss_registration': loss_registration.item(),
                                             'pretrain/train/non_diffeomorphic_voxels': non_diffeomorphic_voxels.item(),
                                             'pretrain/train/non_diffeomorphic_voxels_pct': non_diffeomorphic_voxels_pct.item()})

                            grid_warped = model.warp_image(grid)
                            fixed_masked, moving_masked = fixed['im'] * fixed['mask'], moving['im'] * moving['mask']
                            log_images(writer, GLOBAL_STEP, fixed, moving, fixed_masked, moving_masked, moving_warped, grid_warped, 'pretrain')
                            
                            if args.wandb:
                               wandb_data['pretrain'].update(log_dict)
                               wandb_data['pretrain'].update({'fixed': utils.plot_tensor(fixed['im']),
                                                              'moving': utils.plot_tensor(moving['im']),
                                                              'moving_warped': utils.plot_tensor(moving_warped),
                                                              'transformation': utils.plot_tensor(grid_warped, grid=True),
                                                              'fixed_masked': utils.plot_tensor(fixed_masked),
                                                              'moving_masked': utils.plot_tensor(moving_masked)})

                # SIMILARITY METRIC
                if not args.baseline and args.pretrain_sim:
                    enc.eval(), enc.disable_grads()
                    dec.eval(), dec.disable_grads()
                    sim.train(), sim.enable_grads()
                    
                    with torch.no_grad():
                        moving_warped = model(input)

                        cartesian_prod = list(itertools.product([fixed['im'], moving['im'], moving_warped], [fixed['im'], moving['im'], moving_warped]))
                        inputs = [torch.cat((el[0], el[1]), dim=1) for el in cartesian_prod]

                        rand1, rand2 = rescale_im_intensity(torch.rand_like(fixed['im'])), rescale_im_intensity(torch.rand_like(moving['im']))
                        cartesian_prod_rand = list(itertools.product([rand1, rand2], [rand1, rand2]))
                        inputs_rand = [torch.cat((el[0], el[1]), dim=1) for el in cartesian_prod_rand]

                    no_samples = len(inputs) + len(inputs_rand)
                    loss_similarity = 0.0
                    
                    if config['sim_pretrain_grads']:
                        zero_disp = torch.zeros_like(dec.grid)
                        zero_disp.requires_grad_(True)

                    for input in inputs + inputs_rand:
                        if config['sim_pretrain_grads']:
                            moving_warped = dec.warp_image(input[:, 0:1], disp=zero_disp)
                            input_warped = torch.cat((moving_warped, input[:, 1:2]), dim=1)
                        else:
                            input_warped = input

                        data_term_sim, data_term_sim_pred = loss_init(input_warped), sim(input_warped).sum()
                        loss_similarity += F.mse_loss(data_term_sim_pred, data_term_sim) / no_samples
                        
                        if config['sim_pretrain_grads']:
                           data_term_sim_grad = torch.autograd.grad(data_term_sim, zero_disp, retain_graph=True)[0]
                           data_term_sim_pred_grad = torch.autograd.grad(data_term_sim_pred, zero_disp, retain_graph=True)[0]

                           loss_similarity += F.mse_loss(data_term_sim_grad, data_term_sim_pred_grad) / no_samples

                    optimizer_sim_pretrain.zero_grad(set_to_none=True)
                    loss_similarity.backward()
                    optimizer_sim_pretrain.step()

                    # tensorboard
                    if GLOBAL_STEP % config['log_period'] == 0:
                        with torch.no_grad():
                            log_dict['pretrain/train/loss_similarity'] = loss_similarity.item()

                            if args.wandb:
                                wandb_data['pretrain'].update(log_dict)

                GLOBAL_STEP += 1

            # VALIDATION
            if epoch == start_epoch or epoch % config['val_period'] == 0:
                enc.eval(), enc.disable_grads()
                dec.eval(), dec.disable_grads()
                sim.eval(), sim.disable_grads()

                with torch.no_grad():
                    dsc = torch.zeros(len(dataloader_val), len(structures_dict))
                    non_diffeomorphic_voxels, non_diffeomorphic_voxels_pct = torch.zeros(len(dataloader_val)), torch.zeros(len(dataloader_val))

                    loss_val_registration, loss_val_similarity = 0.0, 0.0

                    for idx, (fixed, moving) in enumerate(dataloader_val):
                        fixed, moving = to_device(DEVICE, fixed, moving)
                        input = torch.cat((moving['im'], fixed['im']), dim=1)

                        moving_warped = model(input)
                        seg_moving_warped = model.warp_image(moving['seg'].float(), interpolation='nearest').long()

                        input_warped = torch.cat((moving_warped, fixed['im']), dim=1)

                        data_term = loss_init(input_warped, fixed['mask'])
                        reg_term = model.regularizer()
                        loss_val_registration += data_term + reg_weight * reg_term

                        dsc[idx, :] = calc_dsc(fixed['seg'], seg_moving_warped, structures_dict)
                        non_diffeomorphic_voxels[idx] = calc_no_non_diffeomorphic_voxels(model.get_T2())
                        non_diffeomorphic_voxels_pct[idx] = non_diffeomorphic_voxels[idx] / np.prod(fixed['im'].shape)

                        if not args.baseline and args.pretrain_sim:
                            cartesian_prod = list(itertools.product([fixed['im'], moving['im'], moving_warped], [fixed['im'], moving['im'], moving_warped]))
                            inputs = [torch.cat((el[0], el[1]), dim=1) for el in cartesian_prod]

                            rand1, rand2 = rescale_im_intensity(torch.rand_like(fixed['im'])), rescale_im_intensity(torch.rand_like(moving['im']))
                            cartesian_prod_rand = list(itertools.product([rand1, rand2], [rand1, rand2]))
                            inputs_rand = [torch.cat((el[0], el[1]), dim=1) for el in cartesian_prod_rand]

                            no_samples = len(cartesian_prod) + len(cartesian_prod_rand)

                            for input in inputs + inputs_rand:
                                data_term_init, data_term_pred = loss_init(input), sim(input).sum()
                                loss_val_similarity += F.l1_loss(data_term_init, data_term_pred) / no_samples
                    
                    # tensorboard            
                    if GLOBAL_STEP % config['log_period'] == 0:
                        with torch.no_grad():
                            dsc_mean = torch.mean(dsc, dim=0)
                            non_diffeomorphic_voxels_mean = torch.mean(non_diffeomorphic_voxels)
                            non_diffeomorphic_voxels_pct_mean = torch.mean(non_diffeomorphic_voxels_pct)
                            
                            log_dict.update({'pretrain/val/loss_data': data_term.item(),
                                             'pretrain/val/loss_regularisation': reg_weight * reg_term.item(),
                                             'pretrain/val/loss_registration': loss_val_registration.item() / len(dataloader_val),
                                             'pretrain/val/non_diffeomorphic_voxels': non_diffeomorphic_voxels_mean.item(),
                                             'pretrain/val/non_diffeomorphic_voxels_pct_mean': non_diffeomorphic_voxels_pct_mean.item(),
                                             'pretrain/val/metric_dsc_avg': dsc_mean.mean().item()})

                            for structure_idx, structure_name in enumerate(structures_dict):
                                log_dict[f'pretrain/val/metric_dsc_{structure_name}'] = dsc_mean[structure_idx].item()

                            if not args.baseline and args.pretrain_sim:
                                log_dict['pretrain/val/loss_similarity'] =  loss_val_similarity.item() / len(dataloader_val)
                            
                            if args.wandb:
                                wandb_data['pretrain'].update(log_dict)
                    
            if GLOBAL_STEP % config['log_period'] == 0:
                with torch.no_grad():
                    for key, val in log_dict.items():
                        writer.add_scalar(key, val, global_step=GLOBAL_STEP)
                    
                    if args.wandb:
                        wandb.log(wandb_data)

                    plt.close('all')
                    
            save_model(args, epoch, GLOBAL_STEP, model, optimizer_enc, optimizer_dec, optimizer_sim_pretrain, optimizer_sim, scheduler_sim_pretrain, scheduler_sim)

    if args.baseline:
        return

    """
    TRAIN THE MODEL
    """

    start_epoch = end_epoch
    end_epoch = start_epoch + config['epochs'] - 1

    for epoch in trange(start_epoch, end_epoch + 1, desc='MODEL TRAINING'):
        for fixed, moving in tqdm(dataloader_train, desc=f'Epoch {epoch}', unit='batch', leave=False):
            fixed, moving = to_device(DEVICE, fixed, moving)
            input = torch.cat((moving['im'], fixed['im']), dim=1)
            log_dict = {}
            
            if args.wandb:
                wandb_data = {'train': {}}

            # REGISTRATION NETWORK
            enc.train(), enc.enable_grads()
            dec.train(), dec.enable_grads()
            sim.eval(), sim.disable_grads()

            moving_warped = model(input)
            input_warped = torch.cat((moving_warped, fixed['im']), dim=1) * fixed['mask']

            data_term = sim(input_warped)
            reg_term = model.regularizer()
            loss_registration = data_term + reg_weight * reg_term

            optimizer_enc.zero_grad(set_to_none=True)
            optimizer_dec.zero_grad(set_to_none=True)
            loss_registration.backward()
            optimizer_enc.step()
            optimizer_dec.step()

            # tensorboard
            if GLOBAL_STEP % config['log_period'] == 0:
                with torch.no_grad():
                    data_term, reg_term, loss_registration = data_term.mean(), reg_term.mean(), loss_registration.mean()
                    non_diffeomorphic_voxels = calc_no_non_diffeomorphic_voxels(model.get_T2())
                    non_diffeomorphic_voxels_pct = non_diffeomorphic_voxels / np.prod(fixed['im'].shape)

                    dict.update({'train/train/loss_data': data_term.item(),
                                 'train/train/loss_regularisation': reg_weight * reg_term.item(),
                                 'train/train/loss_registration': loss_registration.item(),
                                 'train/train/non_diffeomorphic_voxels': non_diffeomorphic_voxels.item(),
                                 'train/train/non_diffeomorphic_voxels_pct': non_diffeomorphic_voxels_pct.item()})

                    grid_warped = model.warp_image(grid)
                    fixed_masked, moving_masked = fixed['im'] * fixed['mask'], moving['im'] * moving['mask']
                    log_images(writer, GLOBAL_STEP, fixed, moving, fixed_masked, moving_masked, moving_warped, grid_warped, 'train')
                    
                    if args.wandb:
                       wandb_data['train'].update(log_dict)
                       wandb_data['train'].update({'fixed': utils.plot_tensor(fixed['im']),
                                                   'moving': utils.plot_tensor(moving['im']),
                                                   'moving_warped': utils.plot_tensor(moving_warped),
                                                   'transformation': utils.plot_tensor(grid_warped, grid=True),
                                                   'fixed_masked': utils.plot_tensor(fixed_masked),
                                                   'moving_masked': utils.plot_tensor(moving_masked)})

            GLOBAL_STEP += 1

            if GLOBAL_STEP % config['log_period'] == 0:
                with torch.no_grad():
                    for key, val in log_dict.items():
                        writer.add_scalar(key, val, global_step=GLOBAL_STEP)
                    
                    if args.wandb:
                        wandb.log(wandb_data)

                    plt.close('all')

        # SIMILARITY METRIC
        enc.eval(), enc.disable_grads()
        dec.eval(), dec.disable_grads()

        moving_warped = model(input)
        sample_plus = generate_samples_from_EBM(config, epoch, sim, fixed, moving, moving_warped, writer)

        sim.train(), sim.enable_grads()

        input_plus = torch.cat((moving_warped, sample_plus), dim=1) * fixed['mask']
        loss_plus = sim(input_plus)
        input_minus = torch.cat((moving['im'], fixed['im']), dim=1) * fixed['mask']
        loss_minus = sim(input_minus)

        loss_sim = loss_plus.mean() - loss_minus.mean()

        if config['reg_energy_type'] == 'exp':
            exponent = 1
            reg_energy = torch.exp(-1.0 * (loss_minus.mean() - loss_plus.mean()) ** exponent)
        elif config['reg_energy_type'] == 'tikhonov':
            reg_energy = (loss_plus.mean() - loss_minus.mean()) ** 2
        else:
            raise NotImplementedError

        loss_sim = loss_sim + alpha * reg_energy

        optimizer_sim.zero_grad(set_to_none=True)
        loss_sim.backward()
        optimizer_sim.step()
        # scheduler_sim.step()

        GLOBAL_STEP += 1

        # tensorboard
        if GLOBAL_STEP % config['log_period'] == 0:
            with torch.no_grad():
                log_dict['train/train/loss_similarity'] = loss_sim.item()

                if args.wandb:
                    wandb_data['train'].update(log_dict)

        # VALIDATION
        if epoch == start_epoch or epoch % config['val_period'] == 0:
            enc.eval(), enc.disable_grads()
            dec.eval(), dec.disable_grads()
            sim.eval(), sim.disable_grads()

            with torch.no_grad():
                dsc = torch.zeros(len(dataloader_val), len(structures_dict))
                non_diffeomorphic_voxels, non_diffeomorphic_voxels_pct = torch.zeros(len(dataloader_val)), torch.zeros(len(dataloader_val))

                for idx, (fixed, moving) in enumerate(dataloader_val):
                    fixed, moving = to_device(DEVICE, fixed, moving)
                    input = torch.cat((moving['im'], fixed['im']), dim=1)

                    noving_warped = model(input)
                    seg_moving_warped = model.warp_image(moving['seg'].float(), interpolation='nearest').long()

                    dsc[idx, :] = calc_dsc(fixed['seg'], seg_moving_warped, structures_dict)
                    non_diffeomorphic_voxels[idx] = calc_no_non_diffeomorphic_voxels(model.get_T2())
                    non_diffeomorphic_voxels_pct[idx] = non_diffeomorphic_voxels[idx] / np.prod(fixed['im'].shape)

                dsc_mean = torch.mean(dsc, dim=0)
                non_diffeomorphic_voxels_mean = torch.mean(non_diffeomorphic_voxels)
                non_diffeomorphic_voxels_pct_mean = torch.mean(non_diffeomorphic_voxels_pct)

                # scheduler_sim.step(dsc_mean.mean())

                log_dict.update({'train/val/metric_dsc_avg': dsc_mean.mean().item(),
                                 'train/val/non_diffeomorphic_voxels': non_diffeomorphic_voxels_mean.item(),
                                 'train/val/non_diffeomorphic_voxels_pct': non_diffeomorphic_voxels_pct_mean.item()})

                for structure_idx, structure_name in enumerate(structures_dict):
                    log_dict[f'train/val/metric_dsc_{structure_name}'] = dsc_mean[structure_idx].item()
                
                if args.wandb:
                    wandb_data['train'].update(log_dict)

        if GLOBAL_STEP % config['log_period'] == 0:
            with torch.no_grad():
                for key, val in log_dict.items():
                    writer.add_scalar(key, val, global_step=GLOBAL_STEP)
                
                if args.wandb:
                    wandb.log(wandb_data)

                plt.close('all')

        save_model(args, epoch, GLOBAL_STEP, model, optimizer_enc, optimizer_dec, optimizer_sim_pretrain, optimizer_sim, scheduler_sim_pretrain, scheduler_sim)


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='learnsim-EBM')
    # wandb args
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--config', default=None, required=True, help='config file')
    parser.add_argument('--baseline', action='store_true', default=False, help='')
    parser.add_argument('--exp-name', default=None, help='experiment name')
    parser.add_argument('--pretrain-sim', dest='pretrain_sim', default=True, action='store_true')
    parser.add_argument('--dont-pretrain-sim', dest='pretrain_sim', action='store_false')
    parser.add_argument('--pretrained-stn', default=None, help='path to a pre-trained STN checkpoint')
    parser.add_argument('--resume', default=None, help='path to a model checkpoint')

    # logging args
    parser.add_argument('--out', default='saved', help='output root directory')

    # training
    args = parser.parse_args()

    if args.wandb:
        wandb.login(key=args.wandb_key)

    train(args)
