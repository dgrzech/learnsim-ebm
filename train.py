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
from utils import CNN_SSD, LCC, Learn2RegDataLoader, MI, OasisDataset, SGLD, SSD, UNet,\
    calc_DSC, calc_no_non_diffeomorphic_voxels, init_grid_im, log_images, save_model, write_hparams, write_json


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
        loss_init = lambda x, y: SSD(x[:, 1:2], x[:, 0:1], mask=y)
    elif config['loss_init'] == 'lcc':
        lcc_module = LCC().to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
        loss_init = lambda x, y: lcc_module(x[:, 1:2], x[:, 0:1])
    elif config['loss_init'] == 'mi':
        mi_module = MI().to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
        loss_init = lambda x, y: mi_module(x[:, 1:2], x[:, 0:1])
    else:
        raise NotImplementedError(f'Loss {args.loss} not supported')

    # model
    model = UNet(config['dims'], activation_fn_sim=config['activation_fn_sim'], enable_spectral_norm=config['spectral_norm']).to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
    no_params_sim, no_params_enc, no_params_dec = model.no_params
    print(f'NO. PARAMETERS OF THE SIMILARITY METRIC: {no_params_sim}, ENCODER: {no_params_enc}, DECODER: {no_params_dec}')

    # optimisers
    optimizer_enc = torch.optim.Adam(list(model.submodules['enc'].parameters()), lr=config['lr'])
    optimizer_dec = torch.optim.Adam(list(model.submodules['dec'].parameters()), lr=config['lr'])
    optimizer_sim_pretrain = torch.optim.Adam(list(model.submodules['sim'].parameters()), lr=config['lr_sim_pretrain'])
    optimizer_sim = torch.optim.Adam(list(model.submodules['sim'].parameters()), lr=config['lr_sim'])

    # lr schedulers
    scheduler_sim_pretrain = torch.optim.lr_scheduler.StepLR(optimizer_sim_pretrain, config['sim_step_size_pretrain'], config['sim_gamma_pretrain'])
    scheduler_sim = torch.optim.lr_scheduler.StepLR(optimizer_sim, config['sim_step_size'], config['sim_gamma'])

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


def generate_samples_from_EBM(config, epoch, sim, fixed, moving_warped, writer):
    global GLOBAL_STEP

    no_samples_SGLD = config['no_samples_SGLD']
    
    def init_optimizer_LD(config, sample_plus, sample_minus):
        return torch.optim.Adam([sample_plus, sample_minus], lr=config['tau'])

    def init_samples(config, fixed):
        if config['init_sample_minus'] == 'rand':
            sample_plus, sample_minus = torch.rand_like(fixed['im']).detach(), torch.rand_like(fixed['im']).detach()
        elif config['init_sample_minus'] == 'fixed':
            sample_plus, sample_minus = fixed['im'].clone().detach(), fixed['im'].clone().detach()
        else:
            raise NotImplementedError

        sample_plus.requires_grad_(True), sample_minus.requires_grad_(True)
        return sample_plus, sample_minus

    sample_plus, sample_minus = init_samples(config, fixed)
    optimizer_LD = init_optimizer_LD(config, sample_plus, sample_minus)

    with torch.no_grad():
        sigma, tau = torch.ones_like(sample_minus), config['tau']
        sigma.requires_grad_(False)

    for _ in trange(1, no_samples_SGLD + 1, desc=f'sampling from EBM', colour='#808080', dynamic_ncols=True, leave=False, unit='sample'):
        sample_plus_noise, sample_minus_noise = SGLD.apply(sample_plus, sigma, tau), SGLD.apply(sample_minus, sigma, tau)
        input_plus, input_minus = torch.cat((moving_warped, sample_plus_noise), dim=1), torch.cat((moving_warped, sample_minus_noise), dim=1)

        loss_plus = sim(input_plus)
        loss_minus = sim(input_minus)

        loss = (loss_plus - loss_minus) * fixed['mask'].sum()

        optimizer_LD.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_LD.step()

        with torch.no_grad():
            writer.add_scalar(f'train/epoch_{epoch}/sample_plus_energy', loss_plus.item(), GLOBAL_STEP)
            writer.add_scalar(f'train/epoch_{epoch}/sample_minus_energy', loss_minus.item(), GLOBAL_STEP)

        GLOBAL_STEP += 1

    with torch.no_grad():
        writer.add_images('train/sample_plus/sagittal', sample_plus[:, :, sample_plus.size(2) // 2, ...], GLOBAL_STEP)
        writer.add_images('train/sample_plus/coronal', sample_plus[:, :, :, sample_plus.size(3) // 2, ...], GLOBAL_STEP)
        writer.add_images('train/sample_plus/axial', sample_plus[..., sample_plus.size(4) // 2], GLOBAL_STEP)
        writer.add_images('train/sample_minus/sagittal', sample_minus[:, :, sample_minus.size(2) // 2, ...], GLOBAL_STEP)
        writer.add_images('train/sample_minus/coronal', sample_minus[:, :, :, sample_minus.size(3) // 2, ...], GLOBAL_STEP)
        writer.add_images('train/sample_minus/axial', sample_minus[..., sample_minus.size(4) // 2], GLOBAL_STEP)

    wandb_data = {'sample_plus': utils.plot_tensor(sample_plus), 'sample_minus': utils.plot_tensor(sample_minus)}
    wandb.log(wandb_data)
    plt.close('all')

    return torch.clamp(sample_plus, min=0.0, max=1.0).detach(), torch.clamp(sample_minus, min=0.0, max=1.0).detach()


def train(args):
    global GLOBAL_STEP

    config_dict = set_up_model_and_preprocessing(args)

    config = config_dict['config']
    loss_init, model = config_dict['loss_init'], config_dict['model']
    enc, dec = model.submodules['enc'], model.submodules['dec']
    sim = model.submodules['sim']

    optimizer_enc, optimizer_dec = config_dict['optimizer_enc'], config_dict['optimizer_dec']
    optimizer_sim_pretrain, optimizer_sim = config_dict['optimizer_sim_pretrain'], config_dict['optimizer_sim']
    scheduler_sim_pretrain, scheduler_sim = config_dict['scheduler_sim_pretrain'], config_dict['scheduler_sim']

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
    PRE-TRAINING
    """

    start_epoch, end_epoch = config['start_epoch'], config['epochs_pretrain_model']

    if args.pretrained_stn is None or args.pretrain_sim:
        for epoch in trange(start_epoch, end_epoch + 1, desc='MODEL PRE-TRAINING'):
            for fixed, moving in tqdm(dataloader_train, desc=f'Epoch {epoch}', unit='batch', leave=False):
                for key in fixed:
                    fixed[key] = fixed[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
                    moving[key] = moving[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)

                input = torch.cat((moving['im'], fixed['im']), dim=1)
                wandb_data = dict({'pretrain': {}})

                if args.pretrained_stn is None:
                    # REGISTRATION NETWORK
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
                            grid_warped = model.warp_image(grid)

                            non_diffeomorphic_voxels = calc_no_non_diffeomorphic_voxels(model.get_T2())
                            non_diffeomorphic_voxels_pct = non_diffeomorphic_voxels / np.prod(fixed['im'].shape)

                            writer.add_scalar('pretrain/train/loss_data', data_term.item(), GLOBAL_STEP)
                            writer.add_scalar('pretrain/train/loss_regularisation', reg_weight * reg_term.item(), GLOBAL_STEP)
                            writer.add_scalar('pretrain/train/loss_registration', loss_registration.item(), GLOBAL_STEP)
                            writer.add_scalar('pretrain/train/non_diffeomorphic_voxels', non_diffeomorphic_voxels.item(), GLOBAL_STEP)
                            writer.add_scalar('pretrain/train/non_diffeomorphic_voxels_pct', non_diffeomorphic_voxels_pct.item(), GLOBAL_STEP)

                            fixed_masked = fixed['im'] * fixed['mask']
                            moving_masked = moving['im'] * moving['mask']

                            log_images(writer, GLOBAL_STEP, fixed, moving, fixed_masked, moving_masked, moving_warped, grid_warped, 'pretrain')
                            wandb_data['pretrain'].update({'loss_data': data_term.item(),
                                                            'loss_regularisation': reg_weight * reg_term.item(),
                                                            'loss_registration': loss_registration.item(),
                                                            'non_diffeomorphic_voxels': non_diffeomorphic_voxels.item(),
                                                            'non_diffeomorphic_voxels_pct': non_diffeomorphic_voxels_pct.item(),
                                                            'fixed': utils.plot_tensor(fixed['im']),
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

                        rand1, rand2 = torch.rand_like(fixed['im']), torch.rand_like(moving['im'])
                        cartesian_prod_rand = list(itertools.product([rand1, rand2], [rand1, rand2]))
                        inputs_rand = [torch.cat((el[0], el[1]), dim=1) for el in cartesian_prod_rand]

                    no_samples = len(inputs) + len(inputs_rand)
                    loss_similarity = 0.0

                    for input in inputs:
                        data_term_sim, data_term_sim_pred = loss_init(input, torch.ones_like(fixed['mask'])), sim(input).sum()
                        loss_similarity += F.l1_loss(data_term_sim_pred, data_term_sim) / no_samples

                    for input in inputs_rand:
                        data_term_sim, data_term_sim_pred = loss_init(input, torch.ones_like(fixed['mask'])), sim(input).sum()
                        loss_similarity += F.l1_loss(data_term_sim_pred, data_term_sim) / no_samples

                    optimizer_sim_pretrain.zero_grad(set_to_none=True)
                    loss_similarity.backward()
                    optimizer_sim_pretrain.step()

                    # tensorboard
                    if GLOBAL_STEP % config['log_period'] == 0:
                        with torch.no_grad():
                            writer.add_scalar('pretrain/train/loss_similarity', loss_similarity.item(), GLOBAL_STEP)
                            wandb_data['pretrain']['loss_similarity'] = loss_similarity.item()

                if GLOBAL_STEP % config['log_period'] == 0:
                    wandb.log(wandb_data)
                    plt.close('all')

                GLOBAL_STEP += 1

            if not args.baseline and args.pretrain_sim:
                scheduler_sim_pretrain.step()

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
                        for key in fixed:
                            fixed[key] = fixed[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
                            moving[key] = moving[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)

                        input = torch.cat((moving['im'], fixed['im']), dim=1)
                        moving_warped = model(input)
                        seg_moving_warped = model.warp_image(moving['seg'].float(), interpolation='nearest').long()

                        input_warped = torch.cat((moving_warped, fixed['im']), dim=1)

                        data_term = loss_init(input_warped, fixed['mask'])
                        reg_term = model.regularizer()
                        loss_val_registration += data_term + reg_weight * reg_term

                        dsc[idx, :] = calc_DSC(fixed['seg'], seg_moving_warped, structures_dict)
                        non_diffeomorphic_voxels[idx] = calc_no_non_diffeomorphic_voxels(model.get_T2())
                        non_diffeomorphic_voxels_pct[idx] = non_diffeomorphic_voxels[idx] / np.prod(fixed['im'].shape)

                        if not args.baseline:
                            cartesian_prod = list(itertools.product([fixed['im'], moving['im'], moving_warped], [fixed['im'], moving['im'], moving_warped]))
                            inputs = [torch.cat((el[0], el[1]), dim=1) for el in cartesian_prod]

                            rand1, rand2 = torch.rand_like(fixed['im']), torch.rand_like(moving['im'])
                            cartesian_prod_rand = list(itertools.product([rand1, rand2], [rand1, rand2]))
                            inputs_rand = [torch.cat((el[0], el[1]), dim=1) for el in cartesian_prod_rand]

                            no_samples = len(cartesian_prod) + len(cartesian_prod_rand)

                            for input in inputs:
                                data_term = loss_init(input, torch.ones_like(fixed['mask']))
                                data_term_pred = sim(input).sum()

                                loss_val_similarity += F.l1_loss(data_term_pred, data_term) / no_samples

                            for input in inputs_rand:
                                data_term = loss_init(input, torch.ones_like(fixed['mask']))
                                data_term_pred = sim(input).sum()

                                loss_val_similarity += F.l1_loss(data_term_pred, data_term) / no_samples

                    # tensorboard
                    writer.add_scalar('pretrain/val/loss_registration', loss_val_registration.item() / len(dataloader_val), GLOBAL_STEP)

                    dsc_mean = torch.mean(dsc, dim=0)
                    non_diffeomorphic_voxels_mean = torch.mean(non_diffeomorphic_voxels)
                    non_diffeomorphic_voxels_pct_mean = torch.mean(non_diffeomorphic_voxels_pct)

                    writer.add_scalar('pretrain/val/metric_dsc_avg', torch.mean(dsc_mean).item(), GLOBAL_STEP)
                    writer.add_scalar('pretrain/val/non_diffeomorphic_voxels', non_diffeomorphic_voxels_mean.item(), GLOBAL_STEP)
                    writer.add_scalar('pretrain/val/non_diffeomorphic_voxels_pct', non_diffeomorphic_voxels_pct_mean.item(), GLOBAL_STEP)

                    wandb_data = {'pretrain_val': {'loss_registration': loss_val_registration.item(),
                                                   'metric_dsc_avg': torch.mean(dsc_mean).item(),
                                                   'non_diffeomorphic_voxels': non_diffeomorphic_voxels_mean.item(),
                                                   'non_diffeomorphic_voxels_pct_mean': non_diffeomorphic_voxels_pct_mean.item()}}

                    for structure_idx, structure_name in enumerate(structures_dict):
                        writer.add_scalar(f'pretrain/val/metric_dsc_{structure_name}', dsc_mean[structure_idx].item(), GLOBAL_STEP)
                        wandb_data['pretrain_val'][f'metric_dsc_{structure_name}'] = dsc_mean[structure_idx].item()

                    if not args.baseline:
                        writer.add_scalar('pretrain/loss_similarity', loss_val_similarity.item() / len(dataloader_val), GLOBAL_STEP)
                        wandb_data['pretrain_val']['loss_similarity'] = loss_val_similarity.item()

                    wandb.log(wandb_data)
                    plt.close('all')

            save_model(args, epoch, GLOBAL_STEP, model, optimizer_enc, optimizer_dec, optimizer_sim_pretrain, optimizer_sim, scheduler_sim_pretrain, scheduler_sim)

    if args.baseline:
        return

    """
    TRAIN THE MODEL
    """

    alpha = config['alpha']  # energy regularisation weight

    start_epoch = end_epoch
    end_epoch = start_epoch + config['epochs'] - 1

    for epoch in trange(start_epoch, end_epoch + 1, desc='MODEL TRAINING'):
        for fixed, moving in tqdm(dataloader_train, desc=f'Epoch {epoch}', unit='batch', leave=False):
            for key in fixed:
                fixed[key] = fixed[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
                moving[key] = moving[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)

            # REGISTRATION NETWORK
            enc.train(), enc.enable_grads()
            dec.train(), dec.enable_grads()
            sim.eval(), sim.disable_grads()

            input = torch.cat((moving['im'], fixed['im']), dim=1)
            moving_warped = model(input)
            input_warped = torch.cat((moving_warped, fixed['im']), dim=1)

            data_term = sim(input_warped, mask=fixed['mask'])
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

                    writer.add_scalar('train/loss_data', data_term.item(), GLOBAL_STEP)
                    writer.add_scalar('train/loss_regularisation', reg_weight * reg_term.item(), GLOBAL_STEP)
                    writer.add_scalar('train/loss_registration', loss_registration.item(), GLOBAL_STEP)
                    writer.add_scalar('train/non_diffeomorphic_voxels', non_diffeomorphic_voxels.item(), GLOBAL_STEP)
                    writer.add_scalar('train/non_diffeomorphic_voxels_pct', non_diffeomorphic_voxels_pct.item(), GLOBAL_STEP)

                    grid_warped = model.warp_image(grid)

                    fixed_masked = fixed['im'] * fixed['mask']
                    moving_masked = moving['im'] * moving['mask']

                    log_images(writer, GLOBAL_STEP, fixed, moving, fixed_masked, moving_masked, moving_warped, grid_warped, 'train')

                    wandb_data = {'train': {'loss_data': data_term.item(),
                                            'loss_regularisation': reg_weight * reg_term.item(),
                                            'loss_registration': loss_registration.item(),
                                            'non_diffeomorphic_voxels': non_diffeomorphic_voxels.item(),
                                            'non_diffeomorphic_voxels_pct': non_diffeomorphic_voxels_pct.item(),
                                            'fixed': utils.plot_tensor(fixed['im']),
                                            'moving': utils.plot_tensor(moving['im']),
                                            'moving_warped': utils.plot_tensor(moving_warped),
                                            'transformation': utils.plot_tensor(grid_warped, grid=True),
                                            'fixed_masked': utils.plot_tensor(fixed_masked),
                                            'moving_masked': utils.plot_tensor(moving_masked)}}

                    wandb.log(wandb_data)
                    plt.close('all')

            GLOBAL_STEP += 1

        # SIMILARITY METRIC
        enc.eval(), enc.disable_grads()
        dec.eval(), dec.disable_grads()

        with torch.no_grad():
            moving_warped = model(input)

        sample_plus, sample_minus = generate_samples_from_EBM(config, epoch, sim, fixed, moving_warped, writer)

        sim.train(), sim.enable_grads()

        input_plus = torch.cat((moving_warped, sample_plus), dim=1)
        loss_plus = sim(input_plus, mask=fixed['mask'])
        input_minus = torch.cat((moving_warped, sample_minus), dim=1)
        loss_minus = sim(input_minus, mask=fixed['mask'])

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

        scheduler_sim.step()

        # tensorboard
        if GLOBAL_STEP % config['log_period'] == 0:
            with torch.no_grad():
                writer.add_scalar('train/loss_similarity', loss_sim.item(), GLOBAL_STEP)
                wandb_data = {'train': {'loss_similarity': loss_sim.item()}}

                wandb.log(wandb_data)
                plt.close('all')

        GLOBAL_STEP += 1

        # VALIDATION
        if epoch == start_epoch or epoch % config['val_period'] == 0:
            with torch.no_grad():
                dsc = torch.zeros(len(dataloader_val), len(structures_dict))
                non_diffeomorphic_voxels, non_diffeomorphic_voxels_pct = torch.zeros(len(dataloader_val)), torch.zeros(len(dataloader_val))

                for idx, (fixed, moving) in enumerate(dataloader_val):
                    for key in fixed:
                        fixed[key] = fixed[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
                        moving[key] = moving[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)

                    input = torch.cat((moving['im'], fixed['im']), dim=1)
                    model(input)
                    seg_moving_warped = model.warp_image(moving['seg'].float(), interpolation='nearest').long()

                    dsc[idx, :] = calc_DSC(fixed['seg'], seg_moving_warped, structures_dict)
                    non_diffeomorphic_voxels[idx] = calc_no_non_diffeomorphic_voxels(model.get_T2())
                    non_diffeomorphic_voxels_pct[idx] = non_diffeomorphic_voxels[idx] / np.prod(fixed['im'].shape)

                # tensorboard
                dsc_mean = torch.mean(dsc, dim=0)
                non_diffeomorphic_voxels_mean = torch.mean(non_diffeomorphic_voxels)
                non_diffeomorphic_voxels_pct_mean = torch.mean(non_diffeomorphic_voxels_pct)

                writer.add_scalar('val/metric_dsc_avg', torch.mean(dsc_mean).item(), GLOBAL_STEP)
                writer.add_scalar('val/non_diffeomorphic_voxels', non_diffeomorphic_voxels_mean.item(), GLOBAL_STEP)
                writer.add_scalar('val/non_diffeomorphic_voxels_pct', non_diffeomorphic_voxels_pct_mean.item(), GLOBAL_STEP)

                wandb_data = {'validation': {'metric_dsc_avg': torch.mean(dsc_mean).item(),
                                             'non_diffeomorphic_voxels': non_diffeomorphic_voxels_mean.item(),
                                             'non_diffeomorphic_voxels_pct': non_diffeomorphic_voxels_pct_mean.item()}}

                for structure_idx, structure_name in enumerate(structures_dict):
                    writer.add_scalar(f'val/metric_dsc_{structure_name}', dsc_mean[structure_idx].item(), GLOBAL_STEP)
                    wandb_data['validation'][f'metric_dsc_{structure_name}'] = dsc_mean[structure_idx].item()

                wandb.log(wandb_data)
                plt.close('all')

        save_model(args, epoch, GLOBAL_STEP, model, optimizer_enc, optimizer_dec, optimizer_sim_pretrain, optimizer_sim,
                   scheduler_sim_pretrain, scheduler_sim)


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='learnsim-EBM')
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
    wandb.login(key=args.wandb_key)

    train(args)
