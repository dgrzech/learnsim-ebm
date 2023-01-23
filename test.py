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
from utils import UNet, Learn2RegDataLoader, OasisDataset, calc_dsc, calc_no_non_diffeomorphic_voxels, init_grid_im, log_images, rescale_im_intensity, save_model, to_device, write_hparams, write_json


DEVICE = torch.device('cuda:0')
separator = '----------------------------------------'


def set_up_model_and_preprocessing(args):
    global DEVICE

    with open(args.config) as f:
        config = json.load(f)

    print(f'config: {config}')
    config['start_epoch'] = 1

    # model
    model = UNet(config).to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
    # model = torch.compile(model)
    no_params_sim, no_params_enc, no_params_dec = model.no_params
    print(f'NO. PARAMETERS OF THE SIMILARITY METRIC: {no_params_sim}, ENCODER: {no_params_enc}, DECODER: {no_params_dec}')

    # loading the model
    curr_state = model.state_dict()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model'])

    config_dict = {'config': config, 'model': model}
    print(config_dict)

    return config_dict


def test(args):
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

    dataset_val = OasisDataset(save_paths_dict, config['im_pairs_val'], dims, config['data_path'])
    dataloader_val = Learn2RegDataLoader(dataset_val, 1, no_workers)

    dataset_test = dataset_val
    dataloader_test = dataloader_val
    structures_dict = dataset_test.structures_dict

    grid = init_grid_im(config['dims'], spacing=3).to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
    grid = torch.cat(batch_size * [grid])
    
    model = config_dict['model']
    model.submodules['enc'].eval(), model.submodules['enc'].disable_grads()
    model.submodules['dec'].eval(), model.submodules['dec'].disable_grads()

    """
    TEST
    """
    
    idx = -1

    for fixed, moving in tqdm(dataloader_test, desc='Test', unit='image pair', leave=True):
        idx += 1

        fixed, moving = to_device(DEVICE, fixed, moving)
        input = torch.cat((moving['im'], fixed['im']), dim=1)
        log_dict = {}

        if args.wandb:
            wandb_data = {'test': {}}
        
        with torch.no_grad():
            moving_warped = model(input)
            seg_moving_warped = model.warp_image(moving['seg'].float(), interpolation='nearest').long()

            dsc = calc_dsc(fixed['seg'], seg_moving_warped, structures_dict)[0]
            
            for structure_idx, structure_name in enumerate(structures_dict):
                log_dict[f'test/{idx}/metric_dsc_{structure_name}'] = dsc[structure_idx].item()

            dsc_mean = torch.mean(dsc, dim=0)
            log_dict.update({f'test/{idx}/metric_dsc_avg': dsc_mean.item()})
            
            non_diffeomorphic_voxels = calc_no_non_diffeomorphic_voxels(model.get_T2())
            non_diffeomorphic_voxels_pct = non_diffeomorphic_voxels / np.prod(fixed['im'].shape)
            
            log_dict.update({f'test/{idx}/non_diffeomorphic_voxels': non_diffeomorphic_voxels.item(),
                             f'test/{idx}/non_diffeomorphic_voxels_pct': non_diffeomorphic_voxels_pct.item()})
            
            grid_warped = model.warp_image(grid)
            fixed_masked, moving_masked = fixed['im'] * fixed['mask'], moving['im'] * moving['mask']
            log_images(writer, idx, fixed, moving, fixed_masked, moving_masked, moving_warped, grid_warped, 'test')
            
            for key, val in log_dict.items():
                writer.add_scalar(key, val)
            
            if args.wandb:
               wandb_data['test'].update(log_dict)
               wandb_data['test'].update({'fixed': utils.plot_tensor(fixed['im']),
                                          'moving': utils.plot_tensor(moving['im']),
                                          'moving_warped': utils.plot_tensor(moving_warped),
                                          'transformation': utils.plot_tensor(grid_warped, grid=True),
                                          'fixed_masked': utils.plot_tensor(fixed_masked),
                                          'moving_masked': utils.plot_tensor(moving_masked)})
               wandb.log(wandb_data)
            
if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='learnsim-EBM')
    # wandb args
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--config', default=None, required=True, help='config file')
    parser.add_argument('--exp-name', default=None, help='experiment name')
    parser.add_argument('--resume', default=None, help='path to a model checkpoint')

    # logging args
    parser.add_argument('--out', default='saved_TEST', help='output root directory')

    # testing
    args = parser.parse_args()

    if args.wandb:
        wandb.login(key=args.wandb_key)

    test(args)
