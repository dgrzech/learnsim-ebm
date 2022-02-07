import itertools
import os
from abc import abstractmethod

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler


def rescale_im_intensity(im, range_min=0.0, range_max=1.0):
    """
    rescale the intensity of image pixels/voxels to a given range
    """

    im_min, im_max = torch.min(im), torch.max(im)
    return (range_max - range_min) * (im - im_min) / (im_max - im_min) + range_min


class BaseImageRegistrationDataset(Dataset):
    def __init__(self, data_path, im_pairs, im_filename, mask_filename, seg_filename, dims, structures_dict):
        self.data_path = data_path
        self.dims, self.structures_dict = dims, structures_dict

        self.im_filename, self.mask_filename, self.seg_filename = im_filename, mask_filename, seg_filename
        self.im_pairs = pd.read_csv(im_pairs, names=['fixed', 'moving']).applymap(str)

        self.__set_im_spacing(), self.__set_padding()

    def __len__(self):
        return len(self.im_pairs.index)

    @property
    def dims_im(self):
        return 1, *self.dims

    """
    images, masks, and segmentations
    """

    @abstractmethod
    def _preprocess(self, im_or_mask_or_seg):
        pass

    @abstractmethod
    def _preprocess_im(self, im):
        pass

    @abstractmethod
    def _preprocess_mask_or_seg(self, mask_or_seg):
        pass

    @staticmethod
    def _load_im_or_mask_or_seg_file(im_or_mask_or_seg_path):
        im_or_mask_or_seg = sitk.ReadImage(im_or_mask_or_seg_path)
        spacing = im_or_mask_or_seg.GetSpacing()
        im_or_mask_or_seg = sitk.GetArrayFromImage(im_or_mask_or_seg)
        im_or_mask_or_seg = torch.from_numpy(im_or_mask_or_seg)

        return im_or_mask_or_seg.unsqueeze(0).unsqueeze(0), spacing

    def _get_im_path_from_ID(self, ID):
        return os.path.join(os.path.join(self.data_path, ID), self.im_filename)

    def _get_mask_path_from_ID(self, ID):
        return os.path.join(os.path.join(self.data_path, ID), self.mask_filename)

    def _get_seg_path_from_ID(self, ID):
        return os.path.join(os.path.join(self.data_path, ID), self.seg_filename)

    def _get_im(self, ID):
        im_path = self._get_im_path_from_ID(ID)
        im, spacing = self._load_im_or_mask_or_seg_file(im_path)
        im = self._preprocess_im(im)

        return im, spacing

    def _get_mask(self, ID):
        if self.mask_filename == '':
            return torch.ones(self.dims_im).bool(), None

        mask_path = self._get_mask_path_from_ID(ID)
        mask, spacing = self._load_im_or_mask_or_seg_file(mask_path)
        mask = self._preprocess_mask_or_seg(mask)

        return mask.bool(), spacing

    def _get_seg(self, ID):
        seg_path = self._get_seg_path_from_ID(ID)
        seg, spacing = self._load_im_or_mask_or_seg_file(seg_path)
        seg = self._preprocess_mask_or_seg(seg)

        return seg.long(), spacing

    def _get_fixed(self, idx):
        ID_fixed = self.im_pairs['fixed'].iloc[idx]

        # moving image
        im_fixed, _ = self._get_im(ID_fixed)
        mask_fixed, _ = self._get_mask(ID_fixed)
        seg_fixed, _ = self._get_seg(ID_fixed)

        return {'im': im_fixed, 'mask': mask_fixed, 'seg': seg_fixed}

    def _get_moving(self, idx):
        ID_moving = self.im_pairs['moving'].iloc[idx]

        # moving image
        im_moving, _ = self._get_im(ID_moving)
        mask_moving, _ = self._get_mask(ID_moving)
        seg_moving, _ = self._get_seg(ID_moving)

        return {'im': im_moving, 'mask': mask_moving, 'seg': seg_moving}

    def __set_im_spacing(self):
        idx = self.im_pairs['fixed'].sample().iloc[0]
        im_path = self._get_im_path_from_ID(idx)
        im, im_spacing = self._load_im_or_mask_or_seg_file(im_path)
        self.im_spacing = torch.tensor(im_spacing).float()

    def __set_padding(self):
        idx = self.im_pairs['fixed'].sample().iloc[0]
        im_path = self._get_im_path_from_ID(idx)
        im, _ = self._load_im_or_mask_or_seg_file(im_path)
        padding = (max(im.shape) - np.asarray(im.shape)) // 2
        self.padding = (*(padding[4],) * 2, *(padding[3],) * 2, *(padding[2],) * 2)


structures_dict_35 = {'left_cerebral_white_matter': 1,
                      'left_cerebral_cortex': 2,
                      'left_lateral_ventricle': 3,
                      'left_inf_lateral_ventricle': 4,
                      'left_cerebellum_white_matter': 5,
                      'left_cerebellum_cortex': 6,
                      'left_thalamus': 7,
                      'left_caudate': 8,
                      'left_putamen': 9,
                      'left_pallidum': 10,
                      '3rd_ventricle': 11,
                      '4th_ventricle': 12,
                      'brain_stem': 13,
                      'left_hippocampus': 14,
                      'left_amygdala': 15,
                      'left_accumbens': 16,
                      'left_ventral_dorsal_cord': 17,
                      'left_vessel': 18,
                      'left_choroid_plexus': 19,
                      'right_cerebral_white_matter': 20,
                      'right_cerebral_cortex': 21,
                      'right_lateral_ventricle': 22,
                      'right_inf_lateral_ventricle': 23,
                      'right_cerebellum_white_matter': 24,
                      'right_cerebellum_cortex': 25,
                      'right_thalamus': 26,
                      'right_caudate': 27,
                      'right_putamen': 28,
                      'right_pallidum': 29,
                      'right_hippocampus': 30,
                      'right_amygdala': 31,
                      'right_accumbens': 32,
                      'right_ventral_dorsal_cord': 33,
                      'right_vessel': 34,
                      'right_choroid_plexus': 35}


class OasisDataset(BaseImageRegistrationDataset):
    def __init__(self, save_paths, im_pairs, dims):
        data_path = '/vol/biodata/data/learn2reg/2021/task03'
        im_filename, seg_filename, mask_filename = 'aligned_norm.nii.gz', 'aligned_seg35.nii.gz', ''
        structures_dict = structures_dict_35  # segmentation IDs
        im_pairs = self._get_im_pairs(im_pairs, save_paths)

        super().__init__(data_path, im_pairs, im_filename, mask_filename, seg_filename, dims, structures_dict)

    @staticmethod
    def _get_im_pairs(im_pairs_path, save_paths):
        if im_pairs_path == '':
            missing_IDs = [8, 24, 36, 48, 89, 93, 100, 118, 128, 149, 154, 171, 172, 175, 187, 194, 196,
                           215, 219, 225, 242, 245, 248, 251, 252, 253, 257, 276, 297,
                           306, 320, 324, 334, 347, 360, 364, 369, 391, 393, 412, 414, 427, 436]
            existing_IDs = [idx for idx in range(1, 458) if idx not in missing_IDs]

            train_pairs, val_pairs = [], [(idx, idx + 1) for idx in range(438, 457)]

            for IDs in itertools.combinations(existing_IDs, 2):
                if IDs not in val_pairs and reversed(IDs) not in val_pairs:
                    train_pairs.append({'fixed': IDs[0], 'moving': IDs[1]})

            train_pairs = pd.DataFrame(train_pairs)
            im_pairs_path = os.path.join(save_paths['run_dir'], 'train_pairs.csv')
            train_pairs.to_csv(im_pairs_path, header=False, index=False)

        return im_pairs_path

    def _get_im_path_from_ID(self, subject_idx):
        return os.path.join(self.data_path, f'OASIS_OAS1_{str(subject_idx).zfill(4)}_MR1', self.im_filename)

    def _get_seg_path_from_ID(self, subject_idx):
        return os.path.join(self.data_path, f'OASIS_OAS1_{str(subject_idx).zfill(4)}_MR1', self.seg_filename)

    def _preprocess(self, im_or_mask_or_seg):
        return im_or_mask_or_seg.permute([0, 1, 4, 3, 2])

    def _preprocess_im(self, im):
        im = self._preprocess(im)
        im = F.interpolate(im, size=self.dims, mode='trilinear', align_corners=True)

        return rescale_im_intensity(im).squeeze(0)

    def _preprocess_mask_or_seg(self, mask_or_seg):
        mask_or_seg = self._preprocess(mask_or_seg)
        mask_or_seg = F.interpolate(mask_or_seg, size=self.dims, mode='nearest')

        return mask_or_seg.squeeze(0)

    def __getitem__(self, idx):
        return self._get_fixed(idx), self._get_moving(idx)


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, no_workers, no_samples_per_epoch=None):
        init_kwargs = {'batch_size': batch_size, 'dataset': dataset, 'num_workers': no_workers, 'pin_memory': True}

        if no_samples_per_epoch is not None:
            sampler = RandomSampler(dataset, replacement=True, num_samples=no_samples_per_epoch)
        else:
            sampler = None

        super().__init__(sampler=sampler, **init_kwargs)

    @property
    def dims(self):
        return self.dataset.dims

    @property
    def im_spacing(self):
        return self.dataset.im_spacing

    @property
    def save_dirs(self):
        return self.dataset.save_paths

    @property
    def structures_dict(self):
        return self.dataset.structures_dict


class Learn2RegDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, no_workers, no_samples_per_epoch=None):
        super().__init__(dataset, batch_size, no_workers, no_samples_per_epoch)
