import itertools
import os
from abc import abstractmethod

import SimpleITK as sitk
from scipy.ndimage import center_of_mass
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
        self.im_pairs = pd.read_csv(im_pairs, names=['fixed', 'moving'], skiprows=1).applymap(str)

        self.__set_im_spacing(), self.__set_padding()

    def __len__(self):
        return len(self.im_pairs.index)

    @property
    def dims_im(self):
        return (1, *self.dims)

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

        if self.mask_filename == '':
            mask_fixed = (seg_fixed != 0)

        return {'im': im_fixed, 'mask': mask_fixed, 'seg': seg_fixed}

    def _get_moving(self, idx):
        ID_moving = self.im_pairs['moving'].iloc[idx]

        # moving image
        im_moving, _ = self._get_im(ID_moving)
        mask_moving, _ = self._get_mask(ID_moving)
        seg_moving, _ = self._get_seg(ID_moving)

        if self.mask_filename == '':
            mask_moving = (seg_moving != 0)

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
    def __init__(self, save_paths, im_pairs, dims, data_path=''):
        if data_path == '':
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


################################### TASK 1 classes


structures_dict_4 = {'liver': 1,
                     'spleen': 2,
                     'right_kidney': 3,
                     'left_kidney': 4}

structures_dict_RGB = {'liver': [0, 245, 50, 0],
                       'spleen': [0, 20, 80, 160],
                       'right_kidney': [0, 20, 140, 134],
                       'left_kidney': [0, 0, 250, 60]}


class MrCtDataset(BaseImageRegistrationDataset):
    def __init__(self, save_paths, im_pairs, dims, is_val=False, train_paired_only=False, data_path=''):
        if data_path == '':
            self.data_path = '/vol/biodata/data/learn2reg/2021/task01/'
        else:
            self.data_path = data_path
        self.ct_folder, self.mr_folder, self.paired_folder = "L2R_Task1_CT", "L2R_Task1_MR", "L2R_Task1_MRCT_Train"
        self.train_paired_only = train_paired_only

        im_filename, seg_filename, mask_filename = '', '', 'L2R_Task1_ROI'
        structures_dict = structures_dict_4  # segmentation IDs

        im_pairs, val_pairs = self._get_im_pairs(im_pairs, save_paths)

        if is_val:
            im_pairs = val_pairs

        super().__init__(self.data_path, im_pairs, im_filename, mask_filename, seg_filename, dims,
                         structures_dict=structures_dict)

        # self.mean, self.std_dev = 92.89025002195125, 49.03549963976218  # NOTE (DG): pre-computed by me in a jupyter notebook, (HQ): ROI only
        self.mean, self.std_dev = -338.43452, 545.05903  # NOTE (HQ): entire image not only ROI

    def _get_im_pairs(self, im_pairs_path, save_paths, val_ids=[12, 14, 16]):
        val_im_pairs_path = []

        if im_pairs_path == '':
            mr_paths_train, mr_paths_val = self.images_in_folder("_MR.nii.gz",
                                                                 os.path.join(self.data_path, self.mr_folder),
                                                                 os.path.join(self.data_path, self.paired_folder), val_ids)
            ct_paths_train, _ = self.images_in_folder("_CT.nii.gz",
                                                                 os.path.join(self.data_path, self.ct_folder),
                                                                 os.path.join(self.data_path, self.paired_folder), val_ids)

            if self.train_paired_only:
                # HQ: Trying training with paired images only
                train_pairs = [(mr_path, ct_path) for (mr_path, ct_path) in zip(mr_paths_train, ct_paths_train)]
            else:
                train_pairs = []

                for mr_path in mr_paths_train:
                    for ct_path in ct_paths_train:
                        train_pairs.append((mr_path, ct_path))

            val_pairs = []

            for mr_path in mr_paths_val:
                val_pairs.append((mr_path, self.replace_right(mr_path, "MR", "CT", 1)))

            train_pairs_pd = pd.DataFrame(train_pairs)
            im_pairs_path = os.path.join(save_paths['run_dir'], 'train_pairs.csv')
            train_pairs_pd.to_csv(im_pairs_path, header=False, index=False)

            val_pairs_pd = pd.DataFrame(val_pairs)
            val_im_pairs_path = os.path.join(save_paths['run_dir'], 'val_pairs.csv')
            val_pairs_pd.to_csv(val_im_pairs_path, header=False, index=False)

        return im_pairs_path, val_im_pairs_path

    @staticmethod
    def replace_right(source, target, replacement, replacements=None):
        return replacement.join(source.rsplit(target, replacements))

    def images_in_folder(self, key, folder_unpaired: str, folder_paired: str, pairs_val):
        list_img_train, list_seg_train = [], []

        to_exclude = ['img0036_bcv_CT.nii.gz']  # NOTE (DG): bad data

        list_img = [os.path.join(folder_paired, file) for file in os.listdir(folder_paired) if key in file and 'img' in file and 'seg' not in file and 'mask' not in file and file not in to_exclude]
        list_img_val = [file for file in list_img if int(file.split("_")[-3][-4:]) in pairs_val]

        if self.train_paired_only:
            # HQ: Trying training with paired images only
            list_img_train = [file for file in list_img if int(file.split("_")[-3][-4:]) not in pairs_val]
        else:
            list_img_train = [os.path.join(folder_unpaired, file) for file in os.listdir(folder_unpaired) if key in file and 'img' in file and 'seg' not in file and 'mask' not in file and file not in to_exclude]
            list_img_train += [file for file in list_img if int(file.split("_")[-3][-4:]) not in pairs_val]

        list_seg_train += [file.replace("img", "seg") for file in list_img_train if key in file]

        return list_img_train, list_img_val

    def _get_fixed(self, idx):
        ID_fixed = self.im_pairs['fixed'].iloc[idx]

        # fixed image
        mask_fixed, spacing = self._get_mask(ID_fixed)
        im_fixed, _ = self._get_im(ID_fixed)
        # im_fixed, _ = self._get_im(ID_fixed, mask=mask_fixed, normalisation='per-case-z-score')
        seg_fixed, _ = self._get_seg(ID_fixed)

        return {'im': im_fixed, 'mask': mask_fixed, 'seg': seg_fixed}, spacing

    def _get_moving(self, idx, fixed_center=None):
        ID_moving = self.im_pairs['moving'].iloc[idx]

        # moving image
        mask_moving, spacing = self._get_mask(ID_moving)
        moving_center = self._get_center_of_mask(mask_moving, np.array(spacing))
        if fixed_center is not None:
            translation = sitk.TranslationTransform(3)
            tr = ((fixed_center - moving_center) * np.array([2, 2, -2]))
            translation.SetParameters(np.floor(tr))
        else:
            translation = None
        im_moving, spacing = self._get_im(ID_moving, normalisation='z-score', transform=translation)
        seg_moving, _ = self._get_seg(ID_moving, transform=translation)
        mask_moving, _ = self._get_mask(ID_moving, transform=translation)

        return {'im': im_moving, 'mask': mask_moving, 'seg': seg_moving}

    @staticmethod
    def resample(image, transform):
        # Output image Origin, Spacing, Size, Direction are taken from the reference
        # image in this call to Resample
        reference_image = image
        interpolator = sitk.sitkLinear
        default_value = 0.0
        return sitk.Resample(image, reference_image, transform,
                             interpolator, default_value)

    def _load_im_or_mask_or_seg_file(self, im_or_mask_or_seg_path, transform: sitk.TranslationTransform = None):
        im_or_mask_or_seg = sitk.ReadImage(im_or_mask_or_seg_path)
        spacing = im_or_mask_or_seg.GetSpacing()
        if transform is not None:
            im_or_mask_or_seg = self.resample(im_or_mask_or_seg, transform=transform)

        im_or_mask_or_seg = sitk.GetArrayFromImage(im_or_mask_or_seg)
        im_or_mask_or_seg = torch.from_numpy(im_or_mask_or_seg)

        return im_or_mask_or_seg.unsqueeze(0).unsqueeze(0), spacing

    def _get_im(self, ID, mask=None, normalisation='linear', transform: sitk.TranslationTransform = None):
        im_path = self._get_im_path_from_ID(ID)

        im, spacing = self._load_im_or_mask_or_seg_file(im_path, transform=transform)
        im = self._preprocess_im(im, mask=mask, normalisation=normalisation)

        return im, spacing

    def _get_mask(self, ID, transform: sitk.TranslationTransform = None):
        if self.mask_filename == '':
            return torch.ones(self.dims_im).bool(), None

        mask_path = self._get_mask_path_from_ID(ID)
        mask, spacing = self._load_im_or_mask_or_seg_file(mask_path, transform=transform)
        mask = self._preprocess_mask_or_seg(mask)

        return mask.bool(), spacing

    def _get_seg(self, ID, transform: sitk.TranslationTransform = None):
        seg_path = self._get_seg_path_from_ID(ID)
        seg, spacing = self._load_im_or_mask_or_seg_file(seg_path, transform=transform)
        seg = self._preprocess_mask_or_seg(seg)

        return seg.long(), spacing

    def _get_im_path_from_ID(self, subject_idx):
        return subject_idx

    def _get_seg_path_from_ID(self, subject_idx: str):
        return subject_idx.replace("img", "seg")

    def _get_mask_path_from_ID(self, subject_idx: str):
        tmp_list = subject_idx.split("/")
        tmp_list.insert(-2, self.mask_filename)
        tmp_list[-1] = tmp_list[-1].replace('img', 'mask')
        return "/".join(tmp_list)

    def _get_center_of_mask(self, image: torch.Tensor, spacing=2):
        if isinstance(image, torch.Tensor):
            img = image.squeeze().numpy()
        else:
            img = sitk.GetArrayFromImage(image)
        center = center_of_mass(img)
        return center * spacing

    def _preprocess(self, im_or_mask_or_seg):
        im_or_mask_or_seg = im_or_mask_or_seg.float()
        return im_or_mask_or_seg.permute([0, 1, 4, 3, 2])

    def _preprocess_im(self, im, mask=None, normalisation='linear'):
        im = self._preprocess(im)
        im = F.interpolate(im, size=self.dims, mode='trilinear', align_corners=True)

        if normalisation == 'linear':
            return rescale_im_intensity(im).squeeze(0)
        elif normalisation == 'per-case-z-score':
            mean_roi = im[mask].mean()
            std_dev_roi = im[mask].std()

            return (im.squeeze(0) - mean_roi) / std_dev_roi
        elif normalisation == 'z-score':
            return (im.squeeze(0) - self.mean) / self.std_dev

        raise NotImplementedError

    def _preprocess_mask_or_seg(self, mask_or_seg):
        mask_or_seg = self._preprocess(mask_or_seg)
        mask_or_seg = F.interpolate(mask_or_seg, size=self.dims, mode='nearest')

        return mask_or_seg.squeeze(0)

    def __getitem__(self, idx):
        fixed, spacing = self._get_fixed(idx)
        fixed_center = self._get_center_of_mask(fixed['mask'], np.array(spacing))
        moving = self._get_moving(idx, fixed_center=fixed_center)
        # var_params_q_v = self._get_var_params(idx)
        return fixed, moving  #, var_params_q_v

#
# class DatasetTask1(MrCtDataset):
#     def __init__(self, save_paths, im_pairs, dims, is_val=False, train_paired_only=False):  # , offline_augmentation=False
#         super().__init__(save_paths, im_pairs, dims, is_val=is_val, train_paired_only=train_paired_only)
#
#         self.no_classes = 4
#         self.mask_RGB = self._get_mask_RGB(structures_dict_RGB)
#
#         # flag for Zeju's augmentation
#         # self.offline_augmentation = offline_augmentation
#
#     def _get_mask_RGB(self, structures_dict_RGB):
#         mask_RGB = torch.zeros(len(structures_dict_RGB) + 1, 4, *self.dims)
#
#         for idx, (structure_name, RGB) in enumerate(structures_dict_RGB.items()):
#             mask_RGB[idx + 1, 1, ...].add_(RGB[1])
#             mask_RGB[idx + 1, 2, ...].add_(RGB[2])
#             mask_RGB[idx + 1, 3, ...].add_(RGB[3])
#
#         return mask_RGB
#
#     def _preprocess_mask_or_seg(self, mask_or_seg_orig):
#         return super()._preprocess_mask_or_seg(mask_or_seg_orig)

#########################################################

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
