import os, torch, glob
from torch.utils.data import Dataset
from .utils import read_nifti_file

def create_centermap():

    import numpy as np 

    cube_size = 128
    sigma = 20

    grid_x, grid_y, grid_z = np.mgrid[0:cube_size, 0:cube_size, 0:cube_size]
    D2 = (grid_x- cube_size//2)**2 + (grid_y - cube_size//2)**2 + (grid_z - cube_size//2)**2

    return np.exp(-D2/2.0/sigma/sigma)


class segmentation_cubes_loader(Dataset):
    def __init__(self, data_dir, train_ids, idv_vertebra: bool):  # spine or individual vertebra
        super(segmentation_cubes_loader, self).__init__

        self.filelist_img = []
        self.filelist_msk = []


        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)
            img_dir = os.path.join(id_folder, 'img')
            msk_dir = os.path.join(id_folder, 'msk')

            img_files = sorted(glob.glob(os.path.join(img_dir, '*.nii.gz')))
            msk_files = sorted(glob.glob(os.path.join(msk_dir, '*.nii.gz')))

            assert len(img_files) == len(msk_files)

            for img_file, msk_file in zip(img_files, msk_files):
                self.filelist_img.append(img_file)
                self.filelist_msk.append(msk_file)
                
    def __getitem__(self, index):

        assert len(self.filelist_img) == len(self.filelist_msk)

        img_cube = read_nifti_file(self.filelist_img[index])
        msk_cube = read_nifti_file(self.filelist_msk[index])

        # img_cube = globalNormalization(img_cube)
        img_cube = torch.from_numpy(img_cube).to(dtype=torch.float32)
        msk_cube = torch.from_numpy(msk_cube).to(dtype=torch.float32)

        self.img_cube = img_cube.view(1, img_cube.size(0), img_cube.size(1), img_cube.size(2))
        self.msk_cube = msk_cube.view(1, msk_cube.size(0), msk_cube.size(1), msk_cube.size(2))

        if idv_vertebra:
            centermap = create_centermap()
            centermap = torch.from_numpy(centermap).to(dtype=torch.float32)
            centermap = centermap.view(1, centermap.size(0), centermap.size(1), centermap.size(2))
            self.img_cube = torch.cat((self.img_cube, centermap), 0)

        return self.img_cube, self.msk_cube

    def __len__(self):

        return len(self.filelist_img)



class group_cubes_loader(Dataset):
    def __init__(self, data_dir, train_ids):
        super(group_cubes_loader, self).__init__

        self.filelist = []
        self.labellist = []

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)

            msk_files = glob.glob(os.path.join(id_folder, '*.nii.gz'))

            for file in msk_files:
                bone_label = int(file.split('bone')[-1].split('_')[0]) 
                self.filelist.append(file)
                self.labellist.append(bone_label)


    def __getitem__(self, index):

        assert len(self.filelist) == len(self.labellist)

        msk_cube = read_nifti_file(self.filelist[index])
        label = self.labellist[index] 
        if label <= 7:
            label = 0
        elif label <= 19 and label >= 8:
            label = 1
        elif label >= 20 and label != 28:
            label = 2
        else:
            label = 1

        msk_cube = torch.from_numpy(msk_cube).to(dtype=torch.float32)

        self.msk_cube = msk_cube.view(1, msk_cube.size(0), msk_cube.size(1), msk_cube.size(2))
        self.label = torch.tensor(label).to(torch.long)

        return {'image': self.msk_cube, 'label': self.label}

    def __len__(self):

        return len(self.filelist)


class cervical_cubes_loader(Dataset):
    def __init__(self, data_dir, train_ids):
        super(cervical_cubes_loader, self).__init__

        self.filelist = []
        self.labellist = []

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)

            msk_files = glob.glob(os.path.join(id_folder, '*.nii.gz'))

            for file in msk_files:
                bone_label = int(file.split('bone')[-1].split('_')[0]) 
                if bone_label > 7:
                    continue 
                self.filelist.append(file)
                self.labellist.append(bone_label)


    def __getitem__(self, index):

        assert len(self.filelist) == len(self.labellist)

        msk_cube = read_nifti_file(self.filelist[index])
        label = self.labellist[index] - 1

        msk_cube = torch.from_numpy(msk_cube).to(dtype=torch.float32)

        self.msk_cube = msk_cube.view(1, msk_cube.size(0), msk_cube.size(1), msk_cube.size(2))

        self.label = torch.tensor(label).to(torch.long)

        return {'image': self.msk_cube, 'label': self.label}

    def __len__(self):

        return len(self.filelist)



class thoracic_cubes_loader(Dataset):
    def __init__(self, data_dir, train_ids):
        super(thoracic_cubes_loader, self).__init__

        self.filelist = []
        self.labellist = []

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)

            msk_files = glob.glob(os.path.join(id_folder, '*.nii.gz'))

            for file in msk_files:
                bone_label = int(file.split('bone')[-1].split('_')[0]) 
                if bone_label == 28:
                    self.filelist.append(file)
                    self.labellist.append(bone_label)
                elif bone_label < 8 or bone_label > 19:
                    continue
                else:
                    self.filelist.append(file)
                    self.labellist.append(bone_label)                   


    def __getitem__(self, index):

        assert len(self.filelist) == len(self.labellist)

        msk_cube = read_nifti_file(self.filelist[index])
        label = self.labellist[index]

        if label == 28:
            label = 11
        else:
            label -= 8

        msk_cube = torch.from_numpy(msk_cube).to(dtype=torch.float32)

        self.msk_cube = msk_cube.view(1, msk_cube.size(0), msk_cube.size(1), msk_cube.size(2))
        self.label = torch.tensor(label).to(torch.long)

        return {'image': self.msk_cube, 'label': self.label}

    def __len__(self):

        return len(self.filelist)



class lumbar_cubes_loader(Dataset):
    def __init__(self, data_dir, train_ids):
        super(lumbar_cubes_loader, self).__init__

        self.filelist = []
        self.labellist = []

        for vol_id in train_ids:
            id_folder = os.path.join(data_dir, vol_id)

            msk_files = glob.glob(os.path.join(id_folder, '*.nii.gz'))

            for file in msk_files:
                bone_label = int(file.split('bone')[-1].split('_')[0]) 
                if bone_label < 20 or bone_label == 28:
                    continue
                self.filelist.append(file)
                self.labellist.append(bone_label)
                
    def __getitem__(self, index):

        assert len(self.filelist) == len(self.labellist)

        msk_cube = read_nifti_file(self.filelist[index])
        label = self.labellist[index]

        if label == 25:
            label = 4
        else:
            label -= 20

        msk_cube = torch.from_numpy(msk_cube).to(dtype=torch.float32)

        self.msk_cube = msk_cube.view(1, msk_cube.size(0), msk_cube.size(1), msk_cube.size(2))
        self.label = torch.tensor(label).to(torch.long)

        return {'image': self.msk_cube, 'label': self.label}

    def __len__(self):

        return len(self.filelist)