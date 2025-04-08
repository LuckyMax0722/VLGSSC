import os
import glob
import torch

from torch.utils.data import Dataset

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader


class SemanticKITTIBlipDataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
    ):
        self.splits = {
            "predict": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
            "predict01": ["01", "02"],
            "predict02": ["03", "04", "05"],
            "predict03": ["06", "07", "08"],
            "predict04": ["09", "10"],
            "00": ["00"],
        }
        self.split = split
        self.sequences = self.splits[split]

        self.data_root = data_root

        self.data_infos = self.load_annotations()


    def __len__(self):
        return len(self.data_infos)

    def prepare_data(self, index):
        """
        data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in data')
            return None

        return input_dict
    
    def __getitem__(self, idx):
        data = self.prepare_data(idx)

        return data
    
    def get_data_info(self, index):
        info = self.data_infos[index]

        '''
        sample info includes the following:
            "sequence": sequence,
            "frame_id": frame_id,
            "img_2_path": img_2_path,
        '''

        input_dict = dict(
            sequence = info['sequence'],
            frame_id = info['frame_id'],
            img = info['img_2_path'],
        )
        
    
        return input_dict


    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:

            img_base_path = os.path.join(self.data_root, "sequences", sequence)   

            id_base_path = os.path.join(self.data_root, "sequences", sequence, 'voxels', '*.bin')

            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]

                # image
                img_2_path = os.path.join(img_base_path, 'image_2', img_id + '.png')
                
                scans.append(
                    {   
                        "sequence": sequence,
                        "frame_id": img_id,
                        "img_2_path": img_2_path,
                    })
                
        return scans  # return to self.data_infos


class SemanticKITTIBlipDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        batch_size=1,
        num_workers=4,
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        self.predict_dataset = SemanticKITTIBlipDataset(self.data_root, 'predict')
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True)


if __name__ == '__main__':
    s = SemanticKITTIBlipDataset(
        data_root='/u/home/caoh/datasets/SemanticKITTI/dataset',
        split='predict',
    )

    print(s[0])
    #print(s[0]['gt_occ'])
    #print(s[0]['gt_occ_2'])

    # python /u/home/caoh/projects/MA_Jiachen/3DPNA/projects/datasets/semantic_kitti_blip2.py