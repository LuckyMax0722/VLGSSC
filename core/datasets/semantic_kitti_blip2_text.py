import os
import glob
import torch

from torch.utils.data import Dataset

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader


class SemanticKITTIBlipTextDataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
    ):
        self.splits = {
            "predict": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
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
        
        # load text
        input_dict['text'] = self.get_text_info(index, key='text_path')

        return input_dict

    def get_text_info(self, index, key='text_path'):
        info = self.data_infos[index][key]
        
        return self.load_text(info)

    def load_text(self, text_filename):
        with open(text_filename, 'r', encoding='utf-8') as f:
            sentence = f.read()
        
        return sentence

    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:

            img_base_path = os.path.join(self.data_root, "sequences", sequence)   

            id_base_path = os.path.join(self.data_root, "sequences", sequence, 'voxels', '*.bin')

            text_base_path = os.path.join(self.data_root, "text", 'Blip2', sequence)

            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]

                # image
                img_2_path = os.path.join(img_base_path, 'image_2', img_id + '.png')
                
                # text
                text_path = os.path.join(text_base_path, img_id + '.txt')

                scans.append(
                    {   
                        "sequence": sequence,
                        "frame_id": img_id,
                        "img_2_path": img_2_path,
                        "text_path": text_path
                    })
                
        return scans  # return to self.data_infos


class SemanticKITTIBlipTextDataModule(pl.LightningDataModule):
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
        self.predict_dataset = SemanticKITTIBlipTextDataset(self.data_root, 'predict')
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True)


if __name__ == '__main__':
    s = SemanticKITTIBlipTextDataset(
        data_root='/u/home/caoh/datasets/SemanticKITTI/dataset',
        split='predict',
    )

    print(s[0]['img'])
    print(s[0]['text'])
    #print(s[0]['gt_occ'])
    #print(s[0]['gt_occ_2'])

    # python /u/home/caoh/projects/MA_Jiachen/VLGSSC/core/datasets/semantic_kitti_blip2_text.py