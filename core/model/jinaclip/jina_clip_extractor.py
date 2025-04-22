import os

import pytorch_lightning as pl
import numpy as np

from configs import CONF

import torch

from transformers import AutoModel

import torch

class JinaCLIPExtractor(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path = 'jinaai/jina-clip-v2',
        truncate_dim=512
        ):

        super(JinaCLIPExtractor, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.truncate_dim = truncate_dim
        self.text_feat_dir = os.path.join(CONF.PATH.DATA_TEXT_FEAT, f'JinaCLIP_{truncate_dim}')

    def setup(self, stage=None):
        self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True)

    def get_feat_output(self, text):
        text_features = self.model.encode_text([text], truncate_dim=self.truncate_dim)

        return text_features

    def predict_step(self, batch, batch_idx):
        text_feat_dir_seq = os.path.join(self.text_feat_dir, batch['sequence'][0])
        output_file = os.path.join(text_feat_dir_seq, batch['frame_id'][0])

        os.makedirs(text_feat_dir_seq, exist_ok=True)
        
        features_text = self.get_feat_output(batch['text'][0])
        np.save(output_file + ".npy", features_text[0])
