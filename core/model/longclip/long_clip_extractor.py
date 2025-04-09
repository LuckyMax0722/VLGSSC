import os

import pytorch_lightning as pl
import numpy as np

from configs import CONF

from core.model.longclip.model import longclip
import torch
from huggingface_hub import snapshot_download


class LongCLIPExtractor(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path = 'BeichenZhang/LongCLIP-L'
        ):

        super(LongCLIPExtractor, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.text_feat_dir = os.path.join(CONF.PATH.DATA_TEXT_FEAT, 'LongCLIP')
    
    def setup(self, stage=None):

        cache_dir = snapshot_download(repo_id=self.model_name_or_path)
        model_file = os.path.join(cache_dir, "longclip-L.pt")

        self.model, self.preprocess = longclip.load(model_file, device=self.device)

    def get_feat_output(self, text):
        text = longclip.tokenize([text]).to(self.device)
        text_features = self.model.encode_text(text)
        
        return text_features

    def predict_step(self, batch, batch_idx):
        text_feat_dir_seq = os.path.join(self.text_feat_dir, batch['sequence'][0])
        output_file = os.path.join(text_feat_dir_seq, batch['frame_id'][0])

        os.makedirs(text_feat_dir_seq, exist_ok=True)
        
        features_text = self.get_feat_output(batch['text'][0])

        np.save(output_file + ".npy", features_text.squeeze(0).cpu().numpy())
