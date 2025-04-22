import os

import pytorch_lightning as pl
import numpy as np

from configs import CONF

import torch

from transformers import AutoModel, CLIPTokenizer

import torch

class EVACLIPExtractor(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path = 'BAAI/EVA-CLIP-8B'
        ):

        super(EVACLIPExtractor, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.text_feat_dir = os.path.join(CONF.PATH.DATA_TEXT_FEAT, 'EVACLIP')

    def setup(self, stage=None):

        self.model = AutoModel.from_pretrained(
                self.model_name_or_path, 
                torch_dtype=torch.float16,
                trust_remote_code=True).to(self.device)


    def get_feat_output(self, text):
        tokenizer = CLIPTokenizer.from_pretrained(self.model_name_or_path)
        input_ids = tokenizer([text],  return_tensors="pt", padding=True).input_ids.to(self.device)

        with torch.cuda.amp.autocast():
            text_features = self.model.encode_text(input_ids)

        return text_features

    def predict_step(self, batch, batch_idx):
        text_feat_dir_seq = os.path.join(self.text_feat_dir, batch['sequence'][0])
        output_file = os.path.join(text_feat_dir_seq, batch['frame_id'][0])

        os.makedirs(text_feat_dir_seq, exist_ok=True)
        
        features_text = self.get_feat_output(batch['text'][0])

        print(features_text.size())
        #np.save(output_file + ".npy", features_text.squeeze(0).cpu().numpy())
