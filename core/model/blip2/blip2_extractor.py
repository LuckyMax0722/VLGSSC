import os

import pytorch_lightning as pl
import numpy as np

from PIL import Image

from lavis.models import load_model_and_preprocess
from configs import CONF

class Blip2Extractor(pl.LightningModule):
    def __init__(
        self,
        feat_extractor_model,
        ):

        super(Blip2Extractor, self).__init__()

        self.feat_extractor_model = feat_extractor_model
        self.text_feat_dir = os.path.join(CONF.PATH.DATA_TEXT_FEAT, feat_extractor_model)
    
    def setup(self, stage=None):

        if self.feat_extractor_model == 'BLIP2':
            name = "blip2_feature_extractor"
            model_type="pretrain"

        elif self.feat_extractor_model == 'CLIP':
            name="clip_feature_extractor"
            model_type="ViT-B-16"

        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=name, 
            model_type=model_type, 
            is_eval=True, 
            device=self.device)

    def get_feat_output(self, image, text):
        image = Image.open(image).convert("RGB")

        # prepare the image
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)

        # Extract Feature
        if self.feat_extractor_model == 'BLIP2':
            text_input = self.txt_processors["eval"](text)
            sample = {"image": image, "text_input": [text_input]}

            features_image = self.model.extract_features(sample, mode="image")
            features_text = self.model.extract_features(sample, mode="text")

            features_image = features_image.image_embeds_proj
            features_text = features_text.text_embeds_proj

        elif self.feat_extractor_model == 'CLIP':
            sample = {"image": image, "text_input": text}

            features =  self.model.extract_features(sample)

            features_image = features.image_embeds_proj
            features_text = features.text_embeds_proj


        return features_image, features_text

    def predict_step(self, batch, batch_idx):
        text_feat_dir_seq = os.path.join(self.text_feat_dir, batch['sequence'][0])
        output_file = os.path.join(text_feat_dir_seq, batch['frame_id'][0])

        os.makedirs(text_feat_dir_seq, exist_ok=True)
        
        features_image, features_text = self.get_feat_output(batch['img'][0], batch['text'][0])

        if self.feat_extractor_model == 'BLIP2':

            np.save(output_file + ".npy", features_text.squeeze(0).cpu().numpy())

        elif self.feat_extractor_model == 'CLIP':

            np.save(output_file + ".npy", features_text.squeeze(0).cpu().numpy())
