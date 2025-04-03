import os

import pytorch_lightning as pl
import numpy as np

from PIL import Image

from lavis.models import load_model_and_preprocess
from configs import CONF

class Blip2Module(pl.LightningModule):
    def __init__(
        self,
        ):

        super(Blip2Module, self).__init__()

        self.prompt = 'Write a strictly factual and detailed description of the image.'
        self.text_dir = os.path.join(CONF.PATH.DATA_TEXT, 'Blip2')
    
    def setup(self, stage=None):
        self.model_IBLIP, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_vicuna_instruct", 
            model_type="vicuna7b", 
            is_eval=True, 
            device=self.device
        )

        self.model_BLIP_Feat, _, self.txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor", 
            model_type="pretrain", 
            is_eval=True, 
            device=self.device
        )



    def get_text_output(self, raw_image):
        raw_image = Image.open(raw_image).convert("RGB")

        # prepare the image
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)

        out = self.model_IBLIP.generate({"image": image, "prompt" : self.prompt}, use_nucleus_sampling=True, top_p=0.9, temperature=1)[0]
        
        return image, out
        

    def get_feat_output(self, image, text):
        # Extract Feature
        text_input = self.txt_processors["eval"](text)
        sample = {"image": image, "text_input": [text_input]}

        features_image = self.model_BLIP_Feat.extract_features(sample, mode="image")
        features_text = self.model_BLIP_Feat.extract_features(sample, mode="text")

        return features_image, features_text

    def predict_step(self, batch, batch_idx):
        text_dir_seq = os.path.join(self.text_dir, batch['sequence'][0])
        output_file = os.path.join(text_dir_seq, batch['frame_id'][0])

        os.makedirs(text_dir_seq, exist_ok=True)

        image, text_output = self.get_text_output(batch['img'][0])

        with open(output_file + ".txt", "w", encoding="utf-8") as f:
            f.write(text_output)
        
        features_image, features_text = self.get_feat_output(image, text_output)

        np.save(output_file + ".npy", features_text.text_embeds_proj.squeeze(0).cpu().numpy())
        
