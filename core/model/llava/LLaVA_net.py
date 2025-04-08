import os

import pytorch_lightning as pl

from configs import CONF

import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

class LLaVAModule(pl.LightningModule):
    def __init__(
        self,
        model_path,
        model_base=None,
        load_4bit=False,
        load_8bit=False,
        conv_mode=None,
        temperature=0.2,
        top_p=None,
        num_beams=1,
        max_new_tokens=512,
        device='cuda'
        ):

        super(LLaVAModule, self).__init__()

        self.model_path = model_path
        self.model_base = model_base
        self.load_4bit = load_4bit
        self.load_8bit = load_8bit
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        self.text_dir = os.path.join(CONF.PATH.DATA_TEXT, 'LLaVA')
    
    def setup(self, device = 'cuda', stage=None):

        # Model
        disable_torch_init()

        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            self.model_path,self.model_base, model_name, self.load_8bit, self.load_4bit, device=device
        )

        qs = 'Write a strictly factual and detailed description of the image.'
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        
        if conv_mode is not None and conv_mode != conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, conv_mode, conv_mode
                )
            )
        else:
            conv_mode = conv_mode

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        self.prompt = conv.get_prompt()


    def get_text_output(self, image_file):
        image = Image.open(image_file).convert("RGB")
        images = [image]
        image_sizes = [image.size]

        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(self.prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        output_ids = self.model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        outputs = ' '.join(outputs.split())

        return outputs
    

    def forward(self, batch):
        text_dir_seq = os.path.join(self.text_dir, batch['sequence'])
        output_file = os.path.join(text_dir_seq, batch['frame_id'])

        os.makedirs(text_dir_seq, exist_ok=True)

        text_output = self.get_text_output(batch['img'])

        with open(output_file + ".txt", "w", encoding="utf-8") as f:
            f.write(text_output)
