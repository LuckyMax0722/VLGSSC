import torch
from PIL import Image
import os
from lavis.models import load_model_and_preprocess

# setup device to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# CUDA_VISIBLE_DEVICES=5 python /u/home/caoh/projects/MA_Jiachen/VLGSSC/core/model/blip2/predict.py


# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device)


prompt = 'Write a strictly factual and detailed description of the image.'

demo_0 = '/u/home/caoh/projects/MA_Jiachen/3DPNA/demo/demo_0.png'
demo_1 = '/u/home/caoh/projects/MA_Jiachen/3DPNA/demo/demo_1.png'

demo = [demo_0, demo_1]
text_output = []


for demo_image in demo:
    raw_image = Image.open(demo_image).convert("RGB")

    # prepare the image
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    out = model.generate({"image": image, "prompt" : prompt}, use_nucleus_sampling=True, top_p=0.9, temperature=1)[0]
    text_output.append(out)
    print(out)


# model, _, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

# for out in text_output:
#     # Extract Feature
#     text_input = txt_processors["eval"](out)
#     sample = {"image": image, "text_input": [text_input]}

#     features_image = model.extract_features(sample, mode="image")
#     features_text = model.extract_features(sample, mode="text")

#     print(features_image.image_embeds.shape)
#     # torch.Size([1, 32, 768])
#     print(features_text.text_embeds.shape)
#     # torch.Size([1, 12, 768])

#     # low-dimensional projected features
#     print(features_image.image_embeds_proj.shape)
#     # torch.Size([1, 32, 256])
#     print(features_text.text_embeds_proj.shape)
#     # torch.Size([1, 12, 256])
#     similarity = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:,0,:].t()).max()
#     print(similarity)
#     # tensor([[0.3642]])


'''
The image features a street scene with several cars parked in front of a large white building. 
There are two cars parked on the left side of the street, and one car parked on the right side of the street. 
In addition to these vehicles, there is another car parked further down the street. 
A person can also be seen standing on the sidewalk next to the building.The image depicts a narrow city street lined with parked cars on both sides. 

There are several cars parked along the street, including a black car in the middle of the road and another black car near the end of the street. 
In addition to the parked cars, there are two motorcycles visible in the scene, one near the beginning of the street and another near the end of the street. 
The street is lined with trees and buildings, adding to the urban atmosphere.
'''


