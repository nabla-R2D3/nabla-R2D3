import os
from PIL import Image
import io
import numpy as np
import time
import requests

import torch
import torch.distributed as dist

from scripts.distributed import get_local_rank


short_names = {
    "jpeg_incompressibility": "incomp",
    "jpeg_compressibility": "comp",
    "aesthetic_score": "aes",
    "imagereward": "imgr",
    "hpscore": "hps",
    "llava_strict_satisfaction": "llava_strict",
    "llava_bertscore": "llava",
    "pickscore": "pick"
}
use_prompt = {
    "jpeg_incompressibility": False,
    "jpeg_compressibility": False,
    "aesthetic_score": False,
    "imagereward": True,
    "hpscore": True,
    "pickscore": True,
}

def jpeg_incompressibility(dtype=torch.float32, device="cuda"):
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        sizes = np.array(sizes)
        return torch.from_numpy(sizes).cuda(), {}

    return _fn


def jpeg_compressibility(dtype=torch.float32, device="cuda"):
    jpeg_fn = jpeg_incompressibility(dtype, device)

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score(dtype=torch.float32, device="cuda", distributed=True):
    from alignment.aesthetic_scorer import AestheticScorer
    # why cuda() doesn't cause a bug?
    scorer = AestheticScorer(dtype=torch.float32, distributed=distributed).cuda() # ignore type;

    # input can be 3*256*256 or 3*1024*1024
    # @torch.no_grad() # original AestheticScorer already has no_grad()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            # images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            pass # assume float tensor in [0, 1]
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
# For HPSv2 reward
# https://github.com/tgxs002/HPSv2/blob/master/hpsv2/img_score.py
def hpscore(dtype=torch.float32, device=torch.device('cuda')):
    os.environ["HPS_ROOT"] = "../HPSv2"
    import huggingface_hub
    import torchvision.transforms.functional as F
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.utils import root_path, hps_version_map
    from hpsv2.src.open_clip.transform import MaskAwareNormalize, ResizeMaxSize

    hps_version = "v2.1"
    model_dict = {}
    # def initialize_model():
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            # device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
    pass
        # model_dict['model'] = model
        # model_dict['preprocess_val'] = preprocess_val

    # initialize_model()
    # model = model_dict['model']
    # preprocess_val = model_dict['preprocess_val']
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()

    def _fn(images, prompts, metadata):
        # result = []
        # for img, prompt in zip(images, prompts):
        #     # Load your image and prompt
        #     # Process the image
        #     img = F.to_pil_image(img)
        #     assert isinstance(img, Image.Image)
        #     image = preprocess_val(img).unsqueeze(0).to(device=device, non_blocking=True)
        #     text = tokenizer([prompt]).to(device=device, non_blocking=True)
        #
        #     with torch.cuda.amp.autocast():
        #         outputs = model(image, text)
        #         image_features, text_features = outputs["image_features"], outputs["text_features"]
        #         logits_per_image = image_features @ text_features.T
        #         hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        #     result.append(hps_score[0])
        # return torch.tensor(result, device=device), {}


        image_size = model.visual.image_size[0]
        transforms = Compose([
            ResizeMaxSize(image_size, fill=0), # resize to 224x224
            MaskAwareNormalize(mean=model.visual.image_mean, std=model.visual.image_std),
        ])

        # these are not numerically identical, because
        # F.to_tensor(F.to_pil_image(img)) != img
        # due to RGB round up (in PIL it is 0~255 integer)

        # images = torch.stack([preprocess_val(F.to_pil_image(img)) for img in images.float()]).to(device)
        images = torch.stack([transforms(img) for img in images])
        texts = tokenizer(prompts).to(device=device, non_blocking=True).detach() ### 
        # texts = tokenizer(prompts).to(device=device, non_blocking=True) 

        with torch.cuda.amp.autocast(dtype=dtype):
            outputs = model(images, texts)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T # (bs, bs)
            hps_score = torch.diagonal(logits_per_image) # (bs,)

        return hps_score, {}

    return _fn

