from typing import Dict, List, Tuple, Optional,Union, Iterable
from cv2 import normalize
from torchvision import transforms
import numpy as np 
import torch 
from PIL import Image

IMAGE_MEAN = [0.5,0.5,0.5]
IMAGE_STD  = [0.5,0.5,0.5]

normalize_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
])

def add_image_tokens_to_prompt(
        prefix_prompt: str,
        bos_token: str,
        image_seq_len: int,
        image_token: str
        ) -> str:
    # Add the image token to the prompt
    prompt = f"{image_token*image_seq_len}{bos_token}{prefix_prompt}\n"
    return prompt


def resize(
        image:Image, 
        size:Tuple[int,int], 
        resample:Image.Resampling,
        reducing_gap:Optional[int]=None
        ) -> np.ndarray:
    h,w = size
    resized_image = image.resize((h,w), resample=resample, reducing_gap=reducing_gap)

    return resized_image

def rescale(
        image:Image, 
        scale:float,
        dtype:np.dtype=np.float32
        ) -> np.ndarray:
    rescaled_image = image*scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def process_images(
        images: List[Image.Image], 
        size: Dict[int, int], 
        resample: int, 
        rescale_factor: float, 
        image_mean: Optional[Union[float,List[float]]], 
        image_std: Optional[Union[float,List[float]]]) -> List[np.ndarray]:
    h, w = size[0],size[1]
    images = [
        resize(image=image, size=(h,w), resample=resample) for image in images
    ]
    images = [np.array(image) for image in images]

    images  = [rescale(image, scale=rescale_factor) for image in images]

    images = [normalize_image(image) for image in images] ## now images are tensors and normalized to have mean 0.5 and std 0.5
    #print(f"Processed images shape (C, H, W): {images[0].shape}")
    images = [image.detach().cpu().numpy() for image in images]  # Convert to numpy arrays
    return images 


class PaliGemmaProcessor : 
    IMAGE_TOKEN = "<image>"
    def __init__(self,tokenizer, num_images_tokens:int, image_size:int):
        super().__init__()  
        self.image_size = image_size    
        self.image_seq_len = num_images_tokens


        tokens_to_add = {"additional_special_tokens":[self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] ## for detection ==> coordination of bounding boxes
        EXTRA_TOKENS += [
            f"<seg{i:04d}>" for i in range(128)
        ] # for segmentation ==> pixel-wise classification
        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(self, text:List[str], images: List[Image.Image], padding:str="longest", truncation:bool=True)-> Dict :
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images and {len(text)} texts, but expected 1 of each."
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor = 1/255.0,
            image_mean = IMAGE_MEAN,
            image_std = IMAGE_STD,
        )


        pixel_values = np.stack(pixel_values, axis=0)  # Shape: (batch_size, num_channels, height, width)
        pixel_values = torch.tensor(pixel_values)

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_len,
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        ]
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {'pixel_values': pixel_values,**inputs}
        return return_data 
