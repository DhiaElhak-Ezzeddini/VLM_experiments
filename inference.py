from PIL import Image
import torch
import fire 
from processing_inputs import PaliGemmaProcessor
from gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


def main(
        model_path:str,
        prompt:str,
        image_path:str,
        max_tokens_to_generate:int,
        temperature:float=0.8,
        top_p:float=0.9,
        do_sample:bool=False,
        only_cpu:bool=False
        ): 
    device = "cpu"

    if not only_cpu : 
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    print(f"Device in use: {device}")

    