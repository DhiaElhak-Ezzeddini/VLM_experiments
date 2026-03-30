from gemma import PaliGemmaConfig,PaliGemmaForConditionalGeneration
import torch
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os 


def load_hf_model(model_path:str,device:str)->Tuple[PaliGemmaForConditionalGeneration,AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side="right")  
    assert tokenizer.padding_side == "right" 

    safetensors_files = glob.glob(os.path.join(model_path,"*.safetensors"))
    tensors = {}

    for file in safetensors_files : 
        with safe_open(file, framework="pt", device="cpu") as f: ## 
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # model config : 
    with open(os.path.join(model_path,"config.json"),"r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    model = PaliGemmaForConditionalGeneration(config)
    #model = model.half() ## trying to avoid OOM errors by loading the model directly in float16,
    model = model.to(device) 
    model.load_state_dict(tensors,strict=False) ## strict = False since we might have some extra tensors in the safetensors files that are not part of the model's state dict (e.g. optimizer states, training states, etc.)
    model.tie_weights()

    return (model, tokenizer)