from PIL import Image
import torch
import fire
from processing_inputs import PaliGemmaProcessor
from gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


    
def _sample_top_p(probs:torch.Tensor, top_p:float):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_cumsum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_cumsum - probs_sort > top_p
    probs_sort[mask] = 0.0

    probs_sort.div_(probs_sort.sum(dim=-1))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, dim=-1, index=next_token)
    return next_token

def move_inputs_to_device(inputs:dict, device:str):
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

def get_model_inputs(processor:PaliGemmaProcessor, prompt:str, image_path:str, device:str):
    image = Image.open(image_path).convert("RGB")
    images = [image]
    prompts = [prompt]

    model_inputs = processor(text=prompts, images=images)

    model_inputs = move_inputs_to_device(model_inputs, device)

    return model_inputs


def test_inference(
        model:PaliGemmaForConditionalGeneration,
        processor:PaliGemmaProcessor,
        device:str,
        prompt:str,
        image_path:str,
        max_tokens_to_generate:int,
        temperature:float,
        top_p:float,
        do_sample:bool
):
    model_inputs = get_model_inputs(processor, prompt, image_path,device)
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    pixel_values = model_inputs['pixel_values'] 

    kv_cache = KVCache()
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    """
    The first iteration is called "pre-filling", since we are passing all the prompt as input
    so the KV cache will be filled with the keys and values for the entire prompt (all the tokens of the initial prompt)
    """
    for _ in range(max_tokens_to_generate):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else : 
            next_token = torch.argmax(next_token_logits, dim=-1,keepdim=True)
        assert next_token.shape == (1,1) , f"Expected next_token to have shape (1,1), but got {next_token.shape}"
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)
        #if next_token == stop_token:
        #    break
        """ 
         starting from the second iteration, after the pre-filling step : 
         as we are using KV Cache, we use the last generated token as    the next input, so
         only one single token will be presented as input and only one Query, one key and one value will be generated
         the generated value vector and key will be stored in the KV cache -- appended to the buffers of K and V
         and than attention will be computed using these K and V buffers alongside with the computed query vector
        """
        input_ids = next_token.unsqueeze(-1)

        attention_mask = torch.cat(
            [attention_mask, torch.ones(1,1, device=input_ids.device)], dim=1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(prompt + decoded)
    

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
    print(f"Loading the model")


    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer,num_image_tokens,image_size)


    print(f"Running inference...")

    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample
        )




if __name__ == "__main__":
    fire.Fire(main)