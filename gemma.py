import torch 
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math 
from SigLip import SiglipVisionConfig, SiglipVisionModel



class KVCache(): 
    def __init__(self)-> None : 
        self.key_cache : List[torch.Tensor] = []
        self.value_cache : List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0 :
            return 0
        else : 
            return self.key_cache[-1].shape[2] ## the sequence length dimension of the last cached key tensor

    def update(self,key: torch.Tensor, value: torch.Tensor,layer_idx:int) -> None:
        if len(self.key_cache) <= layer_idx : 
            # if we never added anything to the kv_cache of this layer , we should create it
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            # otherwise we just concatenate the new key and value tensors to the existing ones 
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=-2) ## concatenate on the sequence length dimension
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
class Gemmaconfig():
    def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim=256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            pad_token_id=None,
            **kwargs
    ):
        super().__init__()
        self.vocab_size=vocab_size
        self.max_position_embeddings=max_position_embeddings
        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.num_key_value_heads=num_key_value_heads
        self.head_dim=head_dim
        self.rms_norm_eps=rms_norm_eps
        self.rope_theta=rope_theta
        self.attention_bias=attention_bias
        self.attention_dropout=attention_dropout
        self.pad_token_id=pad_token_id

class PaliGemmaConfig():
    def __init__(
            self,
            vision_config=None,
            text_config=None,
            ignore_index=-100,
            image_token_index=256000,
            vocab_size=257152,
            projection_dim=2048,
            hidden_size=2048,
            pad_token_id=None,
            **kwargs
    ):
        super().__init__()
        self.text_config = text_config
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size  
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config) if vision_config is not None else SiglipVisionConfig()
        self.text_config = Gemmaconfig(**text_config,pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size
        self.text_config.num_image_tokens = (self.vision_config.image_size//self.vision_config.patch_size)**2
        self.vision_config.projection_dim = self.projection_dim 

class PaliGemmaMultiModalProjector(nn.Module) :
    def __init__(self,config:PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim,bias=True)

    def forward(self,image_features): 
        hidden_states = self.linear(image_features)
        return hidden_states


class GemmaRMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-6):
        super().__init__()
        self.eps=eps
        self.weight = nn.Parameter(torch.zeros(dim)) ## initialize the learnable parameter gi
    def _norm(self,x):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps) ## 1/sqrt(var + eps)

    def forward(self,x):
        output = self._norm(x.float())
        output = output * (1.0+self.weight)
        return output.type_as(x) 
    


class GemmaMLP(nn.Module):
    def __init__(self,config:Gemmaconfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.inter_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.inter_size,bias=False)
        self.up_proj   = nn.Linear(self.hidden_size, self.inter_size,bias=False)
        self.down_proj = nn.Linear(self.inter_size, self.hidden_size,bias=False) 
    
    def forward(self,x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x),approximte="tanh") * self.up_proj(x))



def repeat_kv(hidden_states:torch.Tensor, num_repeats:int) -> torch.Tensor:
    """
    Repeat the input tensor along the sequence length dimension.
    """
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.size()
    if num_repeats == 1:
        return hidden_states
    hidden_states = hidden_states[:,:,None,:,:].expand(batch_size, num_kv_heads, num_repeats, seq_len, head_dim)
    return hidden_states.reshape(batch_size, num_kv_heads * num_repeats, seq_len, head_dim)

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self,dim, max_position_embeddings=2048, base=1000, device = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0/(self.base**(torch.arange(0,self.dim,2,dtype=torch.int64).float()/self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq,persistent=False)
    
    @torch.no_grad()
    def forward(self,x,position_ids,seq_len=None):
        # x : [B, num_attention_heads, seq_len, head_dim]
        self.inv_freq.to(x.device)
        # inv_freq_expanded : [B,head_dim//2,1]
        inv_freq_expanded = self.inv_freq[None,:,None].float().expand(position_ids.shape[0],-1,-1)
        # position_ids_expanded : [B,1,seq_len]
        position_ids_expanded = position_ids[:,None,:].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type=device_type,enabled=True):
            # freqs : [B,seq_len,head_dim//2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,2) ## calculate m*theta (arg of sin and cos)
            emb = torch.cat((freqs,freqs),dim=-1)
            # cos, sin : [B, seq_len, head_dim]
            cos = emb.cos() # calculate cosine
            sin = emb.sin() # calculate sine

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    x1 = x[...,:x.shape[-1]//2]
    x2 = x[...,x.shape[-1]//2:]
    
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q,k,cos,sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GemmaAttention(nn.Module):
    def __init__(self,config:Gemmaconfig,layer_idx:Optional[int]=None):    
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # Number of heads = 8
        # hidden_size = 1024
        # head_dim = 1024/8 = 128
        # Wq = [1024,8*128] = [1024,1024]
        # Wk = [1024,1*128] = [1024,128]
        # Wv = [1024,1*128] = [1024,128]
        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.out_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            self.max_position_embeddings,
            self.rope_theta
        )
    def forward(self,
                hidden_states:torch.Tensor,
                attention_mask:Optional[torch.Tensor],
                position_ids:Optional[torch.LongTensor],
                kv_cache:Optional[KVCache]=None,
                **kwargs) -> Tuple[torch.Tensor,Optional[torch.Tensor],Optional[Tuple[torch.Tensor]]]:
         
        bsz , q_len, _ = hidden_states.size() # [batch_size,seq_len,hidden_size]
        # [B, Seq_len, Num_Heads_Q*Head_Dim]
        query_states = self.q_proj(hidden_states)
        # [B, Seq_len, Num_Heads_KV*Head_Dim]
        key_states = self.k_proj(hidden_states)
        # [B, Seq_len, Num_Heads_KV*Head_Dim]
        value_states = self.v_proj(hidden_states)
        # [B, Num_heads_Q, Seq_len, Head_Dim]
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1,2)
        # [B, Num_heads_KV, Seq_len, Head_Dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        # [B, Num_heads_KV, Seq_len, Head_Dim]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        # [B, Seq_len, head_dim],[B, Seq_len, head_dim], 
        cos,sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [B, Num_heads_Q, Seq_len, Head_Dim], [B, Num_heads_KV, Seq_len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None :
            key_states, value_states = kv_cache.update(key_states, value_states,self.layer_idx)

        ## same num of heads of the query
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        ## attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None, "attention_mask cannot be None"
        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_attention_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output size mismatch: expected {(bsz, self.num_attention_heads, q_len, self.head_dim)}, got {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.view(bsz,q_len,-1) ## conctenate all heads together
        attn_output = self.out_proj(attn_output) # mixing the results of independent heads
        

        return attn_output, attn_weights
    




class GemmaDecoderLayer(nn.Module):
    def __init__(self,config:Gemmaconfig,layer_idx:int):
    
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config,layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_rms_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_rms_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    
    def forward(self,hidden_states,attention_mask,position_ids,kv_cache=None):
        residual = hidden_states
        hidden_states = self.input_rms_norm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids,
            kv_cache
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_rms_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self,config:Gemmaconfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config,layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
            self,
            attention_mask,
            position_ids,
            inputs_embeds,
            kv_cache=None
    ):
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        for layer in self.layers: 
            hidden_states = layer(hidden_states,attention_mask,position_ids,kv_cache)

        hidden_states = self.norm(hidden_states)
        return hidden_states
    


class GemmaForCausalLM(nn.Module):
    def __init__(self,config):
        self.config = config
        self.model  = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
            
        
    def forward(self, attention_mask, position_ids, inputs_embeds, kv_cache=None):
        outputs = self.model(
            attention_mask,
            position_ids,
            inputs_embeds,
            kv_cache
        )
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            'logits':logits,
        }

        if kv_cache is not None :
            return_data['kv_cache'] = kv_cache
        
        return return_data



class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multimodal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1


    def tie_weights(self):
        ## embedding layer and final linear of the decoder are doing weight tying because they share the same parameters 
        ## the embedding layer is doing the inverse of the final linear layer so they can share parameters ==> reduce the number of parameters
        return self.language_model.tie_weights()
    
    def _merge_inputs_ids_with_image_features(self,image_features:torch.tensor,input_embeds:torch.tensor,input_ids:torch.tensor,attention_mask:torch.Tensor= None,kv_cache: Optional[KVCache] = None):
        _,_,embed_dim = image_features.shape
        batch_size, seq_len, _ = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)
        
        final_embedding = torch.zeros(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
        
        text_mask  = (input_ids != self.config.image_token_index) & (input_ids != self.config.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask   = input_ids == self.config.pad_token_id    
        
        text_mask_expanded  = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded   = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embedding = torch.where(text_mask_expanded,input_embeds,final_embedding)
        final_embedding = final_embedding.masked_scatter(image_mask_expanded,scaled_image_features )
        final_embedding = torch.where(pad_mask_expanded,torch.zeros_like(final_embedding),final_embedding)

        ## Attention mask and KV-Cache
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items()==0 : ## pre-filling phase
            causal_mask = torch.full(
                (batch_size, q_len, q_len), ## we should consider all the tokens of the user's prompt since we will pass      
                fill_value=0,               ## the whole prompt the first time and create its corresponding KV Cache
                dtype=dtype,                ## ==> (q_len, q_len) matrices K and V of the input prompt
                device=device
            )
        else : 
            ## here we are generating tokens, so the query must be one token only (1, 1, hidden_size)
            assert q_len ==1 
            kv_len = kv_cache.num_items() + q_len 
            ## no need for masking since this query will attend all the previous tokens resulting from softmax(k*V.T)
            causal_mask = torch.full(
                (batch_size, kv_len), ## mask for the last generated token ==> (1, kv_len)
                fill_value=0,
                dtype=dtype,
                device=device
            )
            ## here by choice the VLM is not causal in the sense that each token can attend to all tokens, even the future ones
            # but this is limited to the prefilling process, as if starting from the first generated token we will be using masking so that the tokens 
            # of the original query will not attend the generated tokens
            # and since we are generating one token at the time that should be attending all the previous tokens we don't need masking !!
            ## during training we must have a mask since the model will be generating all the tokens at the same time and we don't want the model to attend to the future tokens
            ## but during inference since we have the KV Cache we do not need to mask the future tokens
            
        ## adding the head dimension 
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items()>0 : 
            # the position of the query is the last one 
            position_ids = attention_mask.cumsum(-1)[:,-1]
            if position_ids.dim() == 1 : 
                position_ids = position_ids.unsqueeze(0) 
        else : 
            position_ids =(attention_mask.cumsum(-1).masked_fill_((attention_mask==0), 1)).to(device)

        return final_embedding, causal_mask, position_ids 



    def forward(
            self, 
            input_ids:torch.LongTensor,
            pixel_values:torch.FloatTensor,
            attention_mask:Optional[torch.Tensor] = None,
            kv_cache:Optional[KVCache] = None,
    )-> Tuple:
        
        assert torch.all(attention_mask==1), "The input cannot be padded"
        # (B, Seq_len, hidden_size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        # (B, num_patches, embed_dim)
        selected_image_features = self.vision_tower(pixel_values.to(input_embeds.dtype))
        image_features = self.multimodal_projector(selected_image_features) ## ==> (B, num_patches, hidden_size)
        input_embeds, attention_mask, position_ids = self._merge_inputs_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask)


        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        return outputs