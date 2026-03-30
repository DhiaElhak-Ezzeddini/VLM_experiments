import torch
import torch.nn as nn

from typing import Tuple, Optional

class SiglipVisionConfig:
    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens:int=None,
            **kwargs
            ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipMLP(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size,config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size,config.hidden_size)
    
    def forward(self,hidden_states:torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states,approximate='tanh')
        hidden_states = self.fc2(hidden_states)

        return hidden_states
    

class SiglipAttention(nn.Module):
    """ Multi Head attentio - without masking """

    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim,self.embed_dim) 

    def forward(self,hidden_states:torch.Tensor) -> Tuple[torch.Tensor,Optional[torch.Tensor]]:

        batch_size , seq_len, _ = hidden_states.size()
        
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query_states = query.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1, 2)
        key_states = key.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1, 2)
        value_states = value.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(f"Attention output shape mismatch: expected {(batch_size, self.num_heads, seq_len, seq_len)}, got {attn_weights.size()}")

        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(f"Attention output shape mismatch: expected {(batch_size, self.num_heads, seq_len, self.head_dim)}, got {attn_output.size()}")
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipEncoder(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,input_embeds:torch.Tensor):
        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

        
class SiglipEncoderLayer(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 =  nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
         
    def forward(self,hidden_states:torch.Tensor):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_output,_ = self.self_attn(hidden_states)
        hidden_states = residual + attn_output
        mlp_residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = mlp_residual + mlp_output

        return hidden_states



class SiglipVisionEmbeddings(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config =config
        self.embed_dim = self.config.hidden_size
        self.image_size = self.config.image_size
        self.patch_size = self.config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=self.config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid'## no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions,self.embed_dim)
        self.register_buffer(
            'position_ids',
            torch.arange(self.num_positions).expand((1,-1)),
            persistent=False,
        )

    def forward(self,pixel_values):
        #print(f"Pixel values shape: {pixel_values.shape}")
        patch_embeddings = self.patch_embedding(pixel_values) ## (B,embed_dim,num_patches_h,num_patches_w)
        patch_embeddings = patch_embeddings.flatten(2)
        patch_embeddings = patch_embeddings.transpose(1,2) ## (B,num_patches,embed_dim)
        embeddings = patch_embeddings + self.position_embedding(self.position_ids)

        return embeddings

class SiglipEncoder(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self,embeddings:torch.Tensor):
        hidden_states = embeddings
        for layer in self.layers : 
            hidden_states = layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self,config:SiglipVisionConfig): 
        super().__init__()
        self.config = config
        embed_dim = self.config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=self.config.layer_norm_eps)

    def forward(self,pixel_values:torch.Tensor)->torch.Tensor:
        embeddings = self.embeddings(pixel_values)
        encoder_output = self.encoder(embeddings)
        encoder_output = self.post_layernorm(encoder_output)

        return encoder_output

class SiglipVisionModel(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_vals:torch.Tensor)->torch.Tensor:
        ## (B,C,H,W) --> (B,num_patches,embed_dim)
        return self.vision_model(pixel_values = pixel_vals)
    
    