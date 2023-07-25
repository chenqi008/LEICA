from PIL import Image
# import requests
import numpy as np
import json
import os
from tqdm import tqdm

# from io import BytesIO
# import base64
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPOutput, CLIPTextTransformer, CLIPMLP, CLIPPreTrainedModel
from typing import Optional, Tuple, Union
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
# from transformers.configuration_utils import PretrainedConfig

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
# from transformers.modeling_utils import PreTrainedModel

from clip.visualizer import visualize_attention
# from visualizer import visualize_attention


@dataclass
class BaseModelOutputVision(BaseModelOutput):
    """docstring for BaseModelOutputVision"""
    all_value_states: Optional[Tuple[torch.FloatTensor]] = None
        

@dataclass
class BaseModelOutputWithPoolingVision(BaseModelOutputWithPooling):
    """docstring for BaseModelOutputWithPoolingVision"""
    last_value_state: torch.FloatTensor = None


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_positions_pre = 197

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
        # self.position_embedding = nn.Embedding(self.num_positions_pre, self.embed_dim)
        # self.register_buffer("position_ids", torch.arange(self.num_positions_pre).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # print(embeddings.shape)
        # assert False
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLIPAttentionVision(CLIPAttention):
    """docstring for CLIPAttentionVision"""
    def __init__(self, config):
        super(CLIPAttentionVision, self).__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
        # print(attn_output.shape)
        # print(value_states.shape)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        # print(attn_output.shape)
        attn_output = attn_output.transpose(1, 2)
        # print(attn_output.shape)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        # print(attn_output.shape)

        attn_output = self.out_proj(attn_output)
        # print(attn_output.shape)
        # assert False

        # the same manipulation as the attn_output
        value_states_pure = value_states.view(bsz, self.num_heads, tgt_len, self.head_dim)
        value_states_pure = value_states_pure.transpose(1, 2)
        value_states_pure = value_states_pure.reshape(bsz, tgt_len, embed_dim)
        # 
        value_states_pure = self.out_proj(value_states_pure)

        # return attn_output, attn_weights_reshaped
        return attn_output, attn_weights_reshaped, value_states_pure
        

class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        # print(hidden_states, residual)
        # assert False
        # hidden_states_temp = hidden_states / hidden_states.norm(dim=-1, keepdim=True)
        # residual_temp = residual / residual.norm(dim=-1, keepdim=True)
        # cs = torch.matmul(residual_temp[:,0,:], hidden_states_temp[:,0,:].t())
        # print(cs)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPEncoderLayerVision(CLIPEncoderLayer):
    """docstring for CLIPEncoderLayerVision"""
    def __init__(self, config):
        super(CLIPEncoderLayerVision, self).__init__(config)
        self.self_attn = CLIPAttentionVision(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights, value_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = residual + hidden_states
        # value_states = residual + value_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # residual2 = value_states
        # value_states = self.layer_norm2(value_states)
        # value_states = self.mlp(value_states)
        # value_states = residual2 + value_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        # pure value
        outputs += (value_states,)

        return outputs


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].
    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class CLIPEncoderVision(CLIPEncoder):
    """docstring for CLIPEncoderVision"""
    def __init__(self, config):
        super(CLIPEncoderVision, self).__init__(config)
        self.layers = nn.ModuleList([CLIPEncoderLayerVision(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputVision]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_value_states = ()

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # # collect value states
            # all_value_states = all_value_states + (value_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            # collect last value states
            all_value_states = all_value_states + (layer_outputs[2],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions, all_value_states] if v is not None)
        return BaseModelOutputVision(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions, all_value_states=all_value_states,
        )


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim)
        # self.encoder = CLIPEncoder(config)
        self.encoder = CLIPEncoderVision(config)
        self.post_layernorm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingVision]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        # layer norm for both image patch features and CLS feature
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # layer norm for last value state
        last_value_state = encoder_outputs[3][-1]
        last_value_state = self.post_layernorm(last_value_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingVision(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            last_value_state=last_value_state,
        )


class CLIPModel(CLIPPreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config
        # vision_config.image_size = 256  # 224 -> 256

        # print(vision_config)
        # assert False

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].
        Examples:
        ```python
        >>> from transformers import CLIPTokenizer, CLIPModel
        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel
        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        r"""
        Returns:
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel
        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )
        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class CLIPModelFeature(CLIPModel):
    """docstring for CLIPModelFeature"""
    def __init__(self, config: CLIPConfig):
        super(CLIPModelFeature, self).__init__(config)
        self.smooth = torch.nn.Conv2d(1, 1, 3, padding=1, padding_mode='replicate', bias=False)
        self.smooth.weight.data.fill_(1/9.0)

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].
        Examples:
        ```python
        >>> from transformers import CLIPTokenizer, CLIPModel
        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # print(text_outputs[0].shape)
        # assert False

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel
        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # patch_features = vision_outputs[0][:,1:,:]  # last layer
        patch_features = vision_outputs[4][:,1:,:]  # pure value of last layer
        # patch_features = vision_outputs[2][5][:,1:,:]  # layer 6

        image_features = self.visual_projection(patch_features)

        # pooled_output = vision_outputs[1]  # pooled_output
        # image_features = self.visual_projection(pooled_output)

        pooled_output = vision_outputs[1]  # pooled_output
        pooled_output = self.visual_projection(pooled_output)

        # self_attn = vision_outputs[3][-1][:,9,0,1:].view(14, 14)
        # print(vision_outputs[3][-1].shape)
        # assert False

        # return image_features
        # return image_features, pooled_output, self_attn, vision_outputs[2]
        return image_features, pooled_output, vision_outputs[2]


class Data(object):
    """docstring for Data"""
    def __init__(self):
        super(Data, self).__init__()
        self.image_path = "/media/qichen/ea3763ef-ce45-4f0b-bf4c-646ca3ffa4d3/dataset/COCO/val2017"
        self.text_file = "/media/qichen/ea3763ef-ce45-4f0b-bf4c-646ca3ffa4d3/dataset/COCO/annotations/annotations_trainval2017/annotations/captions_val2017.json"
    
    def build_pairs(self):
        # read caption file
        with open(self.text_file, "r") as f:
            texts = json.load(f)

        pairs = []
        num_caps = len(texts["annotations"])
        for i in range(num_caps):
            img_name = os.path.join(self.image_path, "{:012d}.jpg".format(texts["annotations"][i]["image_id"]))
            cap = texts["annotations"][i]["caption"].replace("/", " ")
            pairs.append((img_name, cap))

        return pairs


class CLIPTool(object):
    """docstring for CLIPTool"""
    def __init__(self, clip_mode=None):
        super(CLIPTool, self).__init__()

        # get model and processor
        if clip_mode=="base-p16-224":
            model_name = "openai/clip-vit-base-patch16"
            self.patch_size = 16
            self.num_patch = 14
        elif clip_mode=="base-p32-224":
            model_name = "openai/clip-vit-base-patch32"
            self.patch_size = 32
            self.num_patch = 7
        elif clip_mode=="large-p14-224":
            model_name = "openai/clip-vit-large-patch14"
            self.patch_size = 14
            self.num_patch = 16
        elif clip_mode=="large-p14-336":
            model_name = "openai/clip-vit-large-patch14-336"
            self.patch_size = 14
            self.num_patch = 24

        self.model = CLIPModelFeature.from_pretrained(model_name)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    # mapping the matching score based on the defined mapping function
    def score_mapping(self, matching_score, tau=0.07):
        threshold = 0.5
        if matching_score > threshold:
            slope = np.exp(threshold*(1/tau))
            matching_score = slope + (matching_score - threshold) * slope
        else:
            matching_score = np.exp(matching_score*(1/tau))

        return matching_score

    def get_attn_mask(self, image_path, caption, mask_size, use_credit, use_image_credit, use_smooth_exp):
        # image = Image.open(requests.get(url, stream=True).raw) # 640x480
        image = Image.open(image_path, mode="r")
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(image_path))).convert('RGB')

        # # both text and image
        # inputs = processor(text=["a photo of a cat", "a photo of a small dog"], images=image, return_tensors="pt", padding=True)
        # outputs = model(**inputs)

        # text only
        # text_inputs = processor(text=["The large brown bear has a black nose"], return_tensors="pt", padding=True)
        text_inputs = self.processor(text=[caption], return_tensors="pt", padding=True)
        # text_inputs = processor(text=["Two cats are sleeping on the bed"], return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(output_hidden_states=True, output_attentions=True, **text_inputs)
        text_features = torch.unsqueeze(text_features, 1)

        # image only
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_features, pooled_output, output_hidden_states = self.model.get_image_features(output_hidden_states=True, output_attentions=True, **image_inputs)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        pooled_output = pooled_output / pooled_output.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features = torch.ones(1, 1, 512)

        pooled_output = torch.unsqueeze(pooled_output, 1)

        # batched matrix x batched matrix => batch x 1 x 49
        # attnmap = self_attn
        # attnmap = torch.matmul(text_features, image_features.transpose(1, 2)).abs() # attention to background -> attention to foreground
        attnmap = torch.matmul(text_features, image_features.transpose(1, 2)) # attention to background -> attention to foreground

        # attnmap = image_features.sum(-1).abs() - image_features.sum(-1).abs().mean()
        # attnmap = ((attnmap.max()-attnmap)/(attnmap.max()-attnmap.min())).softmax(dim=-1)*196

        # attnmap = ((attnmap-attnmap.min())/(attnmap.max()-attnmap.min())).softmax(-1)
        # attnmap = attnmap.softmax(-1)*196
        # print(attnmap)
        # print(attnmap.topk(30))
        # print(attnmap.sum())

        # if similarity < 0, set to 0
        zero = torch.zeros_like(attnmap)
        attnmap = torch.where(attnmap<0, zero, attnmap)

        attnmap = attnmap.reshape(1, 1, self.num_patch, self.num_patch)
        attnmap = self.model.smooth(attnmap)  # smooth the attention map

        # assert False

        # # average the self attention from all the layers
        # attnmap2 = torch.matmul(output_hidden_states2[0][:,0,:], output_hidden_states2[0][:,1:,:].transpose(1, 2))
        # for i in range(1, 12):
        #     attnmap2 += torch.matmul(output_hidden_states2[i][:,0,:], output_hidden_states2[i][:,1:,:].transpose(1, 2))
        # attnmap2 /= 12.0

        # # introduce temperature, low -> sharp; high -> smooth
        # tep = 1.0
        # attnmap = torch.softmax(attnmap.view(1, 1, -1)/tep, dim=2)
        # attnmap = attnmap.sigmoid()
        # attnmap = attnmap.view(1, 1, -1).softmax(dim=2)

        # interpolate the attnmap
        attnmap = attnmap.reshape(1, 1, self.num_patch, self.num_patch)
        attnmap = F.interpolate(attnmap, size=(mask_size, mask_size), mode="bicubic")

        # attention_mask = attnmap.squeeze().view(14, 14).detach().numpy()
        attention_mask = attnmap.squeeze().view(mask_size, mask_size).detach().numpy()
        # print(attention_mask)
        # print(attention_mask.mean())

        # global matching
        matching_score = torch.matmul(text_features, pooled_output.transpose(1, 2))
        # matching_score = np.exp(matching_score.squeeze().detach().numpy()*25)/100
        # matching_score = np.exp(matching_score.squeeze().detach().numpy()*(1/0.07))

        matching_score = matching_score.squeeze().detach().numpy()
        if use_smooth_exp:
            matching_score = self.score_mapping(matching_score)  # mapping score based on the defined function

        # assert False

        if use_image_credit and use_credit:
            # weighted with matching score
            attention_mask = matching_score*attention_mask
        elif use_image_credit and not use_credit:
            attention_mask = np.expand_dims(matching_score, 0).repeat(mask_size, axis=0)
            attention_mask = np.expand_dims(attention_mask, 1).repeat(mask_size, axis=1)

        # np.savetxt("attention_mask_new.txt", attention_mask)
        # assert False
        return attention_mask

    # update model to be suitable to arbitary input size
    def update_position_embedding(self, model):
        num_patches = model.vision_model.embeddings.num_patches
        embed_dim = model.vision_model.embeddings.embed_dim

        # resize position embedding layer weight
        pe_layer_weight = model.vision_model.embeddings.position_embedding.weight[1:,:]
        pe_layer_weight = pe_layer_weight.unsqueeze(0).unsqueeze(1)
        pe_layer_weight = F.interpolate(pe_layer_weight, size=(num_patches, embed_dim), mode="bilinear")
        pe_layer_weight = pe_layer_weight.squeeze()
        pe_layer_weight = torch.cat((model.vision_model.embeddings.position_embedding.weight[0,:].unsqueeze(0), pe_layer_weight), 0)

        # define a new position embedding layer
        position_embedding_updated = nn.Embedding(num_patches, embed_dim)
        position_embedding_updated.weight.data = pe_layer_weight  # set the weight

        # replace position embedding layer weight as the new one
        model.vision_model.embeddings.position_embedding = position_embedding_updated

        # replace position ids
        model.vision_model.embeddings.position_ids = torch.arange(num_patches+1).expand((1, -1))

        return model

    def clip_score(self, image_path, caption, output_path, mask_size=32, use_credit=True, use_image_credit=True, use_smooth_exp=True):
        # data = Data()
        # data_pairs = data.build_pairs()

        # model = CLIPModelFeature.from_pretrained("openai/clip-vit-base-patch16")

        # # update the position embedding layer and position ids in the model
        # model = update_position_embedding(model)

        # model.eval()
        # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        # processor.feature_extractor.size = 256
        # processor.feature_extractor.crop_size = 256

        # url = "http://images.cocodataset.org/val2017/000000000285.jpg"
        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"

        # for image_path, caption in tqdm(data_pairs):
        # get attention mask 32x32
        # mask_size = 32   # 32x14
        attention_mask = self.get_attn_mask(image_path, caption, mask_size, use_credit, use_image_credit, use_smooth_exp)

        # print(attention_mask)
        # print(attention_mask.sum())
        # assert False

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # visualize map
        visualize_attention(img_path=image_path, attention_mask=attention_mask, title=caption, output_path=output_path, mask_size=mask_size, desired_size=self.patch_size*mask_size, cmap="jet")

        return attention_mask.reshape(-1)


if __name__ == '__main__':
    # image_path = "/home/qichen/Desktop/Evaluation/MyEvaluation/run_scripts/image_gen/toy_data/coco_karpathy_split/debug/validation_set_part_unipair/image/000000002753.jpg"
    # caption = "a coffee table sits in the middle of a living room."
    # caption = "an elephant with a man and three children on its back drinking water in the jungle."
    # caption = "a dog holding a yellow frisbee in its mouth."
    output_path = "/home/qichen/Desktop/Evaluation/VisualResults/opgan_clipvisual/clip_layer5"
    image_path_folder = "/home/qichen/Desktop/Evaluation/GenerativeResults/opgan/image"
    text_file = "/home/qichen/Desktop/Evaluation/GenerativeResults/opgan/text/caption.txt"

    with open(text_file, 'r') as f:
        text_items = f.readlines()

    # clip_score(image_path, caption, output_path)

    num_patch_ofa = 32

    # data = Data()
    # data_pairs = data.build_pairs()

    clip_tool = CLIPTool(clip_mode="base-p16-224")
    # model, processor = get_model()

    # for image_path, caption in tqdm(data_pairs):
    for item in text_items:
        idx, caption = item.replace("\n", "").split("\t")
        idx = int(idx)
        image_path = os.path.join(image_path_folder, "{:012d}.jpg".format(idx))
        # calculate clip score
        clip_tool.clip_score(image_path, caption, output_path, mask_size=num_patch_ofa)


