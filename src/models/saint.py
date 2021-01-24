import math

import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import (
    BertEncoder, apply_chunking_to_forward,
    BertSelfAttention, BertAttention, BertLayer
)


class EncoderEmbeddings(nn.Module):
    """Construct the embeddings from Exercise ID, Exercise category, and position embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.id_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.category_embeddings = nn.Embedding(config.category_size, config.embedding_size, padding_idx=config.pad_token_id)

        self.linear_embed = nn.Linear(config.embedding_size * 2, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                input_ids=None,
                category_ids=None,
                timestamp=None,
                elapsed_time=None,
                inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.id_embeddings(input_ids)

        category_embeddings = self.category_embeddings(category_ids)

        embeddings = torch.cat([inputs_embeds, category_embeddings], dim=-1)
        embeddings = self.linear_embed(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransposeBatchNorm1d(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_ch)

    def forward(self, x):
        return self.norm(x.transpose(2, 1)).transpose(2, 1)


class DecoderEmbeddings(nn.Module):
    """Construct the embeddings from Response and position embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # response ids
        #   0: padding id
        #   1: start embedding
        #   3: incorrect answer
        #   4: correct answer
        self.response_embeddings = nn.Embedding(4, config.response_embedding_size, padding_idx=0)
        self.numerical_embeddings = nn.Sequential(TransposeBatchNorm1d(2), nn.Linear(2, config.embedding_size))
        self.elapsed_time_embeddings = nn.Embedding(config.max_elapsed_seconds + 2, config.embedding_size, padding_idx=0)
        self.lag_time_embeddings = nn.Embedding(int(config.max_lag_minutes / 10) + 7, config.embedding_size, padding_idx=0)

        self.linear_embed = nn.Linear(config.response_embedding_size + config.embedding_size * 3, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_lag_time(self, timestamp: torch.LongTensor) -> torch.FloatTensor:
        unique_time, inverse_indices = torch.unique_consecutive(timestamp, return_inverse=True)
        inverse_indices = torch.max(inverse_indices.min(dim=1)[0].unsqueeze(1), inverse_indices - 1)
        lag_time = (timestamp - unique_time[inverse_indices]).float() / (1000 * 60)  # minutes
        return lag_time.clamp(min=0, max=self.config.max_lag_minutes)

    def forward(self,
                input_ids=None,
                category_ids=None,
                timestamp=None,
                elapsed_time=None,
                inputs_embeds=None):
        response_ids = input_ids
        response_embeddings = self.response_embeddings(response_ids)

        # numerical features
        lag_time_num = self.get_lag_time(timestamp)
        elapsed_time_num = elapsed_time.clamp(min=0, max=self.config.max_elapsed_seconds)

        numerical_states = torch.stack([
            lag_time_num.log1p(),
            elapsed_time_num
        ], dim=-1)
        numerical_embeddings = self.numerical_embeddings(numerical_states)

        # lag time as categorical embedding
        lag_time_cat = torch.where(
            lag_time_num < 6,
            lag_time_num.long(),
            ((lag_time_num - 1) / 10).long() + 6
        )
        lag_time_embeddings = self.lag_time_embeddings(lag_time_cat)

        # elapsed time as categorical embedding
        elapsed_time_cat = (elapsed_time.long() + 1).clamp(min=0, max=self.config.max_elapsed_seconds)
        elapsed_time_embeddings = self.elapsed_time_embeddings(elapsed_time_cat)

        embeddings = torch.cat([
            response_embeddings,
            numerical_embeddings,
            lag_time_embeddings,
            elapsed_time_embeddings
        ], dim=-1)
        embeddings = self.linear_embed(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SaintSelfAttention(BertSelfAttention):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def scale_attention_scores(self,
                               attention_scores: torch.FloatTensor,
                               timestamp: torch.LongTensor = None,
                               is_crossattention: bool = False) -> torch.FloatTensor:
        """Scale attention scores based on lag time.

        Parameters
        ----------
        attention_scores : torch.FloatTensor with shape (batch_size, num_heads, sequence_length, sequence_length)
        timestamp : torch.LongTensor, optional with shape (batch_size, sequence_length)
        is_crossattention : bool
        """
        if timestamp is not None:
            batch_size = attention_scores.size(0)
            device = attention_scores.device
            if is_crossattention:
                # decoder timestamp
                timestamp_new = torch.cat([
                    torch.LongTensor([[-1]] * batch_size).to(device),
                    timestamp[:, :-1]
                ], dim=1)
                lag_time = (timestamp_new[:, None, :, None] - timestamp[:, None, None, :]).float() / (60 * 1000)
            
            elif self.config.is_decoder:
                # decoder timestamp
                timestamp_new = torch.cat([
                    torch.LongTensor([[-1]] * batch_size).to(device),
                    timestamp[:, :-1]
                ], dim=1)
                lag_time = (timestamp_new[:, None, :, None] - timestamp_new[:, None, None, :]).float() / (60 * 1000)
            
            else:
                lag_time = (timestamp[:, None, :, None] - timestamp[:, None, None, :]).float() / (60 * 1000)

            scale = math.sqrt(self.attention_head_size) - self.config.lag_time_scale_alpha / (lag_time.clamp(min=0) + 1) + self.config.lag_time_scale_alpha

        else:
            scale = math.sqrt(self.attention_head_size)

        return attention_scores / scale

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        timestamp=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
            is_crossattention = True
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
            is_crossattention = False

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = self.scale_attention_scores(attention_scores, timestamp, is_crossattention=is_crossattention)
        attention_scores = attention_scores + position_bias
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer,)


class SaintAttention(BertAttention):

    def __init__(self, config):
        super().__init__(config)
        self.self = SaintSelfAttention(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        timestamp=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            position_bias,
            timestamp,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SaintLayer(BertLayer):

    def __init__(self, config):
        super().__init__(config)
        self.attention = SaintAttention(config)
        if self.add_cross_attention:
            self.crossattention = SaintAttention(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        timestamp=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            position_bias,
            timestamp,
            head_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                position_bias,
                timestamp,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs


class SaintEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([SaintLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        timestamp=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                position_bias,
                timestamp,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
            hidden_states = layer_outputs[0]

        return hidden_states


class SaintModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    
    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.is_decoder:
            self.embeddings = DecoderEmbeddings(config)
        else:
            self.embeddings = EncoderEmbeddings(config)
        
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.position_query_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.position_key_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.position_scaling = float(config.hidden_size / config.num_attention_heads) ** -0.5 
        self.position_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, config.num_attention_heads)

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        self.encoder = SaintEncoder(config)
        self.init_weights()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, position_ids):
        """ Compute position bias """
        # absolute positional embedding
        position_embeddings = self.position_embeddings(position_ids)  # (batch_size, seq_length, hidden_size)
        position_embeddings = self.position_layer_norm(position_embeddings)
        
        position_query = self.position_query_linear(position_embeddings)  # (batch_size, seq_length, hidden_size)
        position_key = self.position_key_linear(position_embeddings)  # (batch_size, seq_length, hidden_size)

        position_query = self.transpose_for_scores(position_query) * self.position_scaling  # (batch_size, num_attention_heads, seq_length, attention_head_size)
        position_key = self.transpose_for_scores(position_key)  # (batch_size, num_attention_heads, seq_length, attention_head_size)

        absolute_bias = torch.matmul(position_query, position_key.transpose(-1, -2))  # (batch_size, num_attention_heads, seq_length, seq_length)

        # relative positional embedding
        relative_position = position_ids[:, None, :] - position_ids[:, :, None]  # (batch_size, seq_length, seq_length)

        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (batch_size, query_length, key_length)
            bidirectional=False,
            num_buckets=self.config.relative_attention_num_buckets,
            max_distance=self.config.max_position_embeddings
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        relative_bias = self.relative_attention_bias(relative_position_bucket)  # shape (batch_size, query_length, key_length, num_heads)
        relative_bias = relative_bias.permute([0, 3, 1, 2])  # shape (batch_size, num_heads, query_length, key_length)

        position_bias = absolute_bias + relative_bias
        return position_bias

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        timestamp=None,
        category_ids=None,
        position_ids=None,
        elapsed_time=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # Upper Triangular Mask
        attention_mask = torch.tril(
            torch.matmul(attention_mask[:, :, None], attention_mask[:, None, :])
        )  # [batch_size, seq_length, seq_length]

        if category_ids is None:
            category_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            # Upper Triangular Mask
            encoder_attention_mask = torch.tril(
                torch.matmul(encoder_attention_mask[:, :, None], encoder_attention_mask[:, None, :])
            )  # [batch_size, seq_length, seq_length]

            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if position_ids is None:
            position_ids = (1 - (timestamp <= timestamp.roll(1, dims=1)).long()).cumsum(dim=1)

        position_bias = self.compute_bias(position_ids)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            category_ids=category_ids,
            timestamp=timestamp,
            elapsed_time=elapsed_time,
            inputs_embeds=inputs_embeds
        )
        encoder_output = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            position_bias=position_bias,
            timestamp=timestamp,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        return encoder_output


class SaintEncoderDecoderModel(nn.Module):
    """SAINT-based model

    References
    ----------
    - https://arxiv.org/abs/2002.07033
    - https://arxiv.org/abs/2010.12042
    """

    def __init__(self, encoder_config, decoder_config, num_labels: int = 1):
        super().__init__()
        self.encoder = SaintModel(encoder_config)
        self.decoder = SaintModel(decoder_config)

        self.dropout = nn.Dropout(decoder_config.hidden_dropout_prob)
        self.classifier = nn.Linear(decoder_config.hidden_size, num_labels)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        timestamp=None,
        category_ids=None,
        elapsed_time=None,
        response_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,  # TODO: (PVP) implement :obj:`use_cache`
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        **kwargs,
    ):
        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith('decoder_')}
        kwargs_decoder = {
            argument[len('decoder_') :]: value for argument, value in kwargs.items() if argument.startswith('decoder_')
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                timestamp=timestamp,
                category_ids=category_ids,
                inputs_embeds=inputs_embeds,
                **kwargs_encoder,
            )

        # Decode
        decoder_output = self.decoder(
            input_ids=response_ids,
            attention_mask=decoder_attention_mask,
            timestamp=timestamp,
            elapsed_time=elapsed_time,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            **kwargs_decoder,
        )

        decoder_output = self.dropout(decoder_output)
        logits = self.classifier(decoder_output).squeeze(2)
        return logits
