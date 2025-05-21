import copy
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer, GenerationMixin
from torch.nn import CrossEntropyLoss
from transformers.models.t5.modeling_t5 import T5Stack, Seq2SeqLMOutput, BaseModelOutput
from model_util import HierarchicalGatedExpertNetwork, NewsEmbedding, PoolingMethod

import torch.nn.functional as F

class UT5ForConditionalGeneration(T5ForConditionalGeneration, GenerationMixin):
    """
    Custom T5 model for conditional generation with user history integration.
    Extends T5ForConditionalGeneration and GenerationMixin.
    """
    def __init__(self, config: T5Config, short_term_k=10):
        """
        Initializes the UT5ForConditionalGeneration model.

        Args:
            config (T5Config): The configuration object for the T5 model.
            short_term_k (int): The number of recent historical clicks to consider for short-term representation.
        """
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.short_term_k = short_term_k  # Define the number of historical clicks for short-term representation

        # Initialize encoder and decoder
        self.init_encoders(config)
        self.init_decoders(config)

        # Gated expert network for combining user representations
        self.gated_expert_network = HierarchicalGatedExpertNetwork(config)

        # Pooling methods
        self.pooling_method_mean = PoolingMethod(config, method='mean')
        # self.pooling_method_attention = PoolingMethod(config, method='attention') # Example of other pooling methods
        self.pooling_method_query = PoolingMethod(config, method='query')
        # self.pooling_method_self_voting = PoolingMethod(config, method='self_voting')
        
        # Initialize interest and style embedding layers
        news_embedding_loader = NewsEmbedding()
        self.interest_embedding = nn.Embedding(news_embedding_loader.vocab_size, news_embedding_loader.embedding_dim)
        self.style_embedding = nn.Embedding(news_embedding_loader.vocab_size, news_embedding_loader.embedding_dim)

        # Load pre-trained weights without replacing nn.Parameter
        self.interest_embedding.weight.data.copy_(news_embedding_loader.get_interest_weights())
        self.style_embedding.weight.data.copy_(news_embedding_loader.get_style_weights())

        # Freeze embedding layers
        self.interest_embedding.weight.requires_grad = False
        self.style_embedding.weight.requires_grad = False
        
        # Add independent projection layers for interest and style
        self.interest_projection = nn.Linear(news_embedding_loader.embedding_dim, config.d_model)
        self.style_projection = nn.Linear(news_embedding_loader.embedding_dim, config.d_model)
        
        # Add GRU layers for short-term representation processing
        self.interest_gru = nn.GRU(input_size=config.d_model, hidden_size=config.d_model, batch_first=True)
        self.style_gru = nn.GRU(input_size=config.d_model, hidden_size=config.d_model, batch_first=True)

        # Language model head for generating output logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and perform final processing
        self.post_init()

    def init_encoders(self, config):
        """
        Initializes the encoder part of the model.
        """
        # Article encoder
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        self.encoder = T5Stack(encoder_config, self.shared)

    def init_decoders(self, config):
        """
        Initializes the decoder part of the model.
        """
        # Decoder embedding layer and decoder stack
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

    def encode_user_history(self, user_history_ids, user_history_mask):
        """
        Encodes user historical clicks, returning style and interest representations
        for each title and the corresponding attention mask.
        
        Args:
            user_history_ids: Tensor of shape (batch_size, max_history_length)
                              containing token IDs of historical click titles.
            user_history_mask: Tensor of shape (batch_size, max_history_length)
                               containing attention mask for historical clicks.
        
        Returns:
            style_title_repr: Tensor of shape (batch_size, max_history_length, d_model)
                              containing style representations of historical titles.
            interest_title_repr: Tensor of shape (batch_size, max_history_length, d_model)
                                 containing interest representations of historical titles.
            valid_title_mask: Tensor of shape (batch_size, max_history_length)
                              containing the mask for valid historical titles.
        """
        # Encode user historical click titles using embedding layers
        style_title_repr = self.style_embedding(user_history_ids)      # (batch_size, max_history_length, embedding_dim)
        interest_title_repr = self.interest_embedding(user_history_ids)  # (batch_size, max_history_length, embedding_dim)
        valid_title_mask = user_history_mask                  # (batch_size, max_history_length)
        # embedding_dim -> 3584 (example dimension)
        
        # Project embeddings to model dimension (d_model)
        style_title_repr = self.style_projection(style_title_repr)  # (batch_size, max_history_length, d_model)
        interest_title_repr = self.interest_projection(interest_title_repr) # (batch_size, max_history_length, d_model)
        
        return style_title_repr, interest_title_repr, valid_title_mask
    
    def calculate_proxies(self, style_title_repr, interest_title_repr, valid_title_mask, short_term_k):
        """
        Calculates proxy representations for long-term and short-term interests
        using mean pooling.
        
        Args:
            style_title_repr: Tensor of shape (batch_size, num_titles, d_model)
                              containing style representations.
            interest_title_repr: Tensor of shape (batch_size, num_titles, d_model)
                                 containing interest representations.
            valid_title_mask: Tensor of shape (batch_size, num_titles)
                              containing the mask for valid titles.
            short_term_k (int): The number of recent historical clicks for short-term proxy.
        
        Returns:
            dict: A dictionary containing the calculated proxy representations.
                  Keys: 'style_long', 'interest_long', 'style_short', 'interest_short'.
                  Values: Tensors of shape (batch_size, d_model).
        """
        # Calculate long-term proxy representations using mean pooling over all valid history
        style_long_proxy = self.pooling_method_mean(style_title_repr, attention_mask=valid_title_mask)
        interest_long_proxy = self.pooling_method_mean(interest_title_repr, attention_mask=valid_title_mask)

        # Short-term proxy: Select the last k valid clicks and calculate their mean
        # Note: This simple slicing might not handle padding correctly if k is larger than actual history length.
        # A more robust approach would involve masking or dynamic slicing based on valid_title_mask.
        # However, based on the original code structure, simple slicing is used here.
        style_short_proxy = self.pooling_method_mean(style_title_repr[:, -short_term_k:, :])
        interest_short_proxy = self.pooling_method_mean(interest_title_repr[:, -short_term_k:, :])
        
        proxies = {
            'style_long': style_long_proxy,
            'interest_long': interest_long_proxy,
            'style_short': style_short_proxy,
            'interest_short': interest_short_proxy
        }
        
        return proxies
    
    def euclidean_distance(self, a, b):
        """
        Calculates the Euclidean distance between two tensors along the last dimension.
        
        Args:
            a: Tensor
            b: Tensor
            
        Returns:
            Tensor: Euclidean distance.
        """
        # Calculate Euclidean distance
        return torch.norm(a - b, p=2, dim=-1)

    def contrastive_loss(self, anchor, positive, negatives, margin=0.5):
        """
        Calculates the contrastive loss: aims to make the distance between anchor
        and positive smaller than the distance between anchor and negatives.
        
        Args:
            anchor: Anchor tensor.
            positive: Positive tensor.
            negatives (list): List of negative tensors.
            margin (float): Margin for the contrastive loss.
            
        Returns:
            Tensor: The calculated contrastive loss.
        """
        loss = 0
        pos_dist = self.euclidean_distance(anchor, positive)
        for neg in negatives:
            neg_dist = self.euclidean_distance(anchor, neg)
            # Max(0, positive_distance - negative_distance + margin)
            loss += F.relu(pos_dist - neg_dist + margin).mean()
        return loss


    def representation_aggregation(self, style_title_repr, interest_title_repr, valid_title_mask, article_repr):
        """
        Aggregates user representations, including long-term and short-term
        style and interest representations.
        
        Args:
            style_title_repr: Tensor of shape (batch_size, num_titles, d_model)
                              containing style representations of historical titles.
            interest_title_repr: Tensor of shape (batch_size, num_titles, d_model)
                                 containing interest representations of historical titles.
            valid_title_mask: Tensor of shape (batch_size, num_titles)
                              containing the mask for valid historical titles.
            article_repr: Tensor of shape (batch_size, d_model)
                          containing the representation of the candidate article.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                style_long_repr: Long-term style representation (batch_size, d_model).
                interest_long_repr: Long-term interest representation (batch_size, d_model).
                style_short_repr: Short-term style representation (batch_size, d_model).
                interest_short_repr: Short-term interest representation (batch_size, d_model).
        """
        # Long-term representation: Attention pooling over all valid titles
        # using the candidate article representation as query
        style_long_repr = self.pooling_method_query(style_title_repr, attention_mask = valid_title_mask, query = article_repr)  # (batch_size, d_model)
        interest_long_repr = self.pooling_method_query(interest_title_repr, attention_mask = valid_title_mask, query = article_repr)  # (batch_size, d_model)

        # Short-term representation: Select the last k valid titles and perform attention pooling
        short_term_k = self.short_term_k
        batch_size, num_titles, d_model = style_title_repr.size()
        
        # Reverse the mask to have the most recent clicks first
        reversed_mask = valid_title_mask.flip(dims=[1])  # (batch_size, num_titles)
        
        # Create an index tensor
        indices = torch.arange(num_titles, device=valid_title_mask.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_titles)
        reversed_indices = indices.flip(dims=[1])  # (batch_size, num_titles)
        
        # Set invalid positions to a large index value so they are sorted to the end
        masked_reversed_indices = torch.where(reversed_mask == 1, reversed_indices, torch.full_like(reversed_indices, num_titles))
        
        # Get the indices of the top k valid clicks (which are the last k in the original order)
        selected_indices = masked_reversed_indices.topk(short_term_k, dim=1, largest=False, sorted=True).values  # (batch_size, k)
        
        # Handle cases with fewer than k valid clicks (e.g., pad with 0 indices)
        selected_indices = torch.where(selected_indices >= num_titles, torch.zeros_like(selected_indices), selected_indices)
        
        # Sort selected_indices to ensure processing from oldest to newest within the short-term window
        sorted_selected_indices, _ = torch.sort(selected_indices, descending=True, dim=1)
        
        # Use gather to select the last k valid titles based on sorted indices
        style_short_selected = torch.gather(style_title_repr, 1, sorted_selected_indices.unsqueeze(-1).expand(-1, -1, d_model))  # (batch_size, k, d_model)
        interest_short_selected = torch.gather(interest_title_repr, 1, sorted_selected_indices.unsqueeze(-1).expand(-1, -1, d_model))  # (batch_size, k, d_model)

        # Process the batch of short-term representations through GRU
        # GRU returns output and hidden state. We use the last hidden state as the aggregated representation.
        _, hidden_style = self.style_gru(style_short_selected)  # hidden_style: (1, batch_size, d_model)
        _, hidden_interest = self.interest_gru(interest_short_selected)  # hidden_interest: (1, batch_size, d_model)

        # Reshape to (batch_size, d_model), storing the last hidden state
        style_short_repr = hidden_style.squeeze(0)  # (batch_size, d_model)
        interest_short_repr = hidden_interest.squeeze(0)  # (batch_size, d_model)

        return style_long_repr, interest_long_repr, style_short_repr, interest_short_repr
    
    def prepare_decoder_inputs(self, decoder_input_ids, user_representation):
        """
        Prepares decoder inputs by embedding the input IDs and optionally
        adding the user representation.
        
        Args:
            decoder_input_ids: Tensor of shape (batch_size, target_seq_length)
                               containing decoder input token IDs.
            user_representation: Tensor of shape (batch_size, d_model)
                                 containing the user representation, or None.
                                 
        Returns:
            Tensor: Decoder input embeddings with user representation added (if provided).
                    Shape: (batch_size, target_seq_length, d_model).
        """
        # Embed decoder input IDs
        decoder_inputs_embeds = self.shared(decoder_input_ids)
        if user_representation is not None:
            # Expand user_representation to match the sequence length of decoder inputs
            user_representation_expanded = user_representation.unsqueeze(1).expand_as(decoder_inputs_embeds)
            # user_representation_expanded -> (batch_size, seq_length, d_model)

            # Fuse user representation into decoder_inputs_embeds
            decoder_inputs_embeds = decoder_inputs_embeds + user_representation_expanded
            
        return decoder_inputs_embeds

    def forward(
        self,
        input_ids=None,  # Input IDs for the candidate news article content
        attention_mask=None, # Attention mask for the candidate news article content
        decoder_input_ids=None, # Input IDs for the decoder (target sequence)
        decoder_attention_mask=None, # Attention mask for the decoder input
        user_history_ids=None,  # Input IDs for user historical click titles
        user_history_mask=None, # Attention mask for user historical click titles
        head_mask = None, # Mask to nullify selected heads of the encoder
        decoder_head_mask = None, # Mask to nullify selected heads of the decoder
        cross_attn_head_mask  = None, # Mask to nullify selected heads of the cross-attention
        encoder_outputs = None, # Pre-computed encoder outputs
        past_key_values = None, # Past key and value states for generation
        inputs_embeds = None, # Input embeddings for the encoder
        decoder_inputs_embeds = None, # Input embeddings for the decoder
        labels=None, # Labels for language modeling loss
        use_cache = None, # Whether to use cache for generation
        output_attentions = None, # Whether to return attentions
        output_hidden_states = None, # Whether to return hidden states
        return_dict = None, # Whether to return a dictionary
        user_representation = None,  # Explicitly passed user representation (optional)
    ):
        """
        Forward pass function of the model.

        Args:
            input_ids: Tensor of shape (batch_size, seq_length) for candidate article content.
            attention_mask: Tensor of shape (batch_size, seq_length) for candidate article content.
            user_history_ids: Tensor of shape (batch_size, max_history_length) for user history titles.
            user_history_mask: Tensor of shape (batch_size, max_history_length) for user history titles.
            decoder_input_ids: Tensor of shape (batch_size, target_seq_length) for decoder input.
            decoder_attention_mask: Tensor of shape (batch_size, target_seq_length) for decoder input.
            labels: Tensor of shape (batch_size, target_seq_length) for language modeling loss.
            user_representation: Pre-computed user representation, if available.

        Returns:
            Seq2SeqLMOutput: Output object containing loss, logits, and other model outputs.
        """
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        cl_loss = None # Initialize contrastive loss

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs into embeddings if needed
            # Encode candidate news article content
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            # If encoder_outputs is a tuple and return_dict is True, convert to BaseModelOutput
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_hidden_states = encoder_outputs[0]  # (batch_size, seq_length, d_model)
        # Get article representation, e.g., via mean pooling
        article_representation = self.pooling_method_mean(encoder_hidden_states, attention_mask)  # (batch_size, d_model)
        
        # If user_representation is not explicitly provided and user history is available, compute it
        if user_representation is None and user_history_ids is not None:
            # Encode user historical click titles
            (
                style_title_repr, 
                interest_title_repr, 
                valid_title_mask
            ) = self.encode_user_history(
                user_history_ids, user_history_mask,
            )
            
            # Aggregate user representations (long-term and short-term style/interest)
            style_long_repr, interest_long_repr, style_short_repr, interest_short_repr = self.representation_aggregation(
                style_title_repr, 
                interest_title_repr, 
                valid_title_mask,
                article_representation # Pass article representation for query pooling
            )
            
            # Calculate proxy representations for contrastive loss
            proxies = self.calculate_proxies(style_title_repr, interest_title_repr, valid_title_mask, self.short_term_k)

            # Calculate contrastive loss
            cl_loss = (
                self.contrastive_loss(style_long_repr, proxies['style_long'], [proxies['style_short'], proxies['interest_long']]) +
                self.contrastive_loss(style_short_repr, proxies['style_short'], [proxies['style_long'], proxies['interest_short']]) +
                self.contrastive_loss(interest_long_repr, proxies['interest_long'], [proxies['interest_short'], proxies['style_long']]) +
                self.contrastive_loss(interest_short_repr, proxies['interest_short'], [proxies['interest_long'], proxies['style_short']])
            )

            # Gated expert network fuses the four representations to generate
            # the candidate news-aware user representation 'u'
            user_representation = self.gated_expert_network(
                style_long_repr, 
                interest_long_repr, 
                style_short_repr, 
                interest_short_repr, 
                article_representation  # Pass article representation to the expert network
            )  # (batch_size, d_model)
        

        # Prepare decoder inputs
        # If labels are provided and decoder_input_ids/embeds are not, shift labels to the right
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        
        # Prepare decoder embeddings only if decoder_input_ids are available and decoder_inputs_embeds are not
        if decoder_input_ids is not None and decoder_inputs_embeds is None:
            decoder_inputs_embeds = self.prepare_decoder_inputs(decoder_input_ids, user_representation)  # (batch_size, seq_length, d_model)
        else:
            # If decoder_inputs_embeds are already provided or decoder_input_ids are not, use provided embeds or None
            decoder_inputs_embeds = decoder_inputs_embeds

        # Decode
        decoder_outputs = self.decoder(
            input_ids=None, # input_ids is None when inputs_embeds is provided
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = decoder_outputs[0]  # (batch_size, seq_length, d_model)

        # Compute logits using the language model head
        lm_logits = self.lm_head(sequence_output)  # (batch_size, seq_length, vocab_size)

        # Calculate loss if labels are provided
        loss = None
        
        if labels is not None:
            # Use CrossEntropyLoss, ignoring index -100 (typically used for padding)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP (Pipeline Parallelism)
            labels = labels.to(lm_logits.device)
            # Reshape logits and labels for loss calculation
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            
            # Add contrastive loss with a weighting factor
            if cl_loss is not None:
                loss += cl_loss * 0.1 # Weighting factor for contrastive loss
            
        # Return output based on return_dict flag
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        # Return Seq2SeqLMOutput object if return_dict is True
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids, # Input IDs for the current step of generation
        past_key_values=None, # Past key and value states
        attention_mask=None, # Attention mask for the encoder output
        head_mask=None, # Mask for encoder heads
        decoder_head_mask=None, # Mask for decoder heads
        decoder_attention_mask=None, # Attention mask for the decoder input
        cross_attn_head_mask=None, # Mask for cross-attention heads
        use_cache=None, # Whether to use cache
        encoder_outputs=None, # Pre-computed encoder outputs
        user_history_ids=None, # User history IDs (needed for first step if user_representation is not pre-computed)
        user_history_mask=None, # User history mask (needed for first step if user_representation is not pre-computed)
        user_representation=None,  # User representation (can be pre-computed or computed in the first step)
        **kwargs, # Additional keyword arguments
    ):
        """
        Prepares inputs for the `forward` method during generation.
        This method is used by the `GenerationMixin`.
        """
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            # Determine the length of the past sequence from the cached key/value states
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            # If the current input_ids sequence is longer than the past sequence length,
            # it means we are likely in the first step or have a prefix.
            if input_ids.shape[1] > past_length:
                # Remove the prefix from input_ids to get only the new tokens
                remove_prefix_length = past_length
            else:
                # Default behavior: keep only the final ID for subsequent steps
                remove_prefix_length = input_ids.shape[1] - 1

            # Slice input_ids to keep only the relevant part for the current step
            input_ids = input_ids[:, remove_prefix_length:]

        # Return a dictionary of inputs for the forward pass
        return {
            "decoder_input_ids": input_ids, # The current decoder input IDs
            "past_key_values": past_key_values, # Pass past key/value states for caching
            "encoder_outputs": encoder_outputs, # Pass encoder outputs
            "attention_mask": attention_mask, # Pass encoder attention mask
            "head_mask": head_mask, # Pass encoder head mask
            "decoder_head_mask": decoder_head_mask, # Pass decoder head mask
            "decoder_attention_mask": decoder_attention_mask, # Pass decoder attention mask
            "cross_attn_head_mask": cross_attn_head_mask, # Pass cross-attention head mask
            "use_cache": use_cache, # Pass use_cache flag
            "user_history_ids": user_history_ids, # Pass user history IDs
            "user_history_mask": user_history_mask, # Pass user history mask
            "user_representation": user_representation,  # Pass user representation
        }