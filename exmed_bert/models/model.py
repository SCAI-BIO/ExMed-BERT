# import -----------------------------------------------------------------------
import logging
from typing import Any, Dict, Optional, OrderedDict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from torch.nn.modules.loss import BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertLMPredictionHead,
    BertModel,
    BertOnlyMLMHead,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)

from exmed_bert.data.encoding import CodeDict
from exmed_bert.models.config import CombinedConfig, ExMedBertConfig

# global vars ------------------------------------------------------------------

logger = logging.getLogger(__name__)


# class definitions ------------------------------------------------------------
class ExMedBertEmbeddings(nn.Module):
    """Construct embeddings for phewas-codes, segments, and age"""

    def __init__(self, config: ExMedBertConfig) -> None:
        """Initialization of embedding layer

        Args:
            config (ExMedBertConfig): Config instance
        """
        super(ExMedBertEmbeddings, self).__init__()

        # Embedding layers
        self.code_embeddings = nn.Embedding(config.code_vocab_size, config.hidden_size)
        self.entity_embeddings = nn.Embedding(
            config.number_of_codes, config.hidden_size
        )
        self.sex_embeddings = nn.Embedding(config.sex_vocab_size, config.hidden_size)

        # optional embeddings
        if config.max_position_embedding is not None:
            self.positional_embeddings = nn.Embedding(
                config.max_position_embedding, config.hidden_size
            ).from_pretrained(
                embeddings=self._init_position_embedding(
                    config.max_position_embedding, config.hidden_size
                )
            )
        else:
            self.positional_embeddings = None

        if config.age_vocab_size is not None:
            self.age_embeddings = nn.Embedding(
                config.age_vocab_size, config.hidden_size
            )
        else:
            self.age_embeddings = None

        if config.region_vocab_size is not None:
            self.state_embeddings = nn.Embedding(
                config.region_vocab_size, config.hidden_size
            )
        else:
            self.state_embeddings = None

        # Other configuration
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embedding).expand((1, -1))
        )

    def forward(
        self,
        input_ids: Tensor,
        entity_ids: Tensor,
        age_ids: Optional[Tensor] = None,
        sex_ids: Optional[Tensor] = None,
        state_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        one_hot_input_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of embedding layer

        Args:
            input_ids (Tensor): Input ids for codes (ATC+Phecodes)
            entity_ids (Tensor): Entity ids
            age_ids (Optional[Tensor], optional): Age ids. Defaults to None.
            sex_ids (Optional[Tensor], optional): Sex ids. Defaults to None.
            state_ids (Optional[Tensor], optional): State ids. Defaults to None.
            position_ids (Optional[Tensor], optional): Position ids. Defaults to None.
            one_hot_input_ids (Optional[Tensor], optional): One hot input ids. Usually not required. Defaults to None.

        Returns:
            Tensor: Embedded input
        """

        # zero tensor if input is not provided
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if position_ids is None:
            position_ids = torch.zeros_like(input_ids)
        if sex_ids is None:
            sex_ids = torch.zeros_like(input_ids)
        if state_ids is None:
            state_ids = torch.zeros_like(input_ids)

        # mapping for all input elements
        if one_hot_input_ids is not None:
            logger.info("Use matrix multiplication instead of embedding.")
            code_embed = torch.matmul(
                one_hot_input_ids.float(),
                self.code_embeddings.weight,
            )
        else:
            code_embed = self.code_embeddings(input_ids)
        entity_embed = self.entity_embeddings(entity_ids)
        sex_embed = self.sex_embeddings(sex_ids)

        age_embed = (
            self.age_embeddings(age_ids)
            if self.age_embeddings is not None
            else torch.zeros_like(code_embed)
        )
        position_embed = (
            self.positional_embeddings(position_ids)
            if self.positional_embeddings is not None
            else torch.zeros_like(code_embed)
        )
        region_embed = (
            self.state_embeddings(state_ids)
            if self.state_embeddings is not None
            else torch.zeros_like(code_embed)
        )

        embedding = (
            code_embed
            + entity_embed
            + age_embed
            + position_embed
            + sex_embed
            + region_embed
        )
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        return embedding

    @staticmethod
    def _init_position_embedding(max_position_embedding, hidden_size):
        def even_code(even_pos, even_idx):
            return np.sin(even_pos / (10000 ** (2 * even_idx / hidden_size)))

        def odd_code(odd_pos, odd_idx):
            return np.cos(odd_pos / (10000 ** (2 * odd_idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return Tensor(lookup_table)


class ExMedBertModel(BertModel):
    def __init__(
        self,
        config: ExMedBertConfig,
        add_pooling_layer: bool = True,
        code_embed: Optional[CodeDict] = None,
    ):
        """Initialize ExMed-BERT base model

        Args:
            config (ExMedBertConfig): Config instance
            add_pooling_layer (bool, optional): Indicator whether pooling layer is added. Defaults to True.
            code_embed (Optional[CodeDict], optional): CodeDict instance. Defaults to None.
        """
        super(BertModel, self).__init__(config)

        self.code_embed = code_embed
        self.config = config

        self.embeddings: ExMedBertEmbeddings = ExMedBertEmbeddings(config)

        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            if self.config.initialization == "normal":
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif self.config.initialization == "orthogonal":
                torch.nn.init.orthogonal_(module.weight)
            else:
                raise Exception("Initialization type has to be provided in config")
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if self.config.initialization == "normal":
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif self.config.initialization == "orthogonal":
                torch.nn.init.orthogonal_(module.weight)
            else:
                raise Exception("Initialization type has to be provided in config")
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def floating_point_ops(
        self,
        input_dict: Dict[str, Union[Tensor, Any]],
        exclude_embeddings: bool = True,
    ) -> int:
        """

        Args:
            input_dict (Dict[str, Any): Dictionary with model inputs
            exclude_embeddings (bool): Whether to count embedding and softmax operations.

        Returns: The number of floating point operations

        """

        return (
            6
            * self.estimate_tokens(input_dict)
            * self.num_parameters(exclude_embeddings=exclude_embeddings)
        )

    def estimate_tokens(self, input_dict: Dict[str, Union[Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            input_dict (:obj:`dict`): The model inputs.

        Returns:
            :obj:`int`: The total number of tokens.
        """
        token_inputs = [tensor for key, tensor in input_dict.items() if "ids" in key]
        if token_inputs:
            return sum([token_input.numel() for token_input in token_inputs])  # type: ignore
        else:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            return 0

    def num_parameters(
        self, only_trainable: bool = False, exclude_embeddings: bool = False
    ) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to return only the number of trainable parameters
            exclude_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to return only the number of non-embeddings parameters

        Returns:
            :obj:`int`: The number of parameters.

        FROM transformers
        """

        def parameter_filter(x):
            return (x.requires_grad or not only_trainable) and not (
                isinstance(x, torch.nn.Embedding) and exclude_embeddings
            )

        params = (
            filter(parameter_filter, self.parameters())
            if only_trainable
            else self.parameters()
        )
        return sum(p.numel() for p in params)

    def forward(
        self,
        input_ids: LongTensor,
        entity_ids: LongTensor,
        age_ids: LongTensor,
        sex_ids: LongTensor,
        state_ids: LongTensor,
        attention_mask: LongTensor,
        position_ids: LongTensor,
        one_hot_input_ids: Optional[LongTensor] = None,
        output_attentions: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Forward pass of ExMed-BERT model

        Args:
            input_ids (Tensor): Input ids for codes (ATC+Phecodes)
            entity_ids (Tensor): Entity ids
            age_ids (Optional[Tensor], optional): Age ids. Defaults to None.
            sex_ids (Optional[Tensor], optional): Sex ids. Defaults to None.
            state_ids (Optional[Tensor], optional): State ids. Defaults to None.
            attention_mask (Optional[Tensor], optional): Attention mask. Defaults to None.
            position_ids (Optional[Tensor], optional): Position ids. Defaults to None.
            one_hot_input_ids (Optional[Tensor], optional): One hot input ids. Usually not required. Defaults to None.
            output_attentions (bool, optional): Indicate whether the attention should be returned. Defaults to False.

        Raises:
            ValueError

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        """
        if input_ids.ndim == 3:
            input_shape = input_ids.size()[:2]
        elif input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)  # type: ignore
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids, device=device)  # type: ignore
        if position_ids is None:
            position_ids = torch.zeros_like(input_ids, device=device)  # type: ignore
        if sex_ids is None:
            sex_ids = torch.zeros_like(input_ids, device=device)  # type: ignore
        if state_ids is None:
            state_ids = torch.zeros_like(input_ids, device=device)  # type: ignore

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
        if input_ids.ndim == 3:
            logger.info("Treat input ids as embedding")
            embedding_output = input_ids
        else:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                entity_ids=entity_ids,
                age_ids=age_ids,
                sex_ids=sex_ids,
                state_ids=state_ids,
                position_ids=position_ids,
                one_hot_input_ids=one_hot_input_ids,
            )
        encoder_output = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_attentions=output_attentions,
        )

        out = (encoder_output["last_hidden_state"],)

        if output_attentions:
            out += (encoder_output["attentions"],)

        if self.pooler is not None:
            pooled_output = self.pooler(encoder_output["last_hidden_state"])
            return out + (pooled_output,)  # type: ignore
        else:
            return  out # type: ignore


class BertLMPredictionHeadVar(BertLMPredictionHead):
    """Inherited LM Head with variable output size (from transformers library)"""

    def __init__(self, config, output_size: int):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, output_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(output_size))  # type: ignore

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias


class BertOnlyMLMHeadVar(BertOnlyMLMHead):
    """Inherited LM Head with variable output size"""

    def __init__(self, config: ExMedBertConfig, output_size: int):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHeadVar(config, output_size)


class ExMedBertPlosHead(nn.Module):
    """Classification head for length of stay"""

    def __init__(self, config: ExMedBertConfig):
        super().__init__()
        self.los_classifier = nn.Linear(config.hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, last_hidden_state: torch.Tensor):
        input_tensor = torch.sum(last_hidden_state, 1)
        logits = self.los_classifier(input_tensor)
        return self.log_softmax(logits), logits


class ExMedBertForMaskedLM(ExMedBertModel):
    def __init__(
        self,
        config: ExMedBertConfig,
        code_embed: CodeDict,
    ):
        """Initialize ExMed-BERT model (pre-training)

        Args:
            config (ExMedBertConfig): Config instance
            code_embed (Optional[CodeDict], optional): CodeDict instance. Defaults to None.
        """
        super(BertPreTrainedModel, self).__init__(config)
        self.bert = ExMedBertModel(
            config,
            add_pooling_layer=False,
            code_embed=code_embed,
        )

        # init classifiers
        if config.predict_codes:
            self.code_classifier = BertOnlyMLMHeadVar(config, config.code_vocab_size)

        if config.predict_los:
            self.plos_classifier = ExMedBertPlosHead(config)

        self.config = config
        self.code_dict = code_embed

        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        entity_ids: Tensor,
        age_ids: Optional[Tensor] = None,
        sex_ids: Optional[Tensor] = None,
        state_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        code_labels: Optional[Tensor] = None,
        plos_label: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass of ExMed-BERT model (pre-training)

        Args:
            input_ids (Tensor): Input ids for codes (ATC+Phecodes)
            entity_ids (Tensor): Entity ids
            age_ids (Optional[Tensor], optional): Age ids. Defaults to None.
            sex_ids (Optional[Tensor], optional): Sex ids. Defaults to None.
            state_ids (Optional[Tensor], optional): State ids. Defaults to None.
            position_ids (Optional[Tensor], optional): Position ids. Defaults to None.
            attention_mask (Optional[Tensor], optional): Attention mask. Defaults to None.
            code_labels (Optional[Tensor], optional): Reference labels. Defaults to None.
            plos_label (Optional[Tensor], optional): Reference labels. Defaults to None.

        Returns:
            Dict[str, Any]: Output with loss, attention, etc.
        """

        last_hidden = self.bert(
            input_ids=input_ids,
            entity_ids=entity_ids,
            age_ids=age_ids,
            sex_ids=sex_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state_ids=state_ids,
        )[0]

        code_prediction_scores = self.code_classifier(last_hidden)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        plos_fct = nn.CrossEntropyLoss(
            weight=torch.Tensor(self.config.plos_weight).to(device=input_ids.device)
        )

        if code_labels is not None:
            code_loss = loss_fct(
                code_prediction_scores.view(-1, self.config.code_vocab_size),
                code_labels.view(-1),
            )

            output_dict = {
                "loss": code_loss,
                "logits": code_prediction_scores,
            }
        else:
            output_dict = {"logits": code_prediction_scores}

        if hasattr(self, "plos_classifier"):
            plos_logprobs, plos_logits = self.plos_classifier(last_hidden)
            output_dict["plos_logits"] = plos_logits

            if plos_label is not None:
                if plos_label.device != input_ids.device:
                    plos_label = plos_label.to(input_ids.device)

                plos_loss = plos_fct(plos_logits, plos_label)

                output_dict["plos_loss"] = plos_loss
                output_dict["loss"] += plos_loss

        return output_dict


class SequenceClassificationHead(nn.Module):
    """Classification head"""

    def __init__(self, config: ExMedBertConfig):
        super().__init__()

        if config.num_endpoints is None:
            raise Exception("Please specify the number of endpoints in the config")
        self.los_classifier = nn.Linear(config.hidden_size, config.num_endpoints)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        observations: Optional[torch.Tensor] = None,
    ):
        input_tensor = torch.sum(last_hidden_state, 1)
        if observations is not None:
            input_tensor = torch.cat((input_tensor, observations), dim=1)
        logits = self.los_classifier(input_tensor)
        return self.sigmoid(logits), logits


# TODO: remove in published version
def check_config(config, key):
    """Helper function"""
    if not hasattr(config, key):
        args = config.to_dict()
        args[key] = False
        config = ExMedBertConfig(args)
    return config


class ExMedBertForSequenceClassification(ExMedBertModel):
    def __init__(
        self,
        config: ExMedBertConfig,
        code_embed: Optional[CodeDict] = None,
    ):
        """Initialize ExMed-BERT model

        Args:
            config (ExMedBertConfig): Config instance
            code_embed (Optional[CodeDict], optional): CodeDict instance. Defaults to None.
        """
        super(BertPreTrainedModel, self).__init__(config)

        assert (
            config.num_endpoints is not None
        ), "Number of endpoints has to be specified"

        self.bert = ExMedBertModel(
            config,
            add_pooling_layer=False,
            code_embed=code_embed,
        )

        cell = None
        self.use_rnn = True
        if config.classification_head == "lstm":
            cell = nn.LSTM
        elif config.classification_head == "gru":
            cell = nn.GRU
        elif config.classification_head == "rnn":
            cell = nn.RNN
        else:
            self.use_rnn = False

        if not self.use_rnn:
            self.endpoint_classifier = SequenceClassificationHead(config)
        else:
            assert cell is not None
            self.rnn_cell = cell(
                input_size=config.hidden_size,
                hidden_size=config.rnn_hidden_size,
                num_layers=config.rnn_num_layers,
                dropout=0.1,
                bidirectional=True,
                batch_first=True,
            )
            self.out = nn.Linear(config.rnn_hidden_size * 2, 1)
            self.log_softmax = nn.LogSoftmax(dim=-1)

        self.config = config
        self.pos_weight = (
            torch.tensor(config.bc_pos_weight)
            if "bc_pos_weight" in config.to_dict().keys()
            and config.bc_pos_weight is not None
            else None
        )
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        entity_ids: Optional[Tensor] = None,
        age_ids: Optional[Tensor] = None,
        sex_ids: Optional[Tensor] = None,
        state_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        endpoint_labels: Optional[Tensor] = None,
        one_hot_input_ids: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass of ExMed-BERT model

        Args:
            input_ids (Tensor): Input ids for codes (ATC+Phecodes)
            entity_ids (Tensor): Entity ids
            age_ids (Optional[Tensor], optional): Age ids. Defaults to None.
            sex_ids (Optional[Tensor], optional): Sex ids. Defaults to None.
            state_ids (Optional[Tensor], optional): State ids. Defaults to None.
            position_ids (Optional[Tensor], optional): Position ids. Defaults to None.
            attention_mask (Optional[Tensor], optional): Attention mask. Defaults to None.
            endpoint_labels (Optional[Tensor], optional): Endpoint labels. Defaults to None.
            one_hot_input_ids (Optional[Tensor], optional): One hot input ids. Usually not required. Defaults to None.
            output_attentions (bool, optional): Indicate whether the attention should be returned. Defaults to False.

        Raises:
            Exception

        Returns:
            Dict[str, Any]: Output with loss, attention, etc.
        """

        output = self.bert(
            input_ids=input_ids,
            entity_ids=entity_ids,
            age_ids=age_ids,
            sex_ids=sex_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state_ids=state_ids,
            one_hot_input_ids=one_hot_input_ids,
            output_attentions=output_attentions,
        )

        if output_attentions:
            last_hidden, attention = output
        else:
            last_hidden = output[0]
            attention = None

        if not self.use_rnn:
            _, logits = self.endpoint_classifier(last_hidden)
        elif self.use_rnn:
            output, _ = self.rnn_cell(last_hidden)
            logits = self.out(output[:, -1, :])
        else:
            raise Exception

        loss = None
        if endpoint_labels is not None:
            if self.config.num_endpoints is not None and self.config.num_endpoints >= 1:
                loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                loss = loss_fct(logits, endpoint_labels.float())
            else:
                raise Exception("Prediction error")

        out = {"loss": loss, "logits": logits}

        if output_attentions:
            out["attention"] = attention

        return out


class CombinedSequenceClassificationHead(nn.Module):
    """Classification head"""

    def __init__(
        self, hidden_dim: int, dropout: float, num_endpoints: int, num_final_blocks: int
    ):
        super().__init__()

        def get_inner_block(idx: int):
            return [
                (f"cls_drop{idx}", nn.Dropout(dropout)),
                (f"cls_linear{idx}", nn.Linear(hidden_dim, hidden_dim)),
                (f"cls_relu{idx}", nn.ReLU()),
            ]

        final_block = [
            (f"cls_drop-fin", nn.Dropout(dropout)),
            (f"cls_linear-fin", nn.Linear(hidden_dim, num_endpoints)),
        ]

        cls_elements = []
        for i in range(num_final_blocks):
            cls_elements.extend(get_inner_block(i))
        cls_elements.extend(final_block)

        self.los_classifier = nn.Sequential(OrderedDict(cls_elements))
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self, input_tensor: torch.Tensor, observations: Optional[torch.Tensor] = None
    ):
        """Forward pass in classification head"""
        if observations is not None:
            input_tensor = torch.cat((input_tensor, observations), dim=1)
        logits = self.los_classifier(input_tensor)
        return self.log_softmax(logits), logits


class CombinedModelForSequenceClassification(ExMedBertModel):
    def __init__(
        self,
        config: CombinedConfig,
        code_embed: Optional[CodeDict] = None,
    ):
        """Initialize ExMed-BERT model for use with quantitative clinical measures

        Args:
            config (CombinedConfig): Config instance
            code_embed (Optional[CodeDict], optional): CodeDict instance. Defaults to None.
        """
        super(BertPreTrainedModel, self).__init__(config)

        assert (
            config.num_endpoints is not None
        ), "Number of endpoints has to be specified"

        self.bert = ExMedBertModel(
            config,
            add_pooling_layer=False,
            code_embed=code_embed,
        )

        cell = None
        self.use_rnn = True
        if config.classification_head == "lstm":
            cell = nn.LSTM
        elif config.classification_head == "gru":
            cell = nn.GRU
        elif config.classification_head == "rnn":
            cell = nn.RNN
        else:
            self.use_rnn = False

        if not self.use_rnn:
            self.endpoint_classifier = CombinedSequenceClassificationHead(
                config.hidden_size + config.num_observations,
                config.hidden_dropout_prob,
                config.num_endpoints,
                config.num_final_blocks,
            )
        else:
            assert cell is not None
            self.rnn_cell = cell(
                input_size=config.hidden_size,
                hidden_size=config.rnn_hidden_size,
                num_layers=config.rnn_num_layers,
                dropout=0.1,
                bidirectional=True,
                batch_first=True,
            )
            self.out = CombinedSequenceClassificationHead(
                (config.rnn_hidden_size * 2) + config.num_observations,
                config.hidden_dropout_prob,
                config.num_endpoints,
                config.num_final_blocks,
            )

        self.pos_weight = (
            torch.tensor(config.bc_pos_weight)
            if config.bc_pos_weight is not None
            else None
        )
        self.config = config
        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        entity_ids: Tensor,
        observation_input: Optional[Tensor] = None,
        age_ids: Optional[Tensor] = None,
        sex_ids: Optional[Tensor] = None,
        state_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        endpoint_labels: Optional[Tensor] = None,
        one_hot_input_ids: Optional[Tensor] = None,
        output_attentions: bool = False,
        iptw_score: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass of combined ExMed-BERT model

        Args:
            input_ids (Tensor): Input ids for codes (ATC+Phecodes)
            entity_ids (Tensor): Entity ids
            observation_input (Optional[Tensor], optional): Quantitative clinical measures. Defaults to None.
            age_ids (Optional[Tensor], optional): Age ids. Defaults to None.
            sex_ids (Optional[Tensor], optional): Sex ids. Defaults to None.
            state_ids (Optional[Tensor], optional): State ids. Defaults to None.
            position_ids (Optional[Tensor], optional): Position ids. Defaults to None.
            attention_mask (Optional[Tensor], optional): Attention mask. Defaults to None.
            endpoint_labels (Optional[Tensor], optional): Endpoint labels. Defaults to None.
            one_hot_input_ids (Optional[Tensor], optional): One hot input ids. Usually not required. Defaults to None.
            output_attentions (bool, optional): Indicate whether the attention should be returned. Defaults to False.
            iptw_score (Optional[Tensor], optional): IPTW score for respective samples. Defaults to None.

        Raises:
            Exception

        Returns:
            Dict[str, Any]: Output with loss, attention, etc.
        """

        if self.config.num_observations != 0 and observation_input is None:
            raise Exception("Set num_observations to 0 if it is not used.")

        bert_output = self.bert(
            input_ids=input_ids,
            entity_ids=entity_ids,
            age_ids=age_ids,
            sex_ids=sex_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state_ids=state_ids,
            one_hot_input_ids=one_hot_input_ids,
            output_attentions=output_attentions,
        )

        if output_attentions:
            last_hidden, attention = bert_output
        else:
            last_hidden = bert_output[0]
            attention = None

        if not self.use_rnn:
            _, logits = self.endpoint_classifier(
                torch.sum(last_hidden, 1), observation_input
            )
        elif self.use_rnn:
            output, _ = self.rnn_cell(last_hidden)
            _, logits = self.out(output[:, -1, :], observation_input)
        else:
            raise Exception

        loss = None
        if endpoint_labels is not None:
            if self.config.num_endpoints is not None and self.config.num_endpoints >= 1:
                if iptw_score is not None:
                    iptw_score = iptw_score.view(-1, 1)
                loss_fct = BCEWithLogitsLoss(
                    pos_weight=self.pos_weight, weight=iptw_score
                )
                loss = loss_fct(logits, endpoint_labels.float())
            else:
                raise Exception("Prediction error")

        out = {"loss": loss, "logits": logits}

        if output_attentions:
            out["attention"] = attention

        return out
