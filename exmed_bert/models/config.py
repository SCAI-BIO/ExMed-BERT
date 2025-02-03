from typing import Optional, Tuple

from transformers import PretrainedConfig  # type: ignore


class ExMedBertConfig(PretrainedConfig):
    """Model configuration for ExMedBertModel"""

    model_type = "exmedbert"

    def __init__(
        self,
        code_vocab_size: int = 100,
        number_of_codes: int = 3,
        age_vocab_size: int = 112,
        sex_vocab_size: int = 3,
        region_vocab_size: int = 52,
        hidden_size: int = 288,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 512,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 64,  # max + 1 for padding
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        gradient_checkpointing: bool = False,
        predict_codes: bool = True,
        predict_los: bool = False,
        plos_weight: Optional[Tuple[float, float]] = None,
        num_endpoints: Optional[int] = None,
        classification_head: str = "ffn",  # ffn, lstm
        initialization: str = "orthogonal",
        rnn_num_layers: int = 2,
        rnn_hidden_size: Optional[int] = None,
        bc_pos_weight: Optional[float] = None,
        **kwargs,
    ):
        """ExMed-BERT configuration

        Args:
            code_vocab_size (int, optional): Size of CodeDict. Defaults to 100.
            number_of_codes (int, optional): _description_. Defaults to 3.
            age_vocab_size (int, optional): Size of AgeDict. Defaults to 112.
            sex_vocab_size (int, optional): Size of SexDict. Defaults to 3.
            region_vocab_size (int, optional): Size of StateDict. Defaults to 52.
            hidden_size (int, optional): Hidden size of ExMed-BERT. Defaults to 288.
            num_hidden_layers (int, optional): Number of hidden layers in the ExMed-BERT model. Defaults to 6.
            num_attention_heads (int, optional): Number of attention heads in the ExMed-BERT model. Defaults to 12.
            intermediate_size (int, optional): Intermediate network size in the ExMed-BERT model. Defaults to 512.
            hidden_act (str, optional): Activation function. Defaults to "gelu".
            hidden_dropout_prob (float, optional): Dropout rate. Defaults to 0.1.
            attention_probs_dropout_prob (float, optional): Dropout for attention layer. Defaults to 0.1.
            max_position_embeddings (int, optional): Maximum number of positional embeddings (sequence length). Defaults to 64.
            layer_norm_eps (float, optional): Eps for layer normalization. Defaults to 1e-12.
            pad_token_id (int, optional): Padding token index. Defaults to 0.
            gradient_checkpointing (bool, optional): Indicate whether gradient checkpointing should be performed. Defaults to False.
            predict_codes (bool, optional): Predict codes (MLM). Defaults to True.
            predict_los (bool, optional): Predict PLOS. Defaults to False.
            plos_weight (Optional[Tuple[float, float]], optional): Weights for PLOS (negative and positive class). Defaults to None.
            num_endpoints (Optional[int], optional): Number of endpoints for fine-tuning. Defaults to None.
            classification_head (str, optional): Type of classification head ("ffn", "gru", "lstm"). Defaults to "ffn".
            lstminitialization (str, optional): Type of LSTM initialization. Defaults to "orthogonal".
            rnn_num_layers (int, optional): Number of RNN layers (if gru or lstm). Defaults to 2.
            rnn_hidden_size (Optional[int], optional): Hidden size of RNN (if gru or lstm). Defaults to None.
            bc_pos_weight (Optional[float], optional): Weights for binary classification. Defaults to None.
        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # MedBERT specific configuration (general)
        self.code_vocab_size = code_vocab_size
        self.number_of_codes = number_of_codes
        self.age_vocab_size = age_vocab_size
        self.sex_vocab_size = sex_vocab_size
        self.region_vocab_size = region_vocab_size

        # MedBERT specific configuration (pretraining)
        self.predict_codes = predict_codes
        self.predict_los = predict_los
        self.plos_weight = plos_weight

        # MedBERT specific configuration (training)
        self.num_endpoints = num_endpoints
        self.classification_head = classification_head
        self.bc_pos_weight = bc_pos_weight

        # "normal" configuration for transformers model
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embedding = max_position_embeddings
        self.initializer_range = initializer_range
        self.initialization = initialization
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing

        # configuration for rnn classification heads
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = (
            hidden_size if rnn_hidden_size is None else rnn_hidden_size
        )


class CombinedConfig(ExMedBertConfig):
    """Model configuration for CombinedModel"""

    model_type = "combined_exmedbert"

    def __init__(
        self,
        num_observations: int = 0,
        code_vocab_size: int = 100,
        number_of_codes: int = 3,
        age_vocab_size: int = 112,
        sex_vocab_size: int = 3,
        region_vocab_size: int = 52,
        hidden_size: int = 288,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 512,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 64,  # max + 1 for padding
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        gradient_checkpointing: bool = False,
        predict_codes: bool = True,
        predict_los: bool = False,
        plos_weight: Optional[Tuple[float, float]] = None,
        num_endpoints: Optional[int] = None,
        classification_head: str = "ffn",  # ffn, lstm
        initialization: str = "orthogonal",
        rnn_num_layers: int = 2,
        rnn_hidden_size: Optional[int] = None,
        bc_pos_weight: Optional[float] = None,
        num_final_blocks: int = 2,
        **kwargs,
    ):
        """ExMed-BERT configuration (plus quantitative clinical measures)

        Args:
            num_observations (int, optional): Number of quantitative variables. Defaults to 0.
            code_vocab_size (int, optional): Size of CodeDict. Defaults to 100.
            number_of_codes (int, optional): _description_. Defaults to 3.
            age_vocab_size (int, optional): Size of AgeDict. Defaults to 112.
            sex_vocab_size (int, optional): Size of SexDict. Defaults to 3.
            region_vocab_size (int, optional): Size of StateDict. Defaults to 52.
            hidden_size (int, optional): Hidden size of ExMed-BERT. Defaults to 288.
            num_hidden_layers (int, optional): Number of hidden layers in the ExMed-BERT model. Defaults to 6.
            num_attention_heads (int, optional): Number of attention heads in the ExMed-BERT model. Defaults to 12.
            intermediate_size (int, optional): Intermediate network size in the ExMed-BERT model. Defaults to 512.
            hidden_act (str, optional): Activation function. Defaults to "gelu".
            hidden_dropout_prob (float, optional): Dropout rate. Defaults to 0.1.
            attention_probs_dropout_prob (float, optional): Dropout for attention layer. Defaults to 0.1.
            max_position_embeddings (int, optional): Maximum number of positional embeddings (sequence length). Defaults to 64.
            layer_norm_eps (float, optional): Eps for layer normalization. Defaults to 1e-12.
            pad_token_id (int, optional): Padding token index. Defaults to 0.
            gradient_checkpointing (bool, optional): Indicate whether gradient checkpointing should be performed. Defaults to False.
            predict_codes (bool, optional): Predict codes (MLM). Defaults to True.
            predict_los (bool, optional): Predict PLOS. Defaults to False.
            plos_weight (Optional[Tuple[float, float]], optional): Weights for PLOS (negative and positive class). Defaults to None.
            num_endpoints (Optional[int], optional): Number of endpoints for fine-tuning. Defaults to None.
            classification_head (str, optional): Type of classification head ("ffn", "gru", "lstm"). Defaults to "ffn".
            lstminitialization (str, optional): Type of LSTM initialization. Defaults to "orthogonal".
            rnn_num_layers (int, optional): Number of RNN layers (if gru or lstm). Defaults to 2.
            rnn_hidden_size (Optional[int], optional): Hidden size of RNN (if gru or lstm). Defaults to None.
            bc_pos_weight (Optional[float], optional): Weights for binary classification. Defaults to None.
            num_final_blocks (int, optional): Number of layers in final FNN layer. Defaults to 2.
        """
        super().__init__(
            code_vocab_size=code_vocab_size,
            number_of_codes=number_of_codes,
            age_vocab_size=age_vocab_size,
            sex_vocab_size=sex_vocab_size,
            region_vocab_size=region_vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            predict_codes=predict_codes,
            predict_los=predict_los,
            plos_weight=plos_weight,
            num_endpoints=num_endpoints,
            classification_head=classification_head,
            initialization=initialization,
            rnn_num_layers=rnn_num_layers,
            rnn_hidden_size=rnn_hidden_size,
            bc_pos_weight=bc_pos_weight,
            **kwargs,
        )

        self.num_observations = num_observations
        self.num_final_blocks = num_final_blocks
