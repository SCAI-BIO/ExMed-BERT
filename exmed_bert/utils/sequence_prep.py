# imports ----------------------------------------------------------------------
import collections
import copy
import logging
import random
from typing import List, Optional, Tuple, Union

import torch

from exmed_bert.data.encoding import CodeDict, EmbedDict

# global vars ------------------------------------------------------------------

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

# functions --------------------------------------------------------------------


def mask_token_ids(
    token_ids: torch.LongTensor,
    masked_lm_prob: float,
    max_predictions_per_seq: int,
    codes_to_use: List[int],
    keep_min_unmasked: int = 1,
    rng: Optional[random.Random] = None,
    codes_to_ignore=list(range(6)),
    mask_id=4,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Masks the sequence according to the BERT paper. The code is mainly based/copied from the official BERT GitHub
    repository.

    Args:
        token_ids (torch.LongTensor): Tensor with token ids
        masked_lm_prob (float): probability of masking
        max_predictions_per_seq (int): maximum number of masked tokens per sequence
        codes_to_use (List): ids to use for masking
        keep_min_unmasked (int, optional): minimum number of tokens to keep unmasked. Defaults to 1.
        rng (random.Random, optional): random number generator. Defaults to None.
        codes_to_ignore ([type], optional): codes to ignore for masking (e.g. defaults, substances).
        mask_id (int, optional): ids of mask token. Defaults to 4.

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor]: Tuple with masked codes and labels
    """

    if rng is None:
        rng = random.Random()

    output_tokens = copy.deepcopy(token_ids)
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.numpy()

    # Generate list of candidates
    logging.debug("Create masked version of sequence.")
    candidate_indices = [
        i for i, token_id in enumerate(token_ids) if token_id not in codes_to_ignore
    ]

    # Add additional randomness
    rng.shuffle(candidate_indices)

    num_to_predict = min(
        max_predictions_per_seq,
        max(1, int(round(len(candidate_indices) * masked_lm_prob))),
    )

    if (len(candidate_indices) - num_to_predict) <= keep_min_unmasked:
        diff = len(candidate_indices) - keep_min_unmasked
        if diff > 0:
            num_to_predict = diff
        else:
            num_to_predict = 0

    masked_lms: List[MaskedLmInstance] = []
    covered_indices = set()
    for index in candidate_indices:
        if len(masked_lms) >= num_to_predict:
            break

        covered_indices.add(index)
        if rng.random() < 0.8:
            masked_token = mask_id
        else:
            if rng.random() < 0.5:
                masked_token = token_ids[index]
            else:
                masked_token = rng.sample(list(codes_to_use), 1)[0]
        output_tokens[index] = masked_token
        masked_lms.append(MaskedLmInstance(index=index, label=token_ids[index]))

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    labels = [-100] * len(token_ids)
    for p in masked_lms:
        labels[p.index] = p.label  # type:ignore

    return output_tokens, torch.LongTensor(labels)


def create_masked_lm_predictions(
    tokens: List[str],
    masked_lm_prob: float,
    max_predictions_per_seq: int,
    vocab_words: List[str],
    keep_min_unmasked: int = 1,
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Masks the sequence according to the BERT paper. The code is mainly based/copied from the official BERT GitHub
    repository.

    Args:
        tokens (List[str]): List of input tokens
        masked_lm_prob (float): Probability of masking
        max_predictions_per_seq (int): maximum of masked tokens'
        vocab_words (List[str]): List of words in vocab
        keep_min_unmasked (int): minimum of unmasked tokens
        rng (random.Random): random number generator

    Returns: Tuple with output tokens, positions, and labels

    """
    if rng is None:
        rng = random.Random()

    # Generate list of candidates
    logging.debug("Create masked version of sequence.")
    candidate_indices = [
        i for i, token in enumerate(tokens) if token not in ["CLS", "SEP", "UNK", "NA"]
    ]

    # Add additional randomness
    rng.shuffle(candidate_indices)

    output_tokens = list(tokens)

    num_to_predict = min(
        max_predictions_per_seq,
        max(1, int(round(len(candidate_indices) * masked_lm_prob))),
    )

    if (len(candidate_indices) - num_to_predict) <= keep_min_unmasked:
        diff = len(candidate_indices) - keep_min_unmasked
        if diff > 0:
            num_to_predict = diff
        else:
            num_to_predict = 0

    masked_lms = []
    covered_indices = set()
    for index in candidate_indices:
        if len(masked_lms) >= num_to_predict:
            break

        covered_indices.add(index)
        if rng.random() < 0.8:
            masked_token = "MASK"
        else:
            if rng.random() < 0.5:
                masked_token = tokens[index]
            else:
                masked_token = rng.sample(list(vocab_words)[5:], 1)[0]
        output_tokens[index] = masked_token
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_tokens = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_tokens.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_tokens


def create_mlm_mask(
    masked_lm_positions: List[int],
    masked_lm_tokens: List[str],
    max_seq_len: int,
    dictionary: CodeDict,
    **kwargs: object
) -> List[int]:
    """
    Creates a mask for the transformers' trainer. If a word is masked, the actual token id is given. If not, the value
    is -100. Tokens with this value will not be included in the masked language modelling task.


    Args:
        masked_lm_positions (List[int]): Masked positions
        masked_lm_tokens (List[str]): Masked tokens
        max_seq_len (int): maximum sequence length
        dictionary (CodeDict): dictionary with codes in vocab
        **kwargs ():

    Returns:

    """
    logging.debug("Create mlm mask for training instance.")
    masked_lm_label = [-100] * max_seq_len
    for i, label in zip(masked_lm_positions, masked_lm_tokens):
        if type(label) is not str:
            label = str(label)
        masked_lm_label[i] = dictionary(label, **kwargs)  # type:ignore

    return masked_lm_label


def position_idx(visit_length: List[int], max_length: int) -> List[int]:
    """Generate position index


    Args:
        visit_length (List[int]): List with number of codes per visit
        max_length (int): maximum sequence length

    Returns:
        List[in]: Vector representation for positions
    """
    pos = []
    flag = 1

    for visit in visit_length:
        pos.extend([flag] * visit)
        flag += 1

    pos.extend([0] * (max_length - sum(visit_length)))

    return pos


def seq_padding(
    tokens: List[str],
    max_len: int,
    embed: Optional[EmbedDict] = None,
    symbol: str = "PAD",
    return_entities: bool = False,
    **kwargs: object
) -> Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]]:
    """Pad sequences to length n

    Args:
        tokens (List[str]): Token sequence
        max_len (int): Maximum sequence length
        embed (Optional[EmbedDict], optional): Instance of EmbedDict. Defaults to None.
        symbol (str): Padding symbol. Defaults to 'PAD'.
        return_entities (bool, optional): Indicate whether entity types should be returned. Defaults to False.

    Returns:
        Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]]
    """
    seq = []
    entities = []
    token_len = len(tokens)
    for i in range(max_len):
        if embed is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                tid, ent = embed.encode(tokens[i], return_entities=True, **kwargs)
            else:
                tid, ent = embed.encode(symbol, return_entities=True, **kwargs)

            seq.append(tid)
            entities.append(ent if not isinstance(ent, list) else ent[0])

    if return_entities:
        return torch.LongTensor(seq), torch.LongTensor(entities)
    else:
        return torch.LongTensor(seq)
