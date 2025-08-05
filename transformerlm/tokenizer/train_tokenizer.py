from transformerlm.tokenizer.tokenizer_utils import (gpt2_bytes_to_unicode,
                                                     find_chunk_boundaries,
                                                     process_chunk_text,)
import os
import regex as re
from collections import Counter
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

     # - initialize vocab with <|endoftext|> and gpt2 encoder
    vocab_list = special_tokens + [gpt2_byte_encoder[k] for k in gpt2_byte_encoder.keys()]
    initial_vocab = {k:i for i, k in enumerate(vocab_list)}

    # - open corpus, remove special token and pretokenize on chunks
    num_processes = 1
    pretoken_frequency = Counter()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f,
                                           num_processes,
                                           "<|endoftext|>".encode("utf-8"))
        
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    partial(process_chunk_text, input_path=input_path, special_tokens=special_tokens),
                    zip(boundaries[:-1], boundaries[1:])
                )
            )
        for partial_pretoken_frequency in results:
            pretoken_frequency.update(partial_pretoken_frequency)

    pretoken_frequency_dict = dict(pretoken_frequency)
    pretoken_frequency_tuple = {
        tuple(gpt2_byte_encoder[b] for b in word.encode('utf-8')): freq
        for word, freq in pretoken_frequency_dict.items()
    }

    # - compute bpe merges until vocab size is reached
    merges = []
    vocab = initial_vocab
    iter = 0
    iter_max = vocab_size - len(list(vocab.keys()))
    while iter < iter_max:
        # compute pair frequency
        pair_frequency = {}
        for freq_tuple, freq_value in pretoken_frequency_tuple.items():
            for i in range(len(freq_tuple)-1):
                pair = (freq_tuple[i], freq_tuple[i+1])
                if pair in pair_frequency: 
                    pair_frequency[pair] = pair_frequency[pair] + freq_value
                elif not pair in pair_frequency:
                    pair_frequency[pair] = freq_value
        
        # get biggest freq and update merges
        max_value = max(pair_frequency.values())
        max_pair_freq = [k for k, v in pair_frequency.items() if v == max_value]
        if len(max_pair_freq)>1:
            best_key = max(max_pair_freq, key=lambda p: ("".join([chr(gpt2_byte_decoder[c]) for c in p[0]]), "".join([chr(gpt2_byte_decoder[c]) for c in p[1]])))
        else:
            best_key = max_pair_freq[0]

        merges.append(best_key[0]+" "+best_key[1])

        # update pretoken frequency tuples with new merges
        new_merge = ''.join(best_key)
        new_pretoken_frequency_tuple = defaultdict(int)

        for key, value in pretoken_frequency_tuple.items():
            new_key = []
            i = 0
            while i < len(key):
                # Check for the merge pair
                if key[i:i+2] == best_key:
                    new_key.append(new_merge)
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            new_pretoken_frequency_tuple[tuple(new_key)] += value

        pretoken_frequency_tuple = dict(new_pretoken_frequency_tuple)

        iter += 1

    # - return vocab w/ additional merges and merges as bytes
    initial_vocab_size = len(list(vocab.keys()))
    for i, merge in enumerate(merges):
        merge_token = "".join(merge.split())
        vocab[merge_token] = initial_vocab_size + i
    bytes_vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in vocab.items()
    }
    
    bytes_merges = [
        (
            bytes([gpt2_byte_decoder[c] for c in merge.split()[0]]),
            bytes([gpt2_byte_decoder[c] for c in merge.split()[1]])
        )
        for merge in merges
    ]

    return (bytes_vocab, bytes_merges)