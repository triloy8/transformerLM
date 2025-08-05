uv run -m transformerlm.data.make_data \
    --input_filename ./data/TinyStoriesV2-GPT4-valid.txt \
    --output_filename ./data/test.dat \
    --total_tokens 5535291 \
    --vocab_path ./data/gpt2_vocab.json \
    --merges_path ./data/gpt2_merges.txt \
    --special_tokens "<|endoftext|>" \