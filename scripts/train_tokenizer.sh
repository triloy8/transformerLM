uv run -m transformerlm.tokenizer.train_tokenizer \
  --input_path ./data/TinyStoriesV2-GPT4-valid.txt \
  --vocab_size 300 \
  --special_tokens "<|endoftext|>" \
  --merges_path ./data/custom_merges.txt \
  --vocab_path ./data/custom_vocab.json
