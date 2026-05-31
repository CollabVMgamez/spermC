import json
import sys
import struct

def convert(tokenizer_json_path, out_bin_path):
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vocab = {}
    if 'model' in data and 'vocab' in data['model']:
        vocab = dict(data['model']['vocab'])

    added = data.get('added_tokens', [])
    for item in added:
        if isinstance(item, dict) and 'content' in item and 'id' in item:
            vocab[item['content']] = item['id']
        elif isinstance(item, dict) and 'token' in item and 'id' in item:
            vocab[item['token']] = item['id']

    if not vocab:
        print("No vocab found in tokenizer.json")
        sys.exit(1)

    max_id = max(vocab.values())
    tokens = [b'' for _ in range(max_id + 1)]
    for s, idx in vocab.items():
        if 0 <= idx <= max_id:
            tokens[idx] = s.encode('utf-8')

    with open(out_bin_path, 'wb') as f:
        f.write(struct.pack('<I', len(tokens)))
        for t in tokens:
            if len(t) > 255:
                t = t[:255]
            f.write(struct.pack('B', len(t)))
            f.write(t)
    print(f"Wrote {len(tokens)} tokens to {out_bin_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_tokenizer.py tokenizer.json tokenizer.bin")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
