#!/usr/bin/env python3
"""
Tokenizer subprocess for the native GLM engine.

Communicates with the C process via JSON on stdin/stdout:
  encode  -> {"type":"encode","text":"hello world"}
  decode  -> {"type":"decode","ids":[123,456]}
  encode_messages -> {"type":"encode_messages","messages":[...]}

Responses:
  {"type":"encode","ids":[...]}
  {"type":"decode","text":"..."}
  {"type":"error","message":"..."}
"""

import json
import sys
import os

# Resolve tokenizer from the original NVFP4 model on disk
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
MODEL_PATH = os.path.join(
    HF_CACHE,
    "models--GadflyII--GLM-4.7-Flash-NVFP4",
    "snapshots", "3cdd7f37b66dc9063aacb0418e74041fe5316984"
)

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=True
        )
    return _tokenizer

def apply_chat_template(messages):
    t = get_tokenizer()
    # GLM chat template uses [gMASK]<sop> prefix
    text = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text

def do_encode(text):
    t = get_tokenizer()
    ids = t.encode(text, add_special_tokens=True)
    return {"type": "encode", "ids": ids}

def do_decode(ids):
    t = get_tokenizer()
    text = t.decode(ids, skip_special_tokens=True)
    return {"type": "decode", "text": text}

def do_encode_messages(messages):
    t = get_tokenizer()
    text = apply_chat_template(messages)
    ids = t.encode(text, add_special_tokens=True)
    return {"type": "encode_messages", "ids": ids, "text": text}

def do_vocab_info():
    t = get_tokenizer()
    return {
        "type": "vocab_info",
        "vocab_size": t.vocab_size,
        "eos_token_id": t.eos_token_id,
        "pad_token_id": t.pad_token_id if t.pad_token else None,
    }

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"type": "error", "message": "invalid JSON"}), flush=True)
            continue

        try:
            t = req.get("type", "")
            if t == "encode":
                result = do_encode(req.get("text", ""))
            elif t == "decode":
                result = do_decode(req.get("ids", []))
            elif t == "encode_messages":
                result = do_encode_messages(req.get("messages", []))
            elif t == "vocab_info":
                result = do_vocab_info()
            else:
                result = {"type": "error", "message": f"unknown type: {t}"}
        except Exception as e:
            result = {"type": "error", "message": str(e)}

        print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()