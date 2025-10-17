import torch, tiktoken

#TOKENIZER = tiktoken.get_encoding("o200k_base")
TOKENIZER = tiktoken.get_encoding("gpt2")

def text_to_token_ids(text, tokenizer, device):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor.to(device)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())