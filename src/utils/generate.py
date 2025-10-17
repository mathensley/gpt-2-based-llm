import os, torch
from .tokenizer import text_to_token_ids, token_ids_to_text


def generate_text(model, idx, max_new_tokens, context_size=50, reset=False):
    model.eval()
    reset = False

    try:
        for _ in range((max_new_tokens + 1) if reset else max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond, reset) if reset else model(idx_cond)
                if logits == None: continue

            reset = False

            logits = logits[:, -1, :]
            probas = torch.nn.functional.softmax(logits, dim=-1)
            idx_next_token = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next_token), dim=1)

        return idx

    except Exception as e:
        print(e)
        return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_embeddings.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer, device).to(device)
    with torch.no_grad():
        token_ids = generate_text(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"Amostra Gerada: '{decoded_text.replace(os.linesep, ' ')}'")
    model.train()