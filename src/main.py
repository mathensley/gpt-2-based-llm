import torch, time

from .utils import get_loaders, TOKENIZER
from .model.gpt2_original import GPT2ModelOriginal
from .train.run import run_train

from dotenv import load_dotenv

load_dotenv(override=True)

torch.manual_seed(40)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Usando dispositivo: {device.type}")

print(TOKENIZER.n_vocab)


CONFIG = {
    "vocab_size": TOKENIZER.n_vocab,
    "embedding_dim": 512,
    "context_length": 256,
    "num_layers": 8,
    "num_heads": 8,
    "bias": False,
    
    "batch_size": 16,
    "max_epochs": 1,
    "num_workers": 2,
    "stride": 128,
    "dtype": torch.float32,
    "device": device,

    "eval_freq": 5,
    "eval_iter": 5,

    "save_wdb": True,
    "save_freq_wdb": 5000,
    "user": "matheus-figueiredo-silva-ufcg",
    "project": "gpt2-test",
    "name": "test1",
    "run_id": "gpt2-test-run1",
    "version": "v0",
    "file_name": "mini_mlp.pth"
}

# train_loader, test_loader, val_loader = get_loaders(
#     data_path="data/processed",
#     tokenizer=TOKENIZER,
#     max_length=CONFIG["context_length"],
#     batch_sz=CONFIG["batch_size"],
#     num_workers=CONFIG["num_workers"],
#     stride=CONFIG["stride"]
# )

# print(f"Tamanho do conjunto de treinamento: {len(train_loader)}\nTamanho do conjunto de teste: {len(test_loader)}\nTamanho do conjunto de validação: {len(val_loader)}")


# model = GPT2ModelOriginal(CONFIG, device).to(device=device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# params = sum(p.numel() for p in model.parameters())
# params_model = params - sum(p.numel() for p in model.out_head.parameters())
# print(f"Número de parâmetros (sem head): {params_model:,}")

# if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)

# start_time = time.time()
# tokens_processed, total_train_time = run_train(model=model, optimizer=optimizer, config=CONFIG, train_loader=train_loader, val_loader=val_loader, tokenizer=TOKENIZER, device=device)
# end_time = time.time()

# elapsed = end_time - start_time
# tokens_per_sec = tokens_processed / elapsed
# max_memory = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB

# print("\nDESEMPENHO:")
# print(f"Tempo total: {elapsed:.2f} s")
# print(f"Tokens/s: {tokens_per_sec:.2f}")
# print(f"Memória máxima: {max_memory:.2f} MB")

# print(f"\nTempo total de treino: {total_train_time:.2f} s ({total_train_time/60:.2f} min)")