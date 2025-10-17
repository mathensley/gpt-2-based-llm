import torch, time, wandb

from .eval import calc_loss_batch_by_cross_entropy, evaluate_model
from ..utils.plots import plot_graph
from ..utils.generate import generate_and_print_sample
from ..utils.wandb_utils import fetch_weights_and_bias, load_weights_and_bias


def train_aux(
        wandb_run, model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, 
        start_context, tokenizer, save_freq_wdb, file_name, save_wdb, state_dict, name, start_time
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    epochs_complete = state_dict.get("epoch", 0)
    batchs_complete = state_dict.get("batch", 0)
    accumulated_time = state_dict.get("train_time", 0)

    for epoch in range(epochs_complete, num_epochs):
        model.train()

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            if epoch == epochs_complete and batch_idx < batchs_complete:
                continue

            optimizer.zero_grad()
            loss = calc_loss_batch_by_cross_entropy(
                model,
                input_batch,
                target_batch,
                device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                wandb_run.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "tokens_seen": tokens_seen,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                })

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

            # salva peso só após o primeiro batch
            if save_wdb and global_step > 0 and global_step % save_freq_wdb == 0:
                elapsed_time = time.time() - start_time + accumulated_time
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "batch": batch_idx,
                    "train_time": elapsed_time
                },  file_name)
                artifact = wandb.Artifact(name, type="model")
                artifact.add_file(file_name)
                wandb_run.log_artifact(artifact)
        print("\nEXEMPLO DE GERAÇÃO:")
        generate_and_print_sample(model, tokenizer, device, start_context)

        # salva após finalizar o treino
        elapsed_time = time.time() - start_time + accumulated_time
        if save_wdb:
          torch.save({
              "epoch": epoch,
              "model_state": model.state_dict(),
              "optimizer_state": optimizer.state_dict(),
              "batch": batch_idx,
              "train_time": elapsed_time
          },  file_name)
          artifact = wandb.Artifact(file_name.split(".")[0] + "_final", type="model")
          artifact.add_file(file_name)
          wandb_run.log_artifact(artifact)

    return train_losses, val_losses, track_tokens_seen, elapsed_time


def run_train(model, optimizer, config, train_loader, val_loader, tokenizer, device, start_context="Bom dia!"):
    run = wandb.init(project=config["project"], name=config["name"], id=config["run_id"], resume="allow")
    res_fetch = fetch_weights_and_bias(
        user=config["user"],
        project=config["project"],
        name=config["name"],
        version=config["version"],
        file_name=config["file_name"]
    )

    state_dict = {}
    if res_fetch:
        loaded, checkpoint = load_weights_and_bias(
            file_name=config["file_name"]
        )
        if loaded:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            state_dict["epoch"] = checkpoint.get("epoch", 0)
            state_dict["batch"] = checkpoint.get("batch", 0)
            state_dict["train_time"] = checkpoint.get("train_time", 0.0)
            print("Pesos carregados com sucesso!")

    print("\nEPOCHS/BATCHS RECUPERADOS: ", state_dict)
    num_epochs = config["max_epochs"]

    start_time = time.time()
    train_losses, val_losses, tokens_seen, total_train_time = train_aux(
        run, model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=start_context, tokenizer=tokenizer,
        save_freq_wdb=config["save_freq_wdb"], file_name=config["file_name"],
        save_wdb=config["save_wdb"], state_dict=state_dict, 
        name=config["name"], start_time=start_time
    )
    print("\nGRÁFICO DE PERDA DURANTE O TREINO:")
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_graph(epochs_tensor, tokens_seen, train_losses, val_losses)
    return tokens_seen[-1], total_train_time