import os, wandb, torch


def fetch_weights_and_bias(user, project, name, version, file_name):
    """
    Busca um artifact de pesos salvo no W&B.
    Exemplo de version: 'v1', 'v2', etc (não o run id).
    """
    try:
        api = wandb.Api()
        artifact = api.artifact(f"{user}/{project}/{name}:{version}", type="model")
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, file_name)
        print(f"Fetch success -> {file_path}")
        return True

    except Exception as e:
        print(f"Fetch Error: {e}")
        return False


def load_weights_and_bias(file_name):
    try:
        checkpoint = torch.load(file_name, map_location="cpu")
        return True, checkpoint
    except Exception as e:
        print(f"Load error: {e}")
        return False, {}


def load_latest_model(config, model_class, device):
    api = wandb.Api()
    artifact = api.artifact(
        f"{config['user']}/{config['project']}/{config['name']}:latest", type="model"
    )
    artifact_dir = artifact.download()

    checkpoint_path = os.path.join(artifact_dir, config["file_name"])
    print(f"Carregando pesos do checkpoint: {checkpoint_path}")

    # Carregar o checkpoint completo
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Inicializa modelo com a mesma config
    model = model_class(config, device).to(device)

    model.load_state_dict(checkpoint["model_state"])
    print("Pesos carregaos com sucesso!")

    return model