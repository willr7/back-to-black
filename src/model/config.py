from pathlib import Path


def get_config():

    return {
        "batch_size": 7,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 135,
        "d_model": 512,
        "lang_src": "AAVE",
        "lang_tgt": "SAE",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "model/tokenizer_{0}.json",
        "data_folder": "data/",
        "experiment_name": "runs/aave_to_sae",
    }


def get_weights_file_path(config, epoch: str):

    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path(".") / model_folder / model_filename)
