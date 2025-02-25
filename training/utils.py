import torch
import yaml
import os
import logging

def load_config(config_path="config.yaml"):
    """Loads configuration settings from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def setup_logger(log_file="results/logs/training.log"):
    """Configures the logger for training logs."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("MGAN_Training")

def save_model(model, path="results/mgan_model.pth"):
    """Saves the trained MGAN model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="results/mgan_model.pth"):
    """Loads a trained MGAN model from a saved checkpoint."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {path}")

def get_device(config):
    """Returns the appropriate device (GPU/CPU) based on config settings."""
    use_gpu = config["hardware"]["use_gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Using device: {device}")
    return device
