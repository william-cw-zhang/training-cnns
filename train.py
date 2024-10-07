"""Main training script. """
import importlib
import logging
from pathlib import Path
from typing import Any, Dict
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import yaml


__author__ = 'William Zhang'


def build_instance(blueprint: DictConfig, updates: Dict = None) -> Any:
    """Build an arbitrary class instance.

    Args:
        blueprint (DictConfig): config w/keys 'module', 'class_name', 'kwargs'
        updates (Dict): additional/non-default kwargs

    Returns:
        Any: instance defined by <module>.<class_name>(**kwargs)
    """
    module = importlib.import_module(blueprint.module)
    instance_kwargs = {}
    if 'kwargs' in blueprint:
        instance_kwargs.update(OmegaConf.to_container(blueprint.kwargs))
    if updates:
        instance_kwargs.update(updates)
    return getattr(module, blueprint.class_name)(**instance_kwargs)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    """Compute model metrics against a dataset.

    Args:
        model (nn.Module): model to evalate
        loader (DataLoader): dataset batch generator

    Returns:
        Dict[str, float]: metrics
    """
    num_correct, num_samp = 0, 0
    device = next(iter(model.parameters())).device
    for imgs, labels in tqdm(loader, leave=False):
        logits = model(imgs.to(device))
        pred = logits.argmax(dim=1)
        num_correct += (labels == pred.cpu()).long().sum().item()
        num_samp += len(labels)
    return {'top1': num_correct / num_samp}


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(config: DictConfig = None) -> None:
    """Train a classifier.

    Args:
        config (DictConfig): script configuration
    """
    logging.info('user note: %s', config.user_note)

    # set up output directories
    outdir = Path(HydraConfig.get().runtime.output_dir)
    torch.manual_seed(config.get('seed', 0))

    # load dataset and model
    logging.info('load dataset and model')
    preproc = ToTensor()
    data_valid = build_instance(
        config.dataset,
        {'train': False, 'transform': preproc}
    )
    loader_valid = DataLoader(
        data_valid,
        batch_size=config.batch_size,
        shuffle=False
    )
    model = build_instance(
        config.model,
        {'num_classes': len(data_valid.classes)}
    ).to(config.device)

    # export final weights to ONNX
    logging.info('export final weights to ONNX')
    torch.onnx.export(
        model,
        torch.rand(2, 3, 128, 128).to(config.device),
        outdir / 'final.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'row', 3: 'col'}
        }
    )

    # evaluate final model
    logging.info('evaluate final model')
    metrics = evaluate(model, loader_valid)
    with open(outdir / 'metrics.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(metrics, file)

    logging.info('main() ran to completion.')


if __name__ == '__main__':
    main()
