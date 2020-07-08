import torch
from mmcv.runner import _load_checkpoint, load_state_dict
from collections import OrderedDict

def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

    # duplicate R channel
    model_dict = model.state_dict()
    key = None
    for name in ['conv1.0.weight', 'conv1.weight']:
        if name in model_dict and name in state_dict:
            key = name
    if key:
        if state_dict[key].shape[1] < model_dict[key].shape[1]:
            print(state_dict[key].shape[1], model_dict[key].shape[1])
            v = state_dict[key]
            tensor = torch.zeros(model_dict[key].shape, dtype=torch.float32)
            for i in range(model_dict[key].shape[1] - state_dict[key].shape[1]):
                tensor[:, i, :, :] = v[:, 0, :, :]
            tensor[:, i+1:, :, :] = v
            state_dict[key] = tensor
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint
