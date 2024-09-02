"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""
import tensorflow as tf
import torch

from typing import Any

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient


@algorithm_client
def partial(client: AlgorithmClient) -> Any:
    """
    Retrieves GPU accessibility details for TensorFlow and PyTorch.

    Args:
        client (AlgorithmClient): The algorithm client instance.

    Returns:
        dict: A dictionary containing the organisation ID and GPU accessibility details for TensorFlow and PyTorch.
    """
    info("Retrieving details of any GPUs available to the algorithm's Docker container")
    return {
        'organisation_id': client.organization_id,
        'gpu_accessibility': {
            "Tensorflow": get_tensorflow_gpu_details(),
            "PyTorch": get_pytorch_gpu_details()
        }
    }


def get_tensorflow_gpu_details():
    """
    Retrieves details about available TensorFlow GPUs.

    Returns:
        dict or str: A dictionary containing the number of GPUs, their names, memory, cores,
                     and compute capability if GPUs are available. Otherwise, returns 'No GPUs available'.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return 'No GPUs available'
    return {
        "number of GPUs": len(gpus),
        "memory": f"{[(tf.config.experimental.get_memory_info(gpu.name)['current']/ (1024 * 1024)) for gpu in gpus]} MB",
        "cores": [tf.config.experimental.get_device_details(gpu)['core_count'] for gpu in gpus],
        "compute capability": [tf.config.experimental.get_device_details(gpu)['compute_capability'] for gpu in gpus]
    }


def get_pytorch_gpu_details():
    """
    Retrieves details about available PyTorch GPUs.

    Returns:
        dict or str: A dictionary containing the number of GPUs, their names, memory, cores,
                     and compute capability if GPUs are available. Otherwise, returns 'No GPUs available'.
    """
    if not torch.cuda.is_available():
        return 'No GPUs available'
    return {
        "number of GPUs": torch.cuda.device_count(),
        "memory": f"{[(torch.cuda.get_device_properties(i).total_memory/ (1024 * 1024)) for i in range(torch.cuda.device_count())]} MB",
        "cores": [torch.cuda.get_device_properties(i).multi_processor_count for i in range(torch.cuda.device_count())],
        "compute capability": [torch.cuda.get_device_properties(i).major for i in range(torch.cuda.device_count())]
    }
