Validation
==========
Consider evaluating whether your GPU is accessible outside of the Docker container in case you have issues with the GPU.
The following steps can be taken to verify the GPU accessibility:

.. code-block:: python

  import tensorflow as tf
  import torch

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print('No GPUs available for TensorFlow')
    else:
     print({
        "number of GPUs": len(gpus),
        "names": [gpu.name for gpu in gpus],
        "memory": f'{[tf.config.experimental.get_memory_info(gpu.name)['current'] for gpu in gpus]} bytes,
        "cores": [tf.config.experimental.get_device_details(gpu)['core_count'] for gpu in gpus],
        "compute capability": [tf.config.experimental.get_device_details(gpu)['compute_capability'] for gpu in gpus]
    })

    if not torch.cuda.is_available():
        print('No GPUs available for PyTorch')
    else:
     print({
        "number of GPUs": torch.cuda.device_count(),
        "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "memory": f'{[torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]} bytes',
        "cores": [torch.cuda.get_device_properties(i).multi_processor_count for i in range(torch.cuda.device_count())],
        "compute capability": [torch.cuda.get_device_properties(i).major for i in range(torch.cuda.device_count())]
    })

