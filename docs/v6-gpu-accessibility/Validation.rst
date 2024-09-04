Validation
==========
Consider evaluating whether your GPU is accessible outside of the Docker container in case you have issues with the GPU.
The following steps can be taken to verify the GPU accessibility:

.. code-block:: python

  import torch


    if not torch.cuda.is_available():
        print('No GPUs available for PyTorch')
    else:
        print({
               "number of GPUs": torch.cuda.device_count(),
               "memory": f'{[torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]} bytes',
               "cores": [torch.cuda.get_device_properties(i).multi_processor_count for i in range(torch.cuda.device_count())],
               "compute capability": [torch.cuda.get_device_properties(i).major for i in range(torch.cuda.device_count())]
              })

