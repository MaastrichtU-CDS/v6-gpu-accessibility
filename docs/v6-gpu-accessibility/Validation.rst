Validation
==========
Consider evaluating whether your GPU is accessible outside of the Docker container in case you have issues with the GPU.
The following steps can be taken to verify the GPU accessibility:

.. code-block:: python

  import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print('No GPUs available for TensorFlow')
    else:
        print({
               "number of GPUs": len(gpus),
               "memory": f'{[tf.config.experimental.get_memory_info(gpu.name)['current'] for gpu in gpus]} bytes,
               "cores": [tf.config.experimental.get_device_details(gpu)['core_count'] for gpu in gpus],
               "compute capability": [tf.config.experimental.get_device_details(gpu)['compute_capability'] for gpu in gpus]
              })
