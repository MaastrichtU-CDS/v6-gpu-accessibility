Implementation
==============

Overview
--------

Central
-------
The central part is responsible for the orchestration and aggregation of the algorithm.

``central``
~~~~~~~~~~~
The central function sends the subtask that is executed on the nodes and returns the nodes' GPU information.

Partials
--------
Partials are the computations that are executed on each node. The partials have access
to the data that is stored on the node. The partials are executed in parallel on each
node.

``partial``
~~~~~~~~~~~
The partial function retrieves GPU accessibility details for TensorFlow and PyTorch through the respective helper functions.

``get_tensorflow_gpu_details``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function checks if TensorFlow is accessible on the GPU.
If a GPU is available it returns the GPU device name(s), memory, cores, and compute capability.

``get_pytorch_gpu_details``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function checks if PyTorch is accessible on the GPU.
If a GPU is available it returns the GPU device name(s), memory, cores, and compute capability.
