"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""
from typing import Any

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient


@algorithm_client
def central(client: AlgorithmClient, organisation_ids: list) -> Any:
    """
    Executes the central algorithm function on a node.

    This function collects all organisations participating in the collaboration,
    creates a subtask for GPU accessibility assessment, waits for the results,
    and returns the results.

    Args:
        client (AlgorithmClient): The algorithm client instance.
        organisation_ids (list): List of organisation IDs to include in the subtask.
                                 If not provided, all organisations in the collaboration are included.

    Returns:
        Any: The results of the GPU accessibility assessment.
    """
    # Collect all organisations that participate in this collaboration unless specified
    if isinstance(organisation_ids, list) is False:
        organisations = client.organization.list()
        ids = [organisation.get("id") for organisation in organisations]
    else:
        ids = organisation_ids

    # Create a subtask for all organisations in the collaboration.
    info("Creating subtask for all organisations in the collaboration")
    task = client.task.create(
        input_={"method": "partial",
                "kwargs": {}},
        organizations=ids,
        name="GPU accessibility assessment",
        description="This is task assesses whether GPUs are available inside of the algorithm's Docker container."
    )

    # Wait for the nodes to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    return results
