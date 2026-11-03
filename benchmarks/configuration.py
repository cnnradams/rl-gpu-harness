"""Common factual instructions for development trials and study manifests."""

from gpu_harness.harbor import instructions


def trial_instructions(seconds, limits=None):
    text = [f"The agent phase has an enforced wall time limit of {seconds} seconds.\n"]
    if limits is not None:
        text.append(instructions(limits))
        if not limits.shared_workspace:
            text.append(
                "GPU boxes start from the initial task image, including the supplied "
                "data and scripts. Later local edits are separate.\n"
            )
    return text
