import json
import logging
from typing import Any, Optional

import torch
from torch._inductor.virtualized import V

from . import config as inductor_config


log = logging.getLogger(__name__)


def _in_use() -> bool:
    """
    Determine if the gemm config lookup table should be used.
    Returns True if either:
    1. The gemm_config_lookup_table_path is set, OR
    2. The gemm_config_lookup_table global has been set
    """
    return (
        inductor_config.triton.gemm_config_lookup_table_path is not None
        or gemm_config_lookup_table is not None
    )


def _get_lookup_table() -> Optional[dict[str, dict[str, dict[str, dict[str, str]]]]]:
    """
    Load and return the gemm config lookup table from file if configured.
    """
    global gemm_config_lookup_table

    if (
        _in_use()
        and inductor_config.triton.gemm_config_lookup_table_path is not None
        and gemm_config_lookup_table is None
    ):
        file_path = inductor_config.triton.gemm_config_lookup_table_path

        # Assert supported file extensions
        if file_path.endswith(".json"):
            with open(file_path) as f:
                gemm_config_lookup_table = json.load(f)
        elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "PyYAML is required to load YAML files. Install with: pip install PyYAML, or convert your file to JSON format instead."
                )
            with open(file_path) as f:
                gemm_config_lookup_table = yaml.safe_load(f)
        else:
            raise AssertionError(
                f"Unsupported file format. Only .json and .yaml/.yml files are supported. Got: {file_path}"
            )

    return gemm_config_lookup_table


# A static lookup table for triton configurations for GEMMs.
# This is a lookup table in the form of
# [op][device name][input_nodes_key][backend_key] = "(128, 128, 64, 2, 2)"
# where the value is a string representation of a string of a GemmConfig for
# `triton` backend key
gemm_config_lookup_table: Optional[dict[str, dict[str, dict[str, dict[str, str]]]]] = (
    None
)


def _lookup_key(input_nodes: list[Any]) -> str:
    return str(
        tuple(
            # List here, because we want a standard encoding e.g. [] instead of ()
            (node.get_dtype(), list(get_size_hint(node)), list(get_stride_hint(node)))
            for node in input_nodes
        )
    )


def get_lookup_table(input_nodes: list[Any], method: str) -> Optional[dict[str, str]]:
    lookup_dict = None
    lookup_table = _get_lookup_table()
    if _in_use() and lookup_table is not None:
        # Assume the first input parameter is used to determine the device
        device = input_nodes[0].get_device()
        device_name = (
            torch.cuda.get_device_name(device.index) if device.type == "cuda" else "cpu"
        )

        # Generate lookup table key
        lookup_key = _lookup_key(input_nodes)
        log.debug(f"lookup_key: {lookup_key}")
        # Retrieve the lookup dictionary
        lookup_dict = (
            lookup_table.get(method, {}).get(device_name, {}).get(lookup_key, {})
        )
    log.debug(f"lookup_dict for {method}: {lookup_dict}")
    return lookup_dict


def lookup_template_dict(
    lookup_dict: Optional[dict[str, str]], key: str
) -> Optional[str]:
    if not _in_use():
        return None
    # This function needs to return "" if the key is not found or
    # if the lookup_dict is None.
    # This is because we treat the case of the dict being empty
    # as the user not providing a lookup table for that input
    # and therefore falling back to using no Triton configs
    return lookup_dict.get(key, "") if lookup_dict is not None else ""


def lookup_table_extract_choice(choices: list[Any]) -> tuple[list[Any], bool]:
    """
    If there are multiple choices, this means we want the last choice as
    the lookup table produces at most one choice. The first choice is ATEN
    and the choice we fall back to when the lookup table has no match
    """
    if not _in_use():
        return choices, False
    if len(choices) > 1:
        return [choices[-1]], False
    # indicate to the caller that we're using the ATEN choice
    # as they might use this information to modify the layout
    return choices, True


def get_size_hint(mat: Any) -> list[int]:
    size = mat.get_size()
    if not all(isinstance(dim, int) for dim in size):
        size = V.graph.sizevars.size_hints(
            size,
            fallback=torch._inductor.config.unbacked_symint_fallback,
        )
    return size


def get_stride_hint(mat: Any) -> list[int]:
    stride = mat.get_stride()
    if not all(isinstance(dim, int) for dim in stride):
        stride = V.graph.sizevars.size_hints(
            stride,
            fallback=torch._inductor.config.unbacked_symint_fallback,
        )
    return stride
