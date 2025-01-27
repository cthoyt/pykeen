# -*- coding: utf-8 -*-

"""Utilities for PyKEEN."""

import ftplib
import functools
import itertools as itt
import json
import logging
import operator
import random
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import (
    Any, Callable, Collection, Dict, Generic, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.modules.batchnorm

from .constants import PYKEEN_BENCHMARKS
from .typing import DeviceHint, TorchRandomHint
from .version import get_git_hash

__all__ = [
    'compose',
    'clamp_norm',
    'compact_mapping',
    'ensure_torch_random_state',
    'format_relative_comparison',
    'imag_part',
    'invert_mapping',
    'is_cuda_oom_error',
    'random_non_negative_int',
    'real_part',
    'resolve_device',
    'split_complex',
    'split_list_in_batches_iter',
    'torch_is_in_1d',
    'normalize_string',
    'normalized_lookup',
    'get_cls',
    'get_until_first_blank',
    'flatten_dictionary',
    'set_random_seed',
    'NoRandomSeedNecessary',
    'Result',
    'fix_dataclass_init_docs',
    'get_benchmark',
    'extended_einsum',
]

logger = logging.getLogger(__name__)

#: An error that occurs because the input in CUDA is too big. See ConvE for an example.
_CUDNN_ERROR = 'cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.'

_CUDA_OOM_ERROR = 'CUDA out of memory.'


def resolve_device(device: DeviceHint = None) -> torch.device:
    """Resolve a torch.device given a desired device (string)."""
    if device is None or device == 'gpu':
        device = 'cuda'
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')
        logger.warning('No cuda devices were available. The model runs on CPU')
    return device


X = TypeVar('X')


def split_list_in_batches_iter(input_list: List[X], batch_size: int) -> Iterable[List[X]]:
    """Split a list of instances in batches of size batch_size."""
    return (
        input_list[i:i + batch_size]
        for i in range(0, len(input_list), batch_size)
    )


def normalize_string(s: str, *, suffix: Optional[str] = None) -> str:
    """Normalize a string for lookup."""
    s = s.lower().replace('-', '').replace('_', '').replace(' ', '')
    if suffix is not None and s.endswith(suffix.lower()):
        return s[:-len(suffix)]
    return s


def normalized_lookup(classes: Iterable[Type[X]]) -> Mapping[str, Type[X]]:
    """Make a normalized lookup dict."""
    return {
        normalize_string(cls.__name__): cls
        for cls in classes
    }


def get_cls(
    query: Union[None, str, Type[X]],
    base: Type[X],
    lookup_dict: Mapping[str, Type[X]],
    lookup_dict_synonyms: Optional[Mapping[str, Type[X]]] = None,
    default: Optional[Type[X]] = None,
    suffix: Optional[str] = None,
) -> Type[X]:
    """Get a class by string, default, or implementation."""
    if query is None:
        if default is None:
            raise ValueError(f'No default {base.__name__} set')
        return default
    elif not isinstance(query, (str, type)):
        raise TypeError(f'Invalid {base.__name__} type: {type(query)} - {query}')
    elif isinstance(query, str):
        key = normalize_string(query, suffix=suffix)
        if key in lookup_dict:
            return lookup_dict[key]
        if lookup_dict_synonyms is not None and key in lookup_dict_synonyms:
            return lookup_dict_synonyms[key]
        raise ValueError(f'Invalid {base.__name__} name: {query}')
    elif issubclass(query, base):
        return query
    raise TypeError(f'Not subclass of {base.__name__}: {query}')


def get_until_first_blank(s: str) -> str:
    """Recapitulate all lines in the string until the first blank line."""
    lines = list(s.splitlines())
    try:
        m, _ = min(enumerate(lines), key=lambda line: line == '')
    except ValueError:
        return s
    else:
        return ' '.join(
            line.lstrip()
            for line in lines[:m + 2]
        )


def flatten_dictionary(
    dictionary: Dict[str, Any],
    prefix: Optional[str] = None,
    sep: str = '.',
) -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    real_prefix = tuple() if prefix is None else (prefix,)
    partial_result = _flatten_dictionary(dictionary=dictionary, prefix=real_prefix)
    return {sep.join(map(str, k)): v for k, v in partial_result.items()}


def _flatten_dictionary(
    dictionary: Dict[str, Any],
    prefix: Tuple[str, ...],
) -> Dict[Tuple[str, ...], Any]:
    """Help flatten a nested dictionary."""
    result = {}
    for k, v in dictionary.items():
        new_prefix = prefix + (k,)
        if isinstance(v, dict):
            result.update(_flatten_dictionary(dictionary=v, prefix=new_prefix))
        else:
            result[new_prefix] = v
    return result


def clamp_norm(
    x: torch.Tensor,
    maxnorm: float,
    p: Union[str, int] = 'fro',
    dim: Union[None, int, Iterable[int]] = None,
    eps: float = 1.0e-08,
) -> torch.Tensor:
    """Ensure that a tensor's norm does not exceeds some threshold.

    :param x:
        The vector.
    :param maxnorm:
        The maximum norm (>0).
    :param p:
        The norm type.
    :param dim:
        The dimension(s).
    :param eps:
        A small value to avoid division by zero.

    :return:
        A vector with |x| <= max_norm.
    """
    norm = x.norm(p=p, dim=dim, keepdim=True)
    mask = (norm < maxnorm).type_as(x)
    return mask * x + (1 - mask) * (x / norm.clamp_min(eps) * maxnorm)


class compose(Generic[X]):  # noqa:N801
    """A class representing the composition of several functions."""

    def __init__(self, *operations: Callable[[X], X]):
        """Initialize the composition with a sequence of operations."""
        self.operations = operations

    def __call__(self, x: X) -> X:
        """Apply the operations in order to the given tensor."""
        for operation in self.operations:
            x = operation(x)
        return x


def set_random_seed(seed: int) -> Tuple[None, torch.Generator, None]:
    """Set the random seed on numpy, torch, and python."""
    np.random.seed(seed=seed)
    generator = torch.manual_seed(seed=seed)
    random.seed(seed)
    return None, generator, None


class NoRandomSeedNecessary:
    """Used in pipeline when random seed is set automatically."""


def all_in_bounds(
    x: torch.Tensor,
    low: Optional[float] = None,
    high: Optional[float] = None,
    a_tol: float = 0.,
) -> bool:
    """Check if tensor values respect lower and upper bound.

    :param x:
        The tensor.
    :param low:
        The lower bound.
    :param high:
        The upper bound.
    :param a_tol:
        Absolute tolerance.

    """
    # lower bound
    if low is not None and (x < low - a_tol).any():
        return False

    # upper bound
    if high is not None and (x > high + a_tol).any():
        return False

    return True


def is_cuda_oom_error(runtime_error: RuntimeError) -> bool:
    """Check whether the caught RuntimeError was due to CUDA being out of memory."""
    return _CUDA_OOM_ERROR in runtime_error.args[0]


def is_cudnn_error(runtime_error: RuntimeError) -> bool:
    """Check whether the caught RuntimeError was due to a CUDNN error."""
    return _CUDNN_ERROR in runtime_error.args[0]


def compact_mapping(
    mapping: Mapping[X, int],
) -> Tuple[Mapping[X, int], Mapping[int, int]]:
    """Update a mapping (key -> id) such that the IDs range from 0 to len(mappings) - 1.

    :param mapping:
        The mapping to compact.

    :return: A pair (translated, translation)
        where translated is the updated mapping, and translation a dictionary from old to new ids.
    """
    translation = {
        old_id: new_id
        for new_id, old_id in enumerate(sorted(mapping.values()))
    }
    translated = {
        k: translation[v]
        for k, v in mapping.items()
    }
    return translated, translation


class Result(ABC):
    """A superclass of results that can be saved to a directory."""

    @abstractmethod
    def save_to_directory(self, directory: str, **kwargs) -> None:
        """Save the results to the directory."""

    @abstractmethod
    def save_to_ftp(self, directory: str, ftp: ftplib.FTP) -> None:
        """Save the results to the directory in an FTP server."""

    @abstractmethod
    def save_to_s3(self, directory: str, bucket: str, s3=None) -> None:
        """Save all artifacts to the given directory in an S3 Bucket.

        :param directory: The directory in the S3 bucket
        :param bucket: The name of the S3 bucket
        :param s3: A client from :func:`boto3.client`, if already instantiated
        """


def split_complex(
    x: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Split a complex tensor into real and imaginary part."""
    dim = x.shape[-1] // 2
    return x[..., :dim], x[..., dim:]


def view_complex(x: torch.FloatTensor) -> torch.Tensor:
    """Convert a PyKEEN complex tensor representation into a torch one."""
    real, imag = split_complex(x=x)
    return torch.complex(real=real, imag=imag)


def combine_complex(
    x_re: torch.FloatTensor,
    x_im: torch.FloatTensor,
) -> torch.FloatTensor:
    """Combine a complex tensor from real and imaginary part."""
    return torch.cat([x_re, x_im], dim=-1)


def real_part(
    x: torch.FloatTensor,
) -> torch.FloatTensor:
    """Get the real part from a complex tensor."""
    dim = x.shape[-1] // 2
    return x[..., :dim]


def imag_part(
    x: torch.FloatTensor,
) -> torch.FloatTensor:
    """Get the imaginary part from a complex tensor."""
    dim = x.shape[-1] // 2
    return x[..., dim:]


def fix_dataclass_init_docs(cls: Type) -> Type:
    """Fix the ``__init__`` documentation for a :class:`dataclasses.dataclass`.

    :param cls: The class whose docstring needs fixing
    :returns: The class that was passed so this function can be used as a decorator

    .. seealso:: https://github.com/agronholm/sphinx-autodoc-typehints/issues/123
    """
    cls.__init__.__qualname__ = f'{cls.__name__}.__init__'
    return cls


def get_benchmark(name: str) -> Path:
    """Get the benchmark directory for this version."""
    rv = PYKEEN_BENCHMARKS / name / get_git_hash()
    rv.mkdir(exist_ok=True, parents=True)
    return rv


def get_model_io(model) -> BytesIO:
    """Get the model as bytes."""
    model_io = BytesIO()
    torch.save(model, model_io)
    model_io.seek(0)
    return model_io


def get_json_bytes_io(obj) -> BytesIO:
    """Get the JSON as bytes."""
    obj_str = json.dumps(obj, indent=2)
    obj_bytes = obj_str.encode('utf-8')
    return BytesIO(obj_bytes)


def get_df_io(df: pd.DataFrame) -> BytesIO:
    """Get the dataframe as bytes."""
    df_io = BytesIO()
    df.to_csv(df_io, sep='\t', index=False)
    df_io.seek(0)
    return df_io


def ensure_ftp_directory(*, ftp: ftplib.FTP, directory: str) -> None:
    """Ensure the directory exists on the FTP server."""
    try:
        ftp.mkd(directory)
    except ftplib.error_perm:
        pass  # its fine...


K = TypeVar("K")
V = TypeVar("V")


def invert_mapping(mapping: Mapping[K, V]) -> Mapping[V, K]:
    """
    Invert a mapping.

    :param mapping:
        The mapping, key -> value.

    :return:
        The inverse mapping, value -> key.
    """
    num_unique_values = len(set(mapping.values()))
    num_keys = len(mapping)
    if num_unique_values < num_keys:
        raise ValueError(f'Mapping is not bijective! Only {num_unique_values}/{num_keys} are unique.')
    return {
        value: key
        for key, value in mapping.items()
    }


def random_non_negative_int() -> int:
    """Generate a random positive integer."""
    sq = np.random.SeedSequence(np.random.randint(0, np.iinfo(np.int_).max))
    return int(sq.generate_state(1)[0])


def ensure_torch_random_state(random_state: TorchRandomHint) -> torch.Generator:
    """Prepare a random state for PyTorch."""
    if random_state is None:
        random_state = random_non_negative_int()
        logger.warning(f'using automatically assigned random_state={random_state}')
    if isinstance(random_state, int):
        random_state = torch.manual_seed(seed=random_state)
    if not isinstance(random_state, torch.Generator):
        raise TypeError
    return random_state


def torch_is_in_1d(
    query_tensor: torch.LongTensor,
    test_tensor: Union[Collection[int], torch.LongTensor],
    max_id: Optional[int] = None,
    invert: bool = False,
) -> torch.BoolTensor:
    """
    Return a boolean mask with Q[i] in T.

    The method guarantees memory complexity of max(size(Q), size(T)) and is thus, memory-wise, superior to naive
    broadcasting.

    :param query_tensor: shape: S
        The query Q.
    :param test_tensor:
        The test set T.
    :param max_id:
        A maximum ID. If not given, will be inferred.
    :param invert:
        Whether to invert the result.

    :return: shape: S
        A boolean mask.
    """
    # normalize input
    if not isinstance(test_tensor, torch.Tensor):
        test_tensor = torch.as_tensor(data=list(test_tensor), dtype=torch.long)
    if max_id is None:
        max_id = max(query_tensor.max(), test_tensor.max()) + 1
    mask = torch.zeros(max_id, dtype=torch.bool)
    mask[test_tensor] = True
    if invert:
        mask = ~mask
    return mask[query_tensor.view(-1)].view(*query_tensor.shape)


def format_relative_comparison(
    part: int,
    total: int,
) -> str:
    """Format a relative comparison."""
    return f"{part}/{total} ({part / total:2.2%})"


def broadcast_cat(
    x: torch.FloatTensor,
    y: torch.FloatTensor,
    dim: int,
) -> torch.FloatTensor:
    """Concatenate with broadcasting.

    :param x:
        The first tensor.
    :param y:
        The second tensor.
    :param dim:
        The concat dimension.

    :return:
    """
    if x.ndimension() != y.ndimension():
        raise ValueError
    if dim < 0:
        dim = x.ndimension() + dim
    x_rep, y_rep = [], []
    for d, (xd, yd) in enumerate(zip(x.shape, y.shape)):
        xr = yr = 1
        if d != dim and xd != yd:
            if xd == 1:
                xr = yd
            elif yd == 1:
                yr = xd
            else:
                raise ValueError
        x_rep.append(xr)
        y_rep.append(yr)
    return torch.cat([x.repeat(*x_rep), y.repeat(*y_rep)], dim=dim)


def get_batchnorm_modules(module: torch.nn.Module) -> List[torch.nn.Module]:
    """Return all submodules which are batch normalization layers."""
    return [
        submodule
        for submodule in module.modules()
        if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm)
    ]


def calculate_broadcasted_elementwise_result_shape(
    first: Tuple[int, ...],
    second: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Determine the return shape of a broadcasted elementwise operation."""
    return tuple(max(a, b) for a, b in zip(first, second))


def estimate_cost_of_sequence(
    shape: Tuple[int, ...],
    *other_shapes: Tuple[int, ...],
) -> int:
    """Cost of a sequence of broadcasted element-wise operations of tensors, given their shapes."""
    return sum(map(
        np.prod,
        itt.islice(
            itt.accumulate(
                (shape,) + other_shapes,
                calculate_broadcasted_elementwise_result_shape,
            ),
            1,
            None,
        ),
    ))


@functools.lru_cache(maxsize=32)
def _get_optimal_sequence(
    *sorted_shapes: Tuple[int, ...],
) -> Tuple[int, Tuple[int, ...]]:
    """Find the optimal sequence in which to combine tensors element-wise based on the shapes.

    The shapes should be sorted to enable efficient caching.
    :param sorted_shapes:
        The shapes of the tensors to combine.
    :return:
        The optimal execution order (as indices), and the cost.
    """
    return min(
        (estimate_cost_of_sequence(*(sorted_shapes[i] for i in p)), p)
        for p in itt.permutations(list(range(len(sorted_shapes))))
    )


@functools.lru_cache(maxsize=64)
def get_optimal_sequence(*shapes: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
    """Find the optimal sequence in which to combine tensors elementwise based on the shapes.

    :param shapes:
        The shapes of the tensors to combine.
    :return:
        The optimal execution order (as indices), and the cost.
    """
    # create sorted list of shapes to allow utilization of lru cache (optimal execution order does not depend on the
    # input sorting, as the order is determined by re-ordering the sequence anyway)
    arg_sort = sorted(range(len(shapes)), key=shapes.__getitem__)

    # Determine optimal order and cost
    cost, optimal_order = _get_optimal_sequence(*(shapes[new_index] for new_index in arg_sort))

    # translate back to original order
    optimal_order = tuple(arg_sort[i] for i in optimal_order)

    return cost, optimal_order


def _reorder(
    tensors: Tuple[torch.FloatTensor, ...],
) -> Tuple[torch.FloatTensor, ...]:
    """Re-order tensors for broadcasted element-wise combination of tensors.

    The optimal execution plan gets cached so that the optimization is only performed once for a fixed set of shapes.

    :param tensors:
        The tensors, in broadcastable shape.

    :return:
        The re-ordered tensors in optimal processing order.
    """
    if len(tensors) < 3:
        return tensors
    # determine optimal processing order
    shapes = tuple(tuple(t.shape) for t in tensors)
    if len(set(s[0] for s in shapes)) < 2:
        # heuristic
        return tensors
    order = get_optimal_sequence(*shapes)[1]
    return tuple(tensors[i] for i in order)


def tensor_sum(*x: torch.FloatTensor) -> torch.FloatTensor:
    """Compute elementwise sum of tensors in broadcastable shape."""
    return sum(_reorder(tensors=x))


def tensor_product(*x: torch.FloatTensor) -> torch.FloatTensor:
    """Compute elementwise product of tensors in broadcastable shape."""
    head, *rest = _reorder(tensors=x)
    return functools.reduce(operator.mul, rest, head)


def negative_norm_of_sum(
    *x: torch.FloatTensor,
    p: Union[str, int] = 2,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """Evaluate negative norm of a sum of vectors on already broadcasted representations.

    :param x: shape: (batch_size, num_heads, num_relations, num_tails, dim)
        The representations.
    :param p:
        The p for the norm. cf. torch.norm.
    :param power_norm:
        Whether to return $|x-y|_p^p$, cf. https://github.com/pytorch/pytorch/issues/28119

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    return negative_norm(tensor_sum(*x), p=p, power_norm=power_norm)


def negative_norm(
    x: torch.FloatTensor,
    p: Union[str, int] = 2,
    power_norm: bool = False,
) -> torch.FloatTensor:
    """Evaluate negative norm of a vector.

    :param x: shape: (batch_size, num_heads, num_relations, num_tails, dim)
        The vectors.
    :param p:
        The p for the norm. cf. torch.norm.
    :param power_norm:
        Whether to return $|x-y|_p^p$, cf. https://github.com/pytorch/pytorch/issues/28119

    :return: shape: (batch_size, num_heads, num_relations, num_tails)
        The scores.
    """
    if power_norm:
        assert not isinstance(p, str)
        return -(x.abs() ** p).sum(dim=-1)

    if torch.is_complex(x):
        assert not isinstance(p, str)
        # workaround for complex numbers: manually compute norm
        return -(x.abs() ** p).sum(dim=-1) ** (1 / p)

    return -x.norm(p=p, dim=-1)


def extended_einsum(
    eq: str,
    *tensors,
) -> torch.FloatTensor:
    """Drop dimensions of size 1 to allow broadcasting."""
    # TODO: check if einsum is still very slow.
    lhs, rhs = eq.split("->")
    mod_ops, mod_t = [], []
    for op, t in zip(lhs.split(","), tensors):
        mod_op = ""
        if len(op) != len(t.shape):
            raise ValueError(f'Shapes not equal: op={op} and t.shape={t.shape}')
        # TODO: t_shape = list(t.shape); del t_shape[i]; t.view(*shape) -> only one reshape operation
        for i, c in reversed(list(enumerate(op))):
            if t.shape[i] == 1:
                t = t.squeeze(dim=i)
            else:
                mod_op = c + mod_op
        mod_ops.append(mod_op)
        mod_t.append(t)
    m_lhs = ",".join(mod_ops)
    r_keep_dims = set("".join(mod_ops))
    m_rhs = "".join(c for c in rhs if c in r_keep_dims)
    m_eq = f"{m_lhs}->{m_rhs}"
    mod_r = torch.einsum(m_eq, *mod_t)
    # unsqueeze
    for i, c in enumerate(rhs):
        if c not in r_keep_dims:
            mod_r = mod_r.unsqueeze(dim=i)
    return mod_r


def project_entity(
    e: torch.FloatTensor,
    e_p: torch.FloatTensor,
    r_p: torch.FloatTensor,
) -> torch.FloatTensor:
    r"""Project entity relation-specific.

    .. math::

        e_{\bot} = M_{re} e
                 = (r_p e_p^T + I^{d_r \times d_e}) e
                 = r_p e_p^T e + I^{d_r \times d_e} e
                 = r_p (e_p^T e) + e'

    and additionally enforces

    .. math::

        \|e_{\bot}\|_2 \leq 1

    :param e: shape: (..., d_e)
        The entity embedding.
    :param e_p: shape: (..., d_e)
        The entity projection.
    :param r_p: shape: (..., d_r)
        The relation projection.

    :return: shape: (..., d_r)

    """
    # The dimensions affected by e'
    change_dim = min(e.shape[-1], r_p.shape[-1])

    # Project entities
    # r_p (e_p.T e) + e'
    e_bot = r_p * torch.sum(e_p * e, dim=-1, keepdim=True)
    e_bot[..., :change_dim] += e[..., :change_dim]

    # Enforce constraints
    e_bot = clamp_norm(e_bot, p=2, dim=-1, maxnorm=1)

    return e_bot


CANONICAL_DIMENSIONS = dict(h=1, r=2, t=3)


def _normalize_dim(dim: Union[int, str]) -> int:
    """Normalize the dimension selection."""
    if isinstance(dim, int):
        return dim
    return CANONICAL_DIMENSIONS[dim.lower()[0]]


def convert_to_canonical_shape(
    x: torch.FloatTensor,
    dim: Union[int, str],
    num: Optional[int] = None,
    batch_size: int = 1,
    suffix_shape: Union[int, Sequence[int]] = -1,
) -> torch.FloatTensor:
    """Convert a tensor to canonical shape.

    :param x:
        The tensor in compatible shape.
    :param dim:
        The "num" dimension.
    :param batch_size:
        The batch size.
    :param num:
        The number.
    :param suffix_shape:
        The suffix shape.

    :return: shape: (batch_size, num_heads, num_relations, num_tails, ``*``)
        A tensor in canonical shape.
    """
    if num is None:
        num = x.shape[0]
    suffix_shape = upgrade_to_sequence(suffix_shape)
    shape = [batch_size, 1, 1, 1]
    dim = _normalize_dim(dim=dim)
    shape[dim] = num
    return x.view(*shape, *suffix_shape)


def upgrade_to_sequence(x: Union[X, Sequence[X]]) -> Sequence[X]:
    """Ensure that the input is a sequence."""
    return x if isinstance(x, Sequence) else (x,)
