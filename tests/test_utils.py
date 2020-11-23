# -*- coding: utf-8 -*-

"""Unittest for for global utilities."""
import itertools
import random
import string
import timeit
import unittest
from typing import Iterable, Tuple

import numpy
import torch

from pykeen.nn import Embedding
from pykeen.utils import (
    calculate_broadcasted_elementwise_result_shape, clamp_norm,
    compact_mapping,
    estimate_cost_of_sequence, flatten_dictionary,
    get_optimal_sequence, get_until_first_blank,
    l2_regularization, tensor_sum,
)


class L2RegularizationTest(unittest.TestCase):
    """Test L2 regularization."""

    def test_one_tensor(self):
        """Test if output is correct for a single tensor."""
        t = torch.ones(1, 2, 3, 4)
        reg = l2_regularization(t)
        self.assertAlmostEqual(float(reg), float(numpy.prod(t.shape)))

    def test_many_tensors(self):
        """Test if output is correct for var-args."""
        ts = []
        exp_reg = 0.
        for i, shape in enumerate([
            (1, 2, 3),
            (2, 3, 4),
            (3, 4, 5),
        ]):
            t = torch.ones(*shape) * (i + 1)
            ts.append(t)
            exp_reg += numpy.prod(t.shape) * (i + 1) ** 2
        reg = l2_regularization(*ts)
        self.assertAlmostEqual(float(reg), exp_reg)


class FlattenDictionaryTest(unittest.TestCase):
    """Test flatten_dictionary."""

    def test_flatten_dictionary(self):
        """Test if the output of flatten_dictionary is correct."""
        nested_dictionary = {
            'a': {
                'b': {
                    'c': 1,
                    'd': 2,
                },
                'e': 3,
            },
        }
        expected_output = {
            'a.b.c': 1,
            'a.b.d': 2,
            'a.e': 3,
        }
        observed_output = flatten_dictionary(nested_dictionary)
        self._compare(observed_output, expected_output)

    def test_flatten_dictionary_mixed_key_type(self):
        """Test if the output of flatten_dictionary is correct if some keys are not strings."""
        nested_dictionary = {
            'a': {
                5: {
                    'c': 1,
                    'd': 2,
                },
                'e': 3,
            },
        }
        expected_output = {
            'a.5.c': 1,
            'a.5.d': 2,
            'a.e': 3,
        }
        observed_output = flatten_dictionary(nested_dictionary)
        self._compare(observed_output, expected_output)

    def test_flatten_dictionary_prefix(self):
        """Test if the output of flatten_dictionary is correct."""
        nested_dictionary = {
            'a': {
                'b': {
                    'c': 1,
                    'd': 2,
                },
                'e': 3,
            },
        }
        expected_output = {
            'Test.a.b.c': 1,
            'Test.a.b.d': 2,
            'Test.a.e': 3,
        }
        observed_output = flatten_dictionary(nested_dictionary, prefix='Test')
        self._compare(observed_output, expected_output)

    def _compare(self, observed_output, expected_output):
        assert not any(isinstance(o, dict) for o in expected_output.values())
        assert expected_output == observed_output


class TestGetUntilFirstBlank(unittest.TestCase):
    """Test get_until_first_blank()."""

    def test_get_until_first_blank_trivial(self):
        """Test the trivial string."""
        s = ''
        r = get_until_first_blank(s)
        self.assertEqual('', r)

    def test_regular(self):
        """Test a regulat case."""
        s = """Broken
        line.

        Now I continue.
        """
        r = get_until_first_blank(s)
        self.assertEqual("Broken line.", r)


class EmbeddingsInCanonicalShapeTests(unittest.TestCase):
    """Test get_embedding_in_canonical_shape()."""

    #: The number of embeddings
    num_embeddings: int = 3

    #: The embedding dimension
    embedding_dim: int = 2

    def setUp(self) -> None:
        """Initialize embedding."""
        self.embedding = Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
        self.generator = torch.manual_seed(42)
        self.embedding._embeddings.weight.data = torch.rand(
            self.num_embeddings,
            self.embedding_dim,
            generator=self.generator,
        )

    def test_no_indices(self):
        """Test getting all embeddings."""
        emb = self.embedding.get_in_canonical_shape(indices=None)

        # check shape
        assert emb.shape == (1, self.num_embeddings, self.embedding_dim)

        # check values
        exp = self.embedding(indices=None).view(1, self.num_embeddings, self.embedding_dim)
        assert torch.allclose(emb, exp)

    def _test_with_indices(self, indices: torch.Tensor) -> None:
        """Help tests with index."""
        emb = self.embedding.get_in_canonical_shape(indices=indices)

        # check shape
        num_ind = indices.shape[0]
        assert emb.shape == (num_ind, 1, self.embedding_dim)

        # check values
        exp = torch.stack([self.embedding(i) for i in indices], dim=0).view(num_ind, 1, self.embedding_dim)
        assert torch.allclose(emb, exp)

    def test_with_consecutive_indices(self):
        """Test to retrieve all embeddings with consecutive indices."""
        indices = torch.arange(self.num_embeddings, dtype=torch.long)
        self._test_with_indices(indices=indices)

    def test_with_indices_with_duplicates(self):
        """Test to retrieve embeddings at random positions with duplicate indices."""
        indices = torch.randint(
            self.num_embeddings,
            size=(2 * self.num_embeddings,),
            dtype=torch.long,
            generator=self.generator,
        )
        self._test_with_indices(indices=indices)

    def test_compact_mapping(self):
        """Test ``compact_mapping()``."""
        mapping = {
            letter: 2 * i
            for i, letter in enumerate(string.ascii_letters)
        }
        compacted_mapping, id_remapping = compact_mapping(mapping=mapping)

        # check correct value range
        self.assertEqual(set(compacted_mapping.values()), set(range(len(mapping))))
        self.assertEqual(set(id_remapping.keys()), set(mapping.values()))
        self.assertEqual(set(id_remapping.values()), set(compacted_mapping.values()))


def test_clamp_norm():
    """Test  clamp_norm() ."""
    max_norm = 1.0
    gen = torch.manual_seed(42)
    eps = 1.0e-06
    for p in [1, 2, float('inf')]:
        for _ in range(10):
            x = torch.rand(10, 20, 30, generator=gen)
            for dim in range(x.ndimension()):
                x_c = clamp_norm(x, maxnorm=max_norm, p=p, dim=dim)

                # check maximum norm constraint
                assert (x_c.norm(p=p, dim=dim) <= max_norm + eps).all()

                # unchanged values for small norms
                norm = x.norm(p=p, dim=dim)
                mask = torch.stack([(norm < max_norm)] * x.shape[dim], dim=dim)
                assert (x_c[mask] == x[mask]).all()


def test_calculate_broadcasted_elementwise_result_shape():
    """Test calculate_broadcasted_elementwise_result_shape."""
    max_dim = 64
    for n_dim, i in itertools.product(range(2, 5), range(10)):
        a_shape = [1 for _ in range(n_dim)]
        b_shape = [1 for _ in range(n_dim)]
        for j in range(n_dim):
            dim = 2 + random.randrange(max_dim)
            mod = random.randrange(3)
            if mod % 2 == 0:
                a_shape[j] = dim
            if mod > 0:
                b_shape[j] = dim
            a = numpy.empty(shape=a_shape)
            b = numpy.empty(shape=b_shape)
            shape = calculate_broadcasted_elementwise_result_shape(first=a.shape, second=b.shape)
            c = a + b
            exp_shape = c.shape
            assert shape == exp_shape


def _generate_shapes(
    n_dim: int = 5,
    n_terms: int = 4,
    iterations: int = 64,
) -> Iterable[Tuple[Tuple[int, ...], ...]]:
    """Generate shapes."""
    max_shape = numpy.random.randint(low=2, high=32, size=(128,))
    for _i in range(iterations):
        # create broadcastable shapes
        numpy.random.shuffle(max_shape)
        this_max_shape = max_shape[:n_dim]
        this_min_shape = numpy.ones_like(this_max_shape)
        shapes = []
        for _j in range(n_terms):
            mask = this_min_shape
            while not (1 < mask.sum() < n_dim):
                mask = numpy.asarray(numpy.random.uniform(size=(n_dim,)) < 0.3, dtype=numpy.int64)
            this_array_shape = this_max_shape * mask + this_min_shape * (1 - mask)
            shapes.append(tuple(this_array_shape.tolist()))
        yield tuple(shapes)


def test_estimate_cost_of_add_sequence():
    """Test ``estimate_cost_of_add_sequence()``."""
    # create random array, estimate the costs of addition, and measure some execution times.
    # then, compute correlation between the estimated cost, and the measured time.
    data = []
    for shapes in _generate_shapes():
        arrays = [numpy.empty(shape=shape) for shape in shapes]
        cost = estimate_cost_of_sequence(*(a.shape for a in arrays))
        consumption = timeit.timeit(stmt='sum(arrays)', globals=locals(), number=25)
        data.append((cost, consumption))
    a = numpy.asarray(data)

    # check for strong correlation between estimated costs and measured execution time
    assert (numpy.corrcoef(x=a[:, 0], y=a[:, 1])[0, 1]) > 0.8


def test_get_optimal_add_sequence():
    """Test ``get_optimal_add_sequence()``."""
    for shapes in _generate_shapes():
        # get optimal sequence
        opt_cost, opt_seq = get_optimal_sequence(*shapes)

        # check correct cost
        exp_opt_cost = estimate_cost_of_sequence(*(shapes[i] for i in opt_seq))
        assert exp_opt_cost == opt_cost

        # check optimality
        for perm in itertools.permutations(list(range(len(shapes)))):
            cost = estimate_cost_of_sequence(*(shapes[i] for i in perm))
            assert cost >= opt_cost


def test_tensor_sum():
    """Test tensor_sum."""
    for shapes in _generate_shapes():
        tensors = [torch.rand(*shape) for shape in shapes]
        result = tensor_sum(*tensors)

        # compare result to sequential addition
        assert torch.allclose(result, sum(tensors))
