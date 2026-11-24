import unittest

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from .augmentation import augmentation_downsampling


def build_random_dataset_labeled(
    n_ch: int = 2, n_label: int = 4, n_samples: int = 1000, n_window=10000
) -> tuple[np.ndarray, np.ndarray]:
    time0 = np.linspace(start=0, stop=2 * np.pi, num=n_window, endpoint=True, dtype=np.float32)
    if n_ch > 1:
        data = np.zeros((n_samples, n_ch, n_window))
    else:
        data = np.zeros((n_samples, n_window))

    data[..., :] = np.sin(time0) + 0.01 * np.random.randn(*data.shape)
    label = np.random.randint(low=0, high=n_label, size=(n_samples,))
    return data, label


def build_random_dataset_unlabeled(
    n_ch: int = 2, n_label: int = 4, n_samples: int = 1000, n_window=10000
) -> tuple[np.ndarray, np.ndarray]:
    data, label = build_random_dataset_labeled(n_ch, n_label, n_samples, n_window)
    return data, np.zeros_like(label)


class BuildDatasetDownsampling(unittest.TestCase):
    def test_build_dataset_labeled_3d(self):
        data, label = build_random_dataset_labeled(n_ch=2, n_label=4, n_samples=1000, n_window=100)
        self.assertEqual(label.size, 1000)
        assert label.min() == 0
        assert label.max() == 3
        assert len(data.shape) == 3

    def test_build_dataset_labeled_2d(self):
        data, label = build_random_dataset_labeled(n_ch=1, n_label=4, n_samples=1000, n_window=100)
        self.assertEqual(label.size, 1000)
        assert label.min() == 0
        assert label.max() == 3
        assert len(data.shape) == 2

    def test_build_dataset_unlabeled_3d(self):
        data, label = build_random_dataset_unlabeled(n_ch=2, n_label=4, n_samples=1000, n_window=100)
        self.assertEqual(label.size, 1000)
        assert label.min() == 0
        assert label.max() == 0
        assert len(data.shape) == 3

    def test_build_dataset_unlabeled_2d(self):
        data, label = build_random_dataset_unlabeled(n_ch=2, n_label=4, n_samples=1000, n_window=100)
        self.assertEqual(label.size, 1000)
        assert label.min() == 0
        assert label.max() == 0
        assert len(data.shape) == 3


def random_dataset_2d(n_samples: int, n_window: int, n_label: int):
    """Random single-channel dataset (n_samples, n_window) with matching labels"""
    return st.tuples(
        arrays(
            dtype=np.float64,
            shape=(n_samples, n_window),
            elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        ),
        arrays(dtype=np.int64, shape=n_samples, elements=st.integers(min_value=0, max_value=n_label - 1)),
    )


class AugmentationDownsamplingLabeled2D(unittest.TestCase):
    samples = 5
    windowsize = 200
    dataset = random_dataset_2d(n_samples=samples, n_window=windowsize, n_label=2)

    @given(dataset=dataset)
    def test_augmentation_level0(self, dataset):
        data_orig, label_orig = dataset
        num_level = 1
        data_new, label_new = augmentation_downsampling(
            data=data_orig,
            label=label_orig,
            n_downsampling=num_level,
            drop_samples=True,
        )
        np.testing.assert_array_equal(data_new, data_orig)
        np.testing.assert_array_equal(label_new, label_orig)

    @given(dataset=dataset)
    def test_augmentation_level1_drop(self, dataset):
        data_orig, label_orig = dataset
        num_level = 2
        data_new, label_new = augmentation_downsampling(
            data=data_orig,
            label=label_orig,
            n_downsampling=num_level,
            drop_samples=True,
        )
        self.assertEqual(data_new.shape[0], data_orig.shape[0])
        self.assertEqual(label_new.size, round(label_orig.size))
        self.assertEqual(data_new.shape[-1], round(data_orig.shape[-1] / num_level))

    @given(dataset=dataset)
    def test_augmentation_level1_add(self, dataset):
        data_orig, label_orig = dataset
        num_level = 2
        data_new, label_new = augmentation_downsampling(
            data=data_orig,
            label=label_orig,
            n_downsampling=num_level,
            drop_samples=False,
        )
        self.assertEqual(data_new.shape[0], round(data_orig.shape[0] * num_level))
        self.assertEqual(label_new.size, round(label_orig.size * num_level))
        self.assertEqual(data_new.shape[-1], round(data_orig.shape[-1] / num_level))
        self.assertEqual(data_new[0, 0], data_orig[0, 0])
        self.assertEqual(label_new[0], label_orig[0])
        for idx in range(self.samples):
            self.assertEqual(label_new[self.samples + idx], label_orig[idx])
        for idx in range(round(self.windowsize / num_level)):
            self.assertEqual(data_new[self.samples, idx], data_orig[0, num_level * idx + 1])

    @given(dataset=dataset)
    def test_augmentation_level2_drop(self, dataset):
        data_orig, label_orig = dataset
        num_level = 3
        data_new, label_new = augmentation_downsampling(
            data=data_orig,
            label=label_orig,
            n_downsampling=num_level,
            drop_samples=True,
        )
        self.assertEqual(data_new.shape[0], data_orig.shape[0])
        self.assertEqual(label_new.size, round(label_orig.size))
        self.assertEqual(data_new.shape[-1], round(data_orig.shape[-1] / num_level))

    @given(dataset=dataset)
    def test_augmentation_level2_add(self, dataset):
        data_orig, label_orig = dataset
        num_level = 3
        data_new, label_new = augmentation_downsampling(
            data=data_orig,
            label=label_orig,
            n_downsampling=num_level,
            drop_samples=False,
        )
        self.assertEqual(data_new.shape[0], round(data_orig.shape[0] * num_level))
        self.assertEqual(label_new.size, round(label_orig.size * num_level))
        self.assertEqual(data_new.shape[-1], round(data_orig.shape[-1] / num_level))
        self.assertEqual(data_new[0, 0], data_orig[0, 0])
        self.assertEqual(label_new[0], label_orig[0])
        for idx in range(self.samples):
            self.assertEqual(label_new[self.samples + idx], label_orig[idx])
        for idx in range(round(self.windowsize / num_level)):
            self.assertEqual(data_new[self.samples, idx], data_orig[0, num_level * idx + 1])

    @given(dataset=dataset)
    def test_augmentation_level9_drop(self, dataset):
        data_orig, label_orig = dataset
        num_level = 10
        data_new, label_new = augmentation_downsampling(
            data=data_orig,
            label=label_orig,
            n_downsampling=num_level,
            drop_samples=True,
        )
        self.assertEqual(data_new.shape[0], data_orig.shape[0])
        self.assertEqual(label_new.size, round(label_orig.size))
        self.assertEqual(data_new.shape[-1], round(data_orig.shape[-1] / num_level))

    @given(dataset=dataset)
    def test_augmentation_level9_add(self, dataset):
        data_orig, label_orig = dataset
        num_level = 10
        data_new, label_new = augmentation_downsampling(
            data=data_orig,
            label=label_orig,
            n_downsampling=num_level,
            drop_samples=False,
        )
        self.assertEqual(data_new.shape[0], round(data_orig.shape[0] * num_level))
        self.assertEqual(label_new.size, round(label_orig.size * num_level))
        self.assertEqual(data_new.shape[-1], round(data_orig.shape[-1] / num_level))
        self.assertEqual(data_new[0, 0], data_orig[0, 0])
        self.assertEqual(label_new[0], label_orig[0])
        for idx in range(self.samples):
            self.assertEqual(label_new[self.samples + idx], label_orig[idx])
        for idx in range(round(self.windowsize / num_level)):
            self.assertEqual(data_new[self.samples, idx], data_orig[0, num_level * idx + 1])


if __name__ == "__main__":
    unittest.main()
