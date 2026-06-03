import unittest

from ._check_funcs import (
    MetricTimestamps,
    check_elem_unique,
    check_key_elements,
    check_keylist_elements_all,
    check_keylist_elements_any,
    check_string_equal_elements_all,
    check_string_equal_elements_any,
    check_value_range,
    compare_timestamps,
    is_close,
)


class TestHelpFunction(unittest.TestCase):
    def test_is_close_true(self):
        rslt = is_close(value=95, target=100, tolerance=5)
        self.assertTrue(rslt)

    def test_is_close_false(self):
        rslt = is_close(value=94, target=100, tolerance=5)
        self.assertFalse(rslt)

    def test_check_string_elements_all_true(self):
        elements = ["number", "true"]
        rslt = check_string_equal_elements_all(text="is_the_number_true", elements=elements)
        self.assertTrue(rslt)

    def test_check_string_elements_all_false(self):
        elements = ["number", "false"]
        rslt = check_string_equal_elements_all(text="is_the_number_true", elements=elements)
        self.assertFalse(rslt)

    def test_check_string_elements_any_true(self):
        elements = ["number", "false"]
        rslt = check_string_equal_elements_any(text="is_the_number_true", elements=elements)
        self.assertTrue(rslt)

    def test_check_string_elements_any_false(self):
        elements = ["zero", "false"]
        rslt = check_string_equal_elements_any(text="is_the_number_true", elements=elements)
        self.assertFalse(rslt)

    def test_check_key_elements_true(self):
        elements = ["num", "ber", "true"]
        rslt = check_key_elements(key="true", elements=elements)
        self.assertTrue(rslt)

    def test_check_key_elements_false(self):
        elements = ["num", "ber", "false"]
        rslt = check_key_elements(key="true", elements=elements)
        self.assertFalse(rslt)

    def test_check_keylist_elements_all_empty(self):
        elements = ["beat", "iful", "true"]
        rslt = check_keylist_elements_all(keylist=[], elements=elements)
        self.assertTrue(rslt)

    def test_check_keylist_elements_all_true(self):
        elements = ["beat", "iful", "true"]
        rslt = check_keylist_elements_all(keylist=["beat", "iful", "true", "maybe"], elements=elements)
        self.assertTrue(rslt)

    def test_check_keylist_elements_all_false(self):
        elements = ["beat", "iful", "today"]
        rslt = check_keylist_elements_all(keylist=["beatiful", "tomorrow"], elements=elements)
        self.assertFalse(rslt)

    def test_check_keylist_elements_any_empty(self):
        elements = ["beat", "iful", "true"]
        rslt = check_keylist_elements_any(keylist=[], elements=elements)
        self.assertTrue(rslt)

    def test_check_keylist_elements_any_true(self):
        elements = ["num", "ber", "true"]
        rslt = check_keylist_elements_any(keylist=["number", "true", "berry"], elements=elements)
        self.assertTrue(rslt)

    def test_check_keylist_elements_any_false(self):
        elements = ["num", "ber", "true", "cherry"]
        rslt = check_keylist_elements_any(keylist=["number", "false", "berry"], elements=elements)
        self.assertFalse(rslt)

    def test_check_elem_unique_string_true(self):
        elements = ["num", "ber", "true"]
        rslt = check_elem_unique(elements)
        self.assertTrue(rslt)

    def test_check_elem_unique_string_false(self):
        elements = ["num", "ber", "ber"]
        rslt = check_elem_unique(elements)
        self.assertFalse(rslt)

    def test_check_elem_unique_number_true(self):
        elements = [0, 1, 2]
        rslt = check_elem_unique(elements)
        self.assertTrue(rslt)

    def test_check_elem_unique_number_false(self):
        elements = [0, 0, 2]
        rslt = check_elem_unique(elements)
        self.assertFalse(rslt)

    def test_check_elem_unique_list_true(self):
        elements = [[0, 1, 2], [3, 4, 5]]
        rslt = check_elem_unique(elements)
        self.assertTrue(rslt)

    def test_check_elem_unique_list_false(self):
        elements = [[0, 1, 2], [3, 1, 5]]
        rslt = check_elem_unique(elements)
        self.assertFalse(rslt)

    def test_check_value_range_true(self):
        rslt = check_value_range(value=1.0, range=[0.9, 1.1])
        self.assertTrue(rslt)

    def test_check_value_range_false(self):
        rslt = check_value_range(value=0.8, range=[0.9, 1.1])
        self.assertFalse(rslt)


class CompareTimestampsTest(unittest.TestCase):
    def test_simple_list_same(self):
        pos_true = [4, 8, 20, 42, 80, 102]
        pos_pred = [4, 8, 20, 42, 80, 102]
        metrics: MetricTimestamps = compare_timestamps(
            true_labels=pos_true, pred_labels=pos_pred, window=2
        )
        self.assertEqual(metrics.f1_score, 1.0)

    def test_simple_list_same_inside_window(self):
        pos_true = [4, 8, 20, 42, 80, 102]
        pos_pred = [2, 9, 20, 41, 81, 103]
        metrics: MetricTimestamps = compare_timestamps(
            true_labels=pos_true, pred_labels=pos_pred, window=2
        )
        self.assertEqual(metrics.f1_score, 1.0)

    def test_simple_list_almost_same(self):
        pos_true = [4, 8, 20, 42, 80, 102]
        pos_pred = [2, 9, 20, 45, 77, 105]
        metrics: MetricTimestamps = compare_timestamps(
            true_labels=pos_true, pred_labels=pos_pred, window=2
        )
        self.assertEqual(metrics.f1_score, 0.5)

    def test_simple_list_different_size0(self):
        pos_true = [4, 8, 20, 42, 80, 102, 115, 134]
        pos_pred = [4, 8, 20, 42, 80, 102]
        metrics: MetricTimestamps = compare_timestamps(
            true_labels=pos_true, pred_labels=pos_pred, window=2
        )
        self.assertEqual(metrics.f1_score, 3 / 4)

    def test_simple_list_different_size1(self):
        pos_true = [4, 8, 20, 42, 80, 102]
        pos_pred = [4, 8, 20, 42, 80, 102, 115, 134]
        metrics: MetricTimestamps = compare_timestamps(
            true_labels=pos_true, pred_labels=pos_pred, window=2
        )
        self.assertEqual(metrics.f1_score, 3 / 4)


if __name__ == "__main__":
    unittest.main()
