from dataclasses import dataclass


def check_key_elements(key: str, elements: list[str]) -> bool:
    """Function for checking if all elements are in key (logical AND)
    :param key:         Key to check
    :param elements:    List of elements to check if available in key
    :return:            True if all elements are present in key
    """
    return any(elem == key for elem in elements)


def check_string_equal_elements_all(text: str, elements: list[str]) -> bool:
    """Function for checking if all elements are in text string (logical AND)
    :param text :       String with a text
    :param elements:    List of elements to check if available in text
    :return:            True if all elements are present in text
    """
    return all(elem in text for elem in elements)


def check_string_equal_elements_any(text: str, elements: list[str]) -> bool:
    """Function for checking if elements are in text string (logical OR)
    :param text:        String with a text
    :param elements:    List of elements to check if available in text
    :return:            True if any elements are present in text
    """
    val = any(elem in text for elem in elements)
    return val


def check_keylist_elements_all(keylist: list[str], elements: list[str]) -> bool:
    """Function for checking if all elements are in key list (logical AND)
    :param keylist:     List with keys to check
    :param elements:    List with elements to check if available in key
    :return:            True if all elements are present in key
    """
    return all(elem in keylist for elem in elements) if len(keylist) else True


def check_keylist_elements_any(keylist: list[str], elements: list[str]) -> bool:
    """Function for checking if all elements are in key list (logical OR)
    :param keylist:     List with keys to check
    :param elements:    List with elements to check if available in key
    :return:            True if any elements are present in key
    """
    return any(elem in keylist for elem in elements) if len(keylist) else True


def check_elem_unique(elements: list) -> bool:
    """Function for checking if all elements are unique
    :param elements:    List of elements to check
    :return:            True if all elements are unique
    """
    from collections import Counter
    from itertools import chain

    chck = elements if not type(elements[0]) == list else list(chain.from_iterable(elements))
    return all(cnt == 1 for cnt in Counter(chck).values())


def check_value_range(value: float | int, range: list[float | int]) -> bool:
    """Function for checking if value is within range
    :param value:     Value to check (float or integer)
    :param range:     List with two values to indicate the range
    :return:          Boolean if value is in range
    """
    assert len(range) == 2, "Array should have 2 elements [min, max]"
    return range[0] <= value <= range[1]


def is_close(value: float, target: float, tolerance: float = 0.05) -> bool:
    """Function for checking if float value is in near of the target value
    :param value:       Float value to check
    :param target:      Target value
    :param tolerance:   Tolerance value [around target value]
    """
    assert tolerance > 0
    return abs(value - target) <= abs(tolerance)


@dataclass
class MetricTimestamps:
    """Class with metrics for comparing timestamps of predicted classes and true classes
    Attributes:
        f1_score:    Float with F1-Score
        TP:          Integer with true positives
        FP:          Integer with false positives
        FN:          Integer with false negatives
    """

    f1_score: float
    TP: int
    FP: int
    FN: int


def compare_timestamps(true_labels: list, pred_labels: list, window: int = 2) -> MetricTimestamps:
    """This function compares the timestamps of the predicted classes and the true classes and returns TP, FP, FN and
    new arrays which only contain the classes that have matched timestamps in both arrays. The function should be used
    before plotting a confusion matrix of the classes when working with actual data from the pipeline.
    Args:
        true_labels:    List with true labels
        pred_labels:    List with predicted labels
        window:         Window size for acceptance rate
    Returns:
        Class MetricTimeStamps with metrics
    """
    new_pred = []
    false_negative = 0
    true_positive_same = 0
    true_positive_diff = 0

    for i in range(0, max(true_labels[-1], pred_labels[-1]) + 1):
        if i in true_labels:
            found = False
            for j in range(i - int(window), i + int(window) + 1):
                if j in pred_labels:
                    pos_true = true_labels.index(i)
                    pos_pred = pred_labels.index(j)
                    new_pred.append(pred_labels[pos_pred])
                    if true_labels[pos_true] == pred_labels[pos_pred]:
                        true_positive_same += 1
                    else:
                        true_positive_diff += 1
                    found = True
            if not found:
                false_negative += 1

    if len(pred_labels) - len(true_labels) > 0:
        false_positive = len(pred_labels) - len(true_labels)
    else:
        false_positive = 0
    true_positive = true_positive_same + true_positive_diff

    f1_score = true_positive / (true_positive + false_positive + false_negative)
    return MetricTimestamps(f1_score=f1_score, FN=false_negative, FP=false_positive, TP=true_positive)
