import unittest

from elasticai.preprocessor import get_path_to_project


class TestHelpFunction(unittest.TestCase):
    def test_get_path_to_project_wo_ref(self):
        ref = ["elastic-ai", "preprocessor"]
        chck = get_path_to_project().as_posix()
        rslt = [True for key in ref if key in chck]
        self.assertTrue(sum(rslt) == 2)

    def test_get_path_to_project_with_ref(self):
        chck = get_path_to_project(new_folder="test")
        rslt = chck == get_path_to_project() / "test"
        self.assertTrue(rslt)


if __name__ == "__main__":
    unittest.main()
