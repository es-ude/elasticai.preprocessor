from unittest import TestCase, main

from .ir2c import get_embedded_datatype, replace_variables_with_parameters


class TestLoweringPassC(TestCase):
    def test_datatype_integer_signed_single(self):
        self.assertEqual(get_embedded_datatype(4, True), "int8_t")

    def test_datatype_integer_signed_array(self):
        bitwidth = [2, 4, 8, 10, 14, 16, 18, 30, 32]
        result = [get_embedded_datatype(bits, True) for bits in bitwidth]
        check = [
            "int8_t",
            "int8_t",
            "int8_t",
            "int16_t",
            "int16_t",
            "int16_t",
            "int32_t",
            "int32_t",
            "int32_t",
        ]
        self.assertEqual(result, check)

    def test_datatype_integer_unsigned_single(self):
        self.assertEqual(get_embedded_datatype(4, False), "uint8_t")

    def test_datatype_integer_unsigned_array(self):
        bitwidth = [2, 4, 8, 10, 14, 16, 18, 30, 32]
        result = [get_embedded_datatype(bits, False) for bits in bitwidth]
        check = [
            "uint8_t",
            "uint8_t",
            "uint8_t",
            "uint16_t",
            "uint16_t",
            "uint16_t",
            "uint32_t",
            "uint32_t",
            "uint32_t",
        ]
        self.assertEqual(result, check)

    def test_replacement_oneline_wo_content(self):
        params = dict(test="YES", bitwidth=14)
        test = ["This is an oneliner without param"]
        result = replace_variables_with_parameters(test, params)
        self.assertEqual(result, test)

    def test_replacement_oneline_content_single(self):
        params = dict(test="YES", bitwidth=14)
        test = ["This is an oneliner with param test={$test}"]
        result = replace_variables_with_parameters(test, params)
        chck = ["This is an oneliner with param test=YES"]
        self.assertEqual(result, chck)

    def test_replacement_oneline_content_double(self):
        params = dict(test="YES", bitwidth="14")
        test = ["This is an oneliner with param={$test} and bitwidth={$bitwidth}"]
        result = replace_variables_with_parameters(test, params)
        chck = ["This is an oneliner with param=YES and bitwidth=14"]
        self.assertEqual(result, chck)

    def test_replacement_twoline_wo_content(self):
        params = dict(test="YES", bitwidth="14")
        test = ["This is a twoliner", "without any params"]
        result = replace_variables_with_parameters(test, params)
        self.assertEqual(result, test)

    def test_replacement_twoline_content(self):
        params = dict(test="YES", bitwidth="14")
        test = [
            "This is a twoliner with param test={$test}",
            "and param bitwidth={$bitwidth}",
        ]
        result = replace_variables_with_parameters(test, params)
        check = ["This is a twoliner with param test=YES", "and param bitwidth=14"]
        self.assertEqual(result, check)


if __name__ == "__main__":
    main()
