from os import makedirs
from os.path import join
from shutil import copyfile

import numpy as np


def get_embedded_datatype(bitwidth: int, signed: bool) -> str:
    """Determine the C datatype for processing data
    :param bitwidth:    Integer with bitwidth value
    :param signed:      Boolean if datatype is signed or unsigned
    :return:            String with datatype in C
    """
    assert bitwidth in range(33), "Allowed range of bitwidth should be >0 and <=32"
    bit_compare = np.array((8, 16, 32, 64))
    used_bitval = np.argwhere((bit_compare / bitwidth) - 1 >= 0).flatten()[0]
    return ("" if signed else "u") + "int" + f"{bit_compare[used_bitval]}" + "_t"


def replace_variables_with_parameters(string_input: list, parameters: dict) -> list:
    """Function for search parameter in string list and replace with new defined real values
    :param string_input:   List with input strings from file
    :param parameters:     Dictionary with parameters (key and value)
    :returns:              List with corrected string output
    """
    string_output = list()
    for line in string_input:
        if "{$" not in line:
            used_line = line
        else:
            overview_split = line.split("{$")
            used_line = line
            for split_param in overview_split[1:]:
                param_search = split_param.split("}")[0]
                for key, value in parameters.items():
                    if param_search == key:
                        used_line = used_line.replace(f"{{${param_search}}}", value)
                        break
        string_output.append(used_line)
    return string_output


def generate_c_files(
    path2save: str,
    template_name: str,
    file_name: str,
    module_id: str,
    proto_file: list,
    impl_file: list,
    path2template: str = "./../c",
) -> None:
    """Function for generating the C files
    :param path2save:       Path to save C files
    :param template_name:   String with template name
    :param file_name:       String with file name
    :param module_id:       String with module ID
    :param proto_file:      List with content of the prototype implementation
    :param impl_file:       List with content of the implementation
    :param path2template:   Path to template file
    :return:                None
    """
    print(path2save)
    makedirs(name=path2save, exist_ok=True)
    copyfile(src=join(path2template, template_name), dst=join(path2save, f"{template_name}"))

    with open(join(path2save, f"{file_name}_{module_id.lower()}.h"), "w") as hndl:
        for line in proto_file:
            hndl.write(line + "\n")
    hndl.close()
    del hndl

    with open(join(path2save, f"{file_name}_{module_id.lower()}.c"), "w") as hndl:
        for line in impl_file:
            hndl.write(line + "\n")
    hndl.close()
