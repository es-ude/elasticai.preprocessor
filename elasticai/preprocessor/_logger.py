import logging

from elasticai.preprocessor import get_path_to_project


def define_logger_testing(save_file: bool = True):
    """Function for preparing the logger configuration in testing routines
    :param save_file:   Boolean for saving the outputs into file (then no terminal output is generated)
    """
    return logging.basicConfig(
        level=logging.DEBUG,
        filename=get_path_to_project() / "run_test_report.log" if save_file else None,
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s = %(message)s",
    )


def define_logger_runtime(save_file: bool = True):
    """Function for preparing the logger configuration in runtime routines
    :param save_file:   Boolean for saving the outputs into file (then no terminal output is generated)
    """
    return logging.basicConfig(
        level=logging.INFO,
        filename=get_path_to_project("runs") / "runtime_report_normal.log" if save_file else None,
        filemode="w",
        format="%(asctime)s: %(message)s",
    )


def define_logger_runtime_debug(save_file: bool = True):
    """Function for preparing the logger configuration in runtime debugging routines
    :param save_file:   Boolean for saving the outputs into file (then no terminal output is generated)
    """
    return logging.basicConfig(
        level=logging.DEBUG,
        filename=get_path_to_project("runs") / "runtime_report_debug.log" if save_file else None,
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s = %(message)s",
    )
