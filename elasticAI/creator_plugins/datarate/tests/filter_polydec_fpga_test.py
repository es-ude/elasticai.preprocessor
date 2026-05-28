import pytest
from copy import deepcopy

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
from elasticai.creator_plugins.datarate.tests.filter_polydec_fpga_tb import cocotb_settings


#@pytest.mark.simulation
@pytest.mark.parametrize(
    ["bitwidth", "poly_order"], [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (4, 0),
        (4, 2),
        (8, 0),
        (8, 3),
        (12, 1),
        (12, 2),
        (16, 0),
        (16, 3)
    ]
)
def test_polydec_fpga(bitwidth: int, poly_order: int):
        set0 = deepcopy(cocotb_settings)
        set0['params'] = {'BITWIDTH': bitwidth, 'POLY_ORDER': poly_order}
        run_cocotb_sim_for_src_dir(**set0)


if __name__ == '__main__':
    pytest.main([__file__])
