import pytest
from copy import deepcopy

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
from elasticai.creator_plugins.datarate.tests.filter_polydec_asic_tb import cocotb_settings


#@pytest.mark.simulation
@pytest.mark.parametrize(
    ["bitwidth", "poly_order"], [
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (4, 0),
        (4, 2),
        (8, 1),
        (8, 2),
        (12, 1),
        (12, 3),
        (16, 1),
        (16, 2)
    ]
)
def test_polydec_filter_asic(bitwidth: int, poly_order: int):
    set0 = deepcopy(cocotb_settings)
    set0['params'] = {'BITWIDTH': bitwidth, 'POLY_ORDER': poly_order}
    run_cocotb_sim_for_src_dir(**set0)


if __name__ == '__main__':
    pytest.main([__file__])
