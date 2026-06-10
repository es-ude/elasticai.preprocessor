import pytest
from copy import deepcopy

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
from elasticai.creator_plugins.datarate.tests.filter_cic_tb import cocotb_settings


#@pytest.mark.simulation
@pytest.mark.parametrize(
    ["bitwidth", "dec_rate", "n_dec"], [
        (2, 2, 2),
        (6, 2, 2),
        (8, 2, 2),
        (12, 2, 2),
        (16, 2, 2),
    ]
)
def test_verilog_polydec_asic(bitwidth: int, dec_rate: int, n_dec: int):
    sets = deepcopy(cocotb_settings)
    sets['params'] = {'BITWIDTH': bitwidth, 'DEC_RATE': dec_rate, 'N_DEC': n_dec}
    run_cocotb_sim_for_src_dir(**sets)


if __name__ == '__main__':
    pytest.main([__file__])
