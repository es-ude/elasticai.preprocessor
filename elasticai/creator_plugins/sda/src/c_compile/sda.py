from datetime import datetime
from pathlib import Path

import elasticai.creator_plugins.sda as design_plugin
from elasticai.preprocessor.translation.ir2c import (
    generate_c_files,
    get_embedded_datatype,
    replace_variables_with_parameters,
)

