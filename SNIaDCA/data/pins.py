import numpy as np


_dtype = [('M_B', np.float64), ('vsi', np.float64),
          ('pew_5972', np.float64), ('pew_6355', np.float64)]

# Group order: core_normal, shallow_si, broad_line, cool
branch_pins = np.array(
    [(-19.35, 11.8, 23, 105), (-19.55, 10.5, 10, 62),
     (-19.4, 14, 15, 149), (-18.5, 11.25, 53, 125)],
    dtype=_dtype)

# Group order: main, fast, dim
polin_pins = np.array(
    [(-19.5, 11.3, 16, 90), (-19.2, 14, 15, 150), (-18.6, 10.7, 55, 125)],
    dtype=_dtype)
