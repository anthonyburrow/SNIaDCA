import numpy as np
import matplotlib.pyplot as plt
import os

from .setup import setup_plot_params
from .plot_lib import plot_branch, plot_polin
from .plot_color_mapping import GMM_P_to_RGB
import SNIaDCA.gmm


_sep = os.path.sep
_data_path = os.path.join(os.path.dirname(__file__), f'..{_sep}data{_sep}')


def generate_plot(gmm, contours=False, *args, **kwargs):
    print('Generating plot...')

    # Setup plot
    setup_plot_params()
    fig, ax = plt.subplots()

    # Get GMM information from CSP, Zheng source data
    source_data = read_source_data(f'{_data_path}csp_zheng.dat')

    source_gmm = SNIaDCA.gmm.GMM(data=source_data,
                                 model=gmm.model)

    # Create plot from CSP, Zheng source data
    prob = gmm.predict(verbose=False)
    point_props = {
        'color': GMM_P_to_RGB(prob),
        'marker': '*',
        's': 150.,
        'edgecolor': 'k',
        'lw': 0.8
    }
    for key, prop in kwargs.items():
        point_props[key] = prop

    if gmm.n_components == 4:
        fig, ax = plot_branch(fig, ax, source_gmm, contours=contours)
        ax.scatter(gmm.pew_6355, gmm.pew_5972, **point_props)
    elif gmm.n_components == 3:
        fig, ax = plot_polin(fig, ax, source_gmm, contours=contours)
        ax.scatter(gmm.vsi, gmm.M_B, **point_props)

    return fig, ax


def read_source_data(filename):
    dt = [('name', 'U8'),
          ('M_B', np.float64), ('M_B_err', np.float64),
          ('vsi', np.float64), ('vsi_err', np.float64),
          ('pew_5972', np.float64), ('pew_5972_err', np.float64),
          ('pew_6355', np.float64), ('pew_6355_err', np.float64)]
    data = np.loadtxt(filename, skiprows=1, dtype=dt)

    return data
