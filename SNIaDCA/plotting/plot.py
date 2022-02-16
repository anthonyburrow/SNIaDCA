import numpy as np
import matplotlib.pyplot as plt
import os

from .setup import setup_plot_params
from .plot_lib import plot_branch, plot_polin, multi_scatter
from .plot_color_mapping import GMM_P_to_RGB, GMM_P_to_MARKER
import SNIaDCA.gmm


_sep = os.path.sep
_data_path = os.path.join(os.path.dirname(__file__), f'..{_sep}data{_sep}')


def generate_plot(gmm, contours=False):
    # Setup plot
    setup_plot_params()
    fig, ax = plt.subplots()

    # Get GMM information from CSP, Zheng source data
    source_data = read_source_data(f'{_data_path}csp_zheng.dat')

    source_gmm = SNIaDCA.gmm.GMM(pew_5972=source_data['p5800'],
                                 pew_5972_err=source_data['p5800_err'],
                                 pew_6355=source_data['p6150'],
                                 pew_6355_err=source_data['p6150_err'],
                                 M_B=source_data['mag'],
                                 M_B_err=source_data['mag_err'],
                                 vsi=source_data['v6150'],
                                 vsi_err=source_data['v6150_err'],
                                 model=gmm.model)

    # Create plot from CSP, Zheng source data
    prob = gmm.predict()
    point_props = {
        'color': GMM_P_to_RGB(prob),
        'marker': '*',
        's': 150.,
        'edgecolor': 'k',
        'lw': 0.8
    }

    if gmm.n_components == 4:
        fig, ax = plot_branch(fig, ax, source_gmm, contours=contours)
        ax.scatter(gmm.pew_6355, gmm.pew_5972, **point_props)
    elif gmm.n_components == 3:
        fig, ax = plot_polin(fig, ax, source_gmm, contours=contours)
        ax.scatter(gmm.vsi, gmm.M_B, **point_props)

    return fig, ax


def read_source_data(filename):
    dt = [('name', 'U8'),
          ('mag', np.float64), ('mag_err', np.float64),
          ('v6150', np.float64), ('v6150_err', np.float64),
          ('p5800', np.float64), ('p5800_err', np.float64),
          ('p6150', np.float64), ('p6150_err', np.float64)]
    data = np.loadtxt(filename, skiprows=1, dtype=dt)

    return data
