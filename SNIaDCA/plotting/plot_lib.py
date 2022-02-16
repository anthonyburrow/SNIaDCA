from matplotlib.ticker import MultipleLocator

from .contour import draw_contours
from .plot_color_mapping import *

_ax_label = {
    'm': r'$\mathrm{M_B\ [mag]}$',
    'v': r'$\mathrm{Si\ II\ \lambda 6355\ Velocity\ [1000\ km/s]}$',
    'p5': r'$\mathrm{Si\ II\ \lambda 5972\ pEW\ [Å]}$',
    'p6': r'$\mathrm{Si\ II\ \lambda 6355\ pEW\ [Å]}$'
}

_ax_range = {
    'm': (-17., -20.3),
    'v': (8., 17.),
    'p5': (-5., 81.),
    'p6': (0., 200.)
}

_ax_minor_tick = {
    'm': 0.25,
    'v': 0.5,
    'p5': 10.,
    'p6': 12.5
}

_which_map = {
    'p5_p6': (0, 1),
    'v_p5_p6': (2, 1),
    'm_p5_p6': (2, 1),
    'm_v_p5_p6': (2, 3)
}


def _scatter(ax, X, Y, colors, markers, s=20, lw=0.6):
    for x, y, c, m in zip(X, Y, colors, markers):
        ax.scatter(x, y, color=c, marker=m, s=s, edgecolor='k', linewidths=lw)


def gmm_legend(ax, n, *args, **kwargs):
    # test points
    point = (-1000, -1000)
    if n == 3:
        labels = ('Main', 'Fast', 'Dim')
        color_scheme = polin_color_scheme / 255
        markers = polin_markers
    if n == 4:
        labels = ('CN', 'SS', 'BL', 'CL')
        color_scheme = branch_color_scheme / 255
        markers = branch_markers

    for i in range(n):
        col = color_scheme[i]
        mark = markers[i]
        label = labels[i]
        ax.scatter(*point, color=col, marker=mark, s=16, edgecolor='k',
                   linewidths=0.7, label=label)

    ax.legend(frameon=False, *args, **kwargs)


def plot_polin(fig, ax, data, output, prob, contour=False, gmm=None):
    if contour:
        assert gmm is not None

    if contour:
        draw_contours(ax, gmm, (0, 30), (-22, -16))

    c = GMM_P_to_RGB(prob)
    m = GMM_P_to_MARKER(prob)
    _scatter(ax, data['v6150'], data['mag'], c, m)

    ax.set_xlim(_ax_range['v'])
    ax.set_ylim(_ax_range['m'])

    ax.set_xlabel(_ax_label['v'])
    ax.set_ylabel(_ax_label['m'])

    ax.xaxis.set_minor_locator(MultipleLocator(_ax_minor_tick['v']))
    ax.yaxis.set_minor_locator(MultipleLocator(_ax_minor_tick['m']))

    gmm_legend(ax, prob.shape[1],
               loc='lower right', bbox_to_anchor=(0.98, 0))

    fig.tight_layout()

    return fig, ax


def plot_branch(fig, ax, gmm, contours=False, gmm_model=None):
    if contours:
        draw_contours(ax, gmm_model, (-30, 250), (-30, 130),
                      which=_which_map[gmm.model])

    prob = gmm.predict()
    c = GMM_P_to_RGB(prob)
    m = GMM_P_to_MARKER(prob)
    _scatter(ax, gmm.pew_6355, gmm.pew_5972, c, m)

    if not contours:
        ax.errorbar(gmm.pew_6355, gmm.pew_5972, xerr=gmm.pew_6355_err,
                    yerr=gmm.pew_5972_err, fmt='none', ecolor='#4a4a4a',
                    elinewidth=0.7, capthick=0.7, capsize=1.3, zorder=-1)

    ax.set_xlabel(_ax_label['p6'])
    ax.set_ylabel(_ax_label['p5'])

    ax.set_xlim(_ax_range['p6'])
    ax.set_ylim(_ax_range['p5'])

    ax.xaxis.set_minor_locator(MultipleLocator(_ax_minor_tick['p6']))
    ax.yaxis.set_minor_locator(MultipleLocator(_ax_minor_tick['p5']))

    gmm_legend(ax, prob.shape[1],
               loc='upper left', bbox_to_anchor=(-0.02, 0.99))

    fig.tight_layout()

    return fig, ax
