import numpy as np
import matplotlib.pyplot as plt


def extract_colors(cm, *args):
    colors = []
    cmap = plt.cm.get_cmap(cm)

    for arg in args:
        colors.append([c * 255 for c in cmap(arg)][0:3])

    return np.array(colors)


branch_color_scheme = np.array([
    [255, 55, 41],    # red
    [102, 245, 59],   # green
    [59, 199, 245],   # blue
    [202, 59, 245]    # magenta
])

polin_color_scheme = np.array([
    [61, 16, 89],     # purple
    [39, 242, 222],   # cyan
    [235, 242, 29]    # yellow
])


branch_markers = ['o', '^', 'D', 's']
polin_markers = ['o', 'd', 's']


def GMM_P_to_MARKER(gmm_p):
    # gmm_p = gmm_p.T

    n_components = gmm_p.shape[1]
    if n_components == 3:
        markers = polin_markers
    elif n_components == 4:
        markers = branch_markers

    m = np.array([markers[np.argmax(p)] for p in gmm_p])
    return m


def GMM_P_to_RGB(gmm_p):
    # gmm_p = gmm_p.T

    rgb = [n_to_rgb(*point) for point in gmm_p]

    return np.array(rgb)


def n_to_rgb(*args):
    n_components = len(args)
    if n_components == 3:
        color_scheme = polin_color_scheme.copy()
    elif n_components == 4:
        color_scheme = branch_color_scheme.copy()

    # c_arr = color_scheme[0:len(args)].T / 255
    c_arr = color_scheme.T / 255
    rgb = c_arr.dot(np.array(args))
    return tuple(rgb)
