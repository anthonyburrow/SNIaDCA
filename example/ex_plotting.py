import numpy as np
from SNIaDCA import GMM


m = np.array([-19.4, -19.55, -19.4, -18.5])
v = np.array([11, 10.5, 14, 11.25])
p5 = np.array([23, 10, 15, 53])
p6 = np.array([105, 62, 149, 125])

model = GMM(pew_5972=p5, pew_6355=p6, M_B=m, vsi=v)
# model = GMM(M_B=m, vsi=v)

# Plot against data set used to generate GMMs.
# ax.scatter() keywords may be passed to change the properties of the
# new point markers.
fig, ax = model.plot(contours=False, marker='*', color='k')

fn = './example_plot.pdf'
fig.savefig(fn)
