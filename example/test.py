import numpy as np
from SnIaDCA import gmm


m = np.array([-19.4, -19.55, -19.4, -18.5])
v = np.array([11, 10.5, 14, 11.25])
p5 = np.array([23, 10, 15, 53])
p6 = np.array([105, 62, 149, 125])

test = gmm(p5=p5, p6=p6)
probs = test.predict_group()

print(probs)
