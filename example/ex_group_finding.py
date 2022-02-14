import numpy as np
from SNIaDCA import GMM


m = np.array([-19.4, -19.55, -19.4, -18.5])
v = np.array([11, 10.5, 14, 11.25])
p5 = np.array([23, 10, 15, 53])
p6 = np.array([105, 62, 149, 125])

model = GMM(pew_5972=p5, pew_6355=p6)
probs = model.predict()
model.get_group_name(probs[3])

print(probs)
