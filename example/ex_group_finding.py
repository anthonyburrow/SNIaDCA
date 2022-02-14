import numpy as np
from SNIaDCA import GMM


m = np.array([-19.4, -19.55, -19.4, -18.5])
v = np.array([11, 10.5, 14, 11.25])
p5 = np.array([23, 10, 15, 53])
p6 = np.array([105, 62, 149, 125])

model = GMM(pew_5972=p5, pew_6355=p6)

# Predict group membership probabilities for each input supernova
# (see docstring for the order, as it depends on the model used).
probs = model.predict()
print(probs)

# Get the group with the maximum associated probability
specific_sn_prob = probs[1]
group_name, group_prob = model.get_group_name(specific_sn_prob)
print(f'Most likely group: {group_name} ({group_prob * 100.:.2f}%)')
