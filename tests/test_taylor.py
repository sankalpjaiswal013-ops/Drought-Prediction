import matplotlib.pyplot as plt
import numpy as np
import skill_metrics as sm

sdev = np.array([1.0, 0.8, 1.2])
crmse = np.array([0.0, 0.5, 0.6])
ccoef = np.array([1.0, 0.8, 0.7])
labels = ['Ref', 'Model1', 'Model2']

fig = plt.figure(figsize=(8, 8))
sm.taylor_diagram(sdev, crmse, ccoef, markerLabel=labels, markercolor='w')
plt.savefig("test_taylor.png")
