import matplotlib.pyplot as plt
import math 
import numpy as np

theta = np.linspace(0, 2*np.pi, 50)
(x_1, y_1) = (10, 10)
(x_2, y_2) = (20, 20)
(x_3, y_3) = (30, 30)
p_1 = []
p_2 = []
p_3 = []
for i in theta:
    p_1.append((x_1)*math.cos(i) + (y_1)*math.sin(i))
    p_2.append((x_2)*math.cos(i) + (y_2)*math.sin(i))
    p_3.append((x_3)*math.cos(i) + (y_3)*math.sin(i))

fig, ax1 = plt.subplots(figsize=(8, 6))


ax1.plot(theta, p_1, linestyle='--', label = '(10, 10)')
ax1.plot(theta, p_2, linestyle='--', label = '(20, 20)' )
ax1.plot(theta, p_3, linestyle='--', label = '(30, 30)')
ax1.legend()
ax1.set_xlabel('theta')
ax1.set_ylabel('p')
plt.show()
