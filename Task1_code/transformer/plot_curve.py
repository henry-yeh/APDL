import matplotlib.pyplot as plt
import re
import numpy as np

path1 = 'slurm/slurm-198880.out'
text1 = open(path1, encoding="utf-8").read()
lst1 = re.findall(r'\| loss  ([0-9\.]*)\s', text1)
lst2 = re.findall(r'valid loss  ([0-9\.]*)\s', text1)

y1 = list(map(float, lst1))
x1 = np.linspace(0, len(lst2), len(lst1))

y2 = list(map(float, lst2))
x2 = [i for i in range(1, len(y2)+1)]

plt.plot(x1,y1, label='Training loss')
plt.plot(x2,y2, label='Validation loss')
plt.xlabel('# of training epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()