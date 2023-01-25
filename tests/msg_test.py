import sys
sys.path.append('../PCWA')
import matplotlib.pyplot as plt
from src.pcwa import pcwa
import numpy as np

y0 = 0
plt.figure(figsize=(10,10),dpi=120)
for n in range(1,12):
    y = pcwa.msg(scale=10,window=1,N=n,skewness=1,dx=0.1)
    x = len(y)
    x = np.linspace(-x/2,x/2,x)
    plt.plot([x[0],x[-1]],[y0,y0],color='black',linestyle='--')
    plt.plot(x,y0+y,label=f'N={n}')
    y0 += np.max(y)
    print(np.sum(y),np.sum(y**2),np.abs(y.max()/y.min()))
plt.legend()
plt.grid(which='both')
plt.tight_layout()
plt.show()