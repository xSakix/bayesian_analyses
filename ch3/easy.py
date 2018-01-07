import numpy as np
from scipy.stats import binom
from scipy.stats import mode
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns

p_grid = np.linspace(0.,1.,1000)
prior = np.repeat(1.,1000)
likehood = binom.pmf(6,9,p_grid)
posterior = likehood*prior
posterior = posterior/posterior.sum()

plt.plot(p_grid, posterior)
plt.show()

sample_size = int(10000)
samples = np.random.choice(p_grid,p=posterior,size=sample_size,replace=True)
sns.kdeplot(samples)
plt.show()

print('bellow 0.2:'+str(sum(samples<0.2)/sample_size))
print('above 0.8:'+str(sum(samples>0.8)/sample_size))
print('between 0.2 and 0.8:'+str(sum((samples > 0.2) & (samples < 0.8))/sample_size))
print('20% of posterior probability lies under which p-value?'+str(np.percentile(samples,20)))
print('20% of posterior probability lies above which p-value?'+str(np.percentile(samples,80)))
print('which values of p contain the narrowest interval equal to 66% of posterior?'+str(pm.hpd(samples,alpha=0.66)))
print('which values of p contain the 66% posterior, assuming equal posterior prob bellow and above the interval?')

#it's 17,83 - but lets confirm it
low = 0
up = 66
while low < 35 and up < 100:
    probs = np.percentile(samples,[low,up])    
    llow = sum(samples<probs[0])/sample_size
    uup = sum(samples>probs[1])/sample_size
    if np.abs(llow - uup) > 0.001:
        low += 1
        up += 1
        continue
    print('-----')
    print('<%d,%d>'%(low,up))
    print(str(probs))
    print('how much posterior prob is bellow %f:%s '%(probs[0],str(llow)))
    print('how much posterior prob is above %f:%s'%(probs[1],str(uup)))
    break
    
    
 