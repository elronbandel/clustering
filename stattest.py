import scikit_posthocs as sp
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Analysis:
  def __init__(self, results):
    self.results = pd.DataFrame(results)
    self.melted = self.results.melt(var_name='parameter', value_name='score')

  def plot(self):
    g = sns.catplot(x='parameter', y='score', kind="violin", inner=None, data=self.melted)
    plt.show()

  def annova(self):
    fvalue, pvalue = stats.f_oneway(*[self.results[c] for c in self.results])
    return fvalue, pvalue
    
  def mean(self):
    return self.results.mean(axis=0)

  def std(self):
    return self.results.std(axis=0)

  def scheffe(self):
    res = sp.posthoc_scheffe(self.melted, val_col='score', group_col='parameter')
    return res