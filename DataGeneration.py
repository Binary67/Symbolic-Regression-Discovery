import numpy as np

def GenerateDummyData(SampleSize, NoiseSigma):
  ActualLaw = lambda x: 3*np.sin(x) + 0.5*x**2
  
  X = np.linspace(-4, 4, SampleSize)
  Y = ActualLaw(X) + np.random.normal(0, NoiseSigma, SampleSize)
  Data = {"X": X, "Y": Y}

  return Data
