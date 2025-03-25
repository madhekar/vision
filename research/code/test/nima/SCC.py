import numpy as np

def _scc_single(I1,I2,win,ws):
  def _scc_filter(inp, axis, output, mode, cval):
    return correlate(inp, win , output, mode, cval, 0)
  
  # simple HPF operation
  I1_hp = generic_laplace(I1.astype(np.float64), _scc_filter)
  I2_hp = generic_laplace(I2.astype(np.float64), _scc_filter)
  
  # creating window for uniform filter
  win = fspecial(Filter.UNIFORM,ws)
  sigmaI1_sq,sigmaI2_sq,sigmaI1_I2 = _get_sigmas(I1_hp,I2_hp,win)
  
  sigmaI1_sq[sigmaI1_sq<0] = 0
  sigmaI2_sq[sigmaI2_sq<0] = 0
  
  den = np.sqrt(sigmaI1_sq) * np.sqrt(sigmaI2_sq)
  idx = (den==0)
  den = _replace_value(den,0,1)
  scc = sigmaI1_I2 / den
  scc[idx] = 0
  return scc

def scc(I1,I2,win=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],ws=8):
  """
  :param I1: represents original image matrix
  :param P: represents degraded image matrix
  :param win: high pass filter for spatial processing (default=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).
  :param ws: sliding window size (default = 8).
  :returns:  float -- scc value.
  """
  I1,I2 = _initial_check(I1,I2)
  
  coefs = np.zeros(I1.shape)
  for i in range(I1.shape[2]):
    coefs[:,:,i] = _scc_single(I1[:,:,i],I2[:,:,i],win,ws)
  return np.mean(coefs)