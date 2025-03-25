import numpy as np

def sam (I1,I2):
  '''
  :param I1: represents original image matrix
  :param I2: represents degraded image matrix
  :returns:  float -- sam value.
  '''
  I1,I2 = _initial_check(I1,I2)
  
  I1 = I1.reshape((I1.shape[0]*I1.shape[1],I1.shape[2]))
  I2 = I2.reshape((I2.shape[0]*I2.shape[1],I2.shape[2]))
  
  N = I1.shape[1]
  sam_angles = np.zeros(N)
  for i in range(I1.shape[1]):
    val = np.clip(np.dot(I1[:,i],I2[:,i]) / (np.linalg.norm(I1[:,i])*np.linalg.norm(I2[:,i])),-1,1)  
    sam_angles[i] = np.arccos(val)
  
  return np.mean(sam_angles)