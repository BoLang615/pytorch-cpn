import numpy as np
import matplotlib.pyplot as plt
def nms(det, size = 9):
  border = size
  dr = np.zeros((64 + 2*border, 48 + 2*border))
  dr[border:-border, border:-border] = det.copy()
 # print(dr[border:-border, border:-border].shape)
  pool = np.zeros(dr.shape)
#  print(det.shape[0]) #64
  for i in range(size // 2, dr.shape[0] - size // 2):
    for j in range(size // 2, dr.shape[1] - size // 2):
      pool[i, j] = max(dr[i - 1, j - 1], dr[i - 1, j], dr[i - 1, j + 1], \
                       dr[i, j - 1], dr[i, j], dr[i, j + 1], \
                       dr[i + 1, j - 1], dr[i + 1, j], dr[i + 1, j + 1])
                          
  pool[pool != dr] = 0
  return pool

def parseHeatmap(hm, thresh = 0.8):
  # hm[0]: map, hm[1:] emb
  
  det = hm
 # print(np.min(det))
#  print(det.shape)
#  det[det < thresh] == 0
  det = np.where(det < thresh, 0, det)
  det = nms(det)
 # plt.imshow(det, cmap='gray', interpolation='nearest')
 # plt.show()
 # print(det.shape)
  pts = np.where(det > 0)
  return pts
  
