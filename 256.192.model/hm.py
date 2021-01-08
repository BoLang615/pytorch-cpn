import numpy as np
import matplotlib.pyplot as plt
def nms(det, size = 5):
  pool = np.zeros(det.shape)
#  print(det.shape[0]) #64
  for i in range(size // 2, det.shape[0] - size // 2):
    for j in range(size // 2, det.shape[1] - size // 2):
      pool[i, j] = max(det[i - 1, j - 1], det[i - 1, j], det[i - 1, j + 1], \
                       det[i, j - 1], det[i, j], det[i, j + 1], \
                       det[i + 1, j - 1], det[i + 1, j], det[i + 1, j + 1])
                          
  pool[pool != det] = 0
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
  pts = np.where(det > 0)
  return pts
  
