import cv2
import matplotlib.pyplot as plt
import numpy as np

path1 = "D:/OneDrive - postech.ac.kr/비전펜듈럼/hyeonggwangdeungdo/"
path2 = "D:/OneDrive - postech.ac.kr/비전펜듈럼/baekyeoldeung only/"

a = []
b = []
for i in range(0, 9):
    a.append(plt.imread(path1 + "{}.png".format(i)))
    b.append(plt.imread(path2 + "{}.png".format(i)))

for i in range(0, 8):
    print((a[i+1] - a[i]).shape)
    new = cv2.resize(a[i+1] - a[i], dsize=(100, 100), interpolation=cv2.INTER_LANCZOS4)
    plt.imshow(new)
    plt.show()
    #cv2.imshow("{}".format(i), cv2.resize(a[i+1] - a[i], dsize=(100, 100), interpolation=cv2.INTER_LANCZOS4))
    #cv2.waitKey(0)