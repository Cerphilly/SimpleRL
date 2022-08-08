import cv2
import matplotlib.pyplot as plt
import numpy as np

a = {'Loss': {'Actor': 1, 'Critic1': 2.23423, 'Critic2': 3.23423, 'Alpha': 4},
                'Value': {'Alpha': 5}}
print(a)
for key, value in a.items():
    if key == 'Loss':
        for key2, value2 in value.items():
            value[key2] = round(value[key2], 2)

print(a)