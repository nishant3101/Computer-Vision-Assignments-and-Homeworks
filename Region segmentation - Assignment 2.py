import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import math
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
import argparse
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import quad
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import math

#img = Image.open('/content/drive/MyDrive/CV materials/hw5.png')
image=cv2.imread('/content/drive/MyDrive/CV materials/hw11.jpg')
cv2_imshow(image)

cv2_imshow(image)

image

rows=image.shape[0]
cols=image.shape[1]
maxx=rows-1
maxy=cols-1
normalised_image=[]
new_image=image.copy()
for i in range(0,rows):
  for j in range(0,cols):
    new_rgbxy=np.zeros(5)
    cur_rgb=np.array(image[i][j], dtype='int64')
    cur_rgb[0]=(cur_rgb[0])
    cur_rgb[1]=(cur_rgb[1])
    cur_rgb[2]=(cur_rgb[2])
    total=cur_rgb[0]+cur_rgb[1]+cur_rgb[2]
    total=int(total)
    if total==0:
      total=total+1
    new_rgbxy[0]=((cur_rgb[0])/total)*255
    new_rgbxy[1]=((cur_rgb[1])/total)*255
    new_rgbxy[2]=((cur_rgb[2])/total)*255
    new_rgbxy[3]=i/maxx
    new_rgbxy[4]=j/maxy
    new_image[i][j]=(new_rgbxy[0],new_rgbxy[1],new_rgbxy[2])
    #new_rgbxy=np.array(new_rgbxy, dtype='int64')
    normalised_image.append(new_rgbxy)
normalised_image=np.asarray(normalised_image)
print(len(normalised_image))

def norm2(image):
  rows=image.shape[0]
  cols=image.shape[1]
  new_normalised=[]
  for i in range(rows):
    for j in range(cols):
      temp=np.zeros(5)
      cur_pixel=image[i][j]
      temp[0]=cur_pixel[0]
      temp[1]=cur_pixel[1]
      temp[2]=cur_pixel[2]
      temp[3]=i
      temp[4]=j

      new_normalised.append(temp)
  
  sum=255
  new_normalised = np.asarray(new_normalised).astype('float32')
  new_normalised[:,0]=new_normalised[:,0]/sum
  new_normalised[:,1]=new_normalised[:,1]/sum
  new_normalised[:,2]=new_normalised[:,2]/sum
  new_normalised[:,3]=new_normalised[:,3]/np.max(new_normalised[:,3])
  new_normalised[:,4]=new_normalised[:,4]/np.max(new_normalised[:,4])

  '''for i in range(len(new_normalised)):
    cur=new_normalised[i]
    sum=cur[0]+cur[1]+cur[2]
    if sum==0:
      sum=sum+1
    new_normalised[i][0]=new_normalised[i][0]/sum
    new_normalised[i][1]=new_normalised[1][0]/sum
    new_normalised[i][2]=new_normalised[2][0]/sum
    new_normalised[i][3]=new_normalised[i][3]/np.max(new_normalised[:,3])
    new_normalised[i][4]=new_normalised[i][4]/np.max(new_normalised[:,4])'''

  return new_normalised

    


op=norm2(image)

op[0]

tt=op[:,0]+op[:,1]+op[:,2]

pip install fuzzy-c-means

import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt


fcm = FCM(n_clusters=10)
#fcm.fit(normalised_image)
fcm.fit(op)

# outputs
X=normalised_image
fcm_centers = fcm.centers
#fcm_labels = fcm.predict(normalised_image)
fcm_labels = fcm.predict(op)
# plot result
f, axes = plt.subplots(1, 2, figsize=(11,5))
axes[0].scatter(X[:,0], X[:,1], alpha=.1)
axes[1].scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
#plt.savefig('images/basic-clustering-output.jpg')
plt.show()

fcm.predict(normalised_image[1])

fcm_unique, fcm_counts = np.unique(fcm_labels, return_counts=True)
print(fcm_counts)

# checking the fcm counts with less than threshold
threshold=0.01*(rows*cols)
print("threshold:", threshold)
stray_clusters=[]
large_groups=[]
#threshold=600
for i in range(len(fcm_counts)):
  if fcm_counts[i]<threshold:
    stray_clusters.append(i)
  else:
    large_groups.append(i)

stray_clusters=np.asarray(stray_clusters)
large_groups=np.asarray(large_groups)


stray_clusters

fcm_labels=np.asarray(fcm_labels)

def calcdist(pointi, pointj):
    distance = 0.0
    l=len(pointi)
    for i in range(l):
        distance += (pointj[i] - pointi[i]) ** 2
    return math.sqrt(distance)

#isolated = [2,7,9]
new_fcm_labels=np.zeros(len(fcm_labels))
for i in range(len(fcm_labels)):
  if fcm_labels[i] in stray_clusters:
    #print("hi")
    cur_point=op[i]
    min=999999999999
    index=-1
    for j in range(len(fcm_centers)):
      if j in large_groups:
        center_p=fcm_centers[j]
        dist=calcdist(cur_point,center_p)
        dist=abs(dist)
        if dist<min:
          min=dist
          index=j
    new_fcm_labels[i]=index
  else:
    new_fcm_labels[i]=fcm_labels[i]


new_fcm_unique, new_fcm_counts = np.unique(new_fcm_labels, return_counts=True)
print((new_fcm_unique))
#[0,1,3,4,5,6,8]

segmented_image=[]
for i in range(len(op)):
  segmented_image.append(np.zeros(3))
segmented_image=np.asarray(segmented_image)

for i in range(len(new_fcm_labels)):
  cur_label=int(new_fcm_labels[i])
  cur_color=fcm_centers[cur_label]
  segmented_image[i][0]=cur_color[0]
  segmented_image[i][1]=cur_color[1]
  segmented_image[i][2]=cur_color[2]


final_image=segmented_image.reshape((rows,cols,3))

plt.imshow(final_image)