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
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import math

#img = Image.open('/content/drive/MyDrive/CV materials/hw5.png')
image=cv2.imread('/content/drive/MyDrive/CV materials/hw11.jpg')
cv2_imshow(image)

from skimage import segmentation, color
from skimage.io import imread
from skimage.future import graph
from matplotlib import pyplot as plt

img_segments = segmentation.slic(image, compactness=20, n_segments=163)
superpixelated_image = color.label2rgb(img_segments, image, kind='avg')

cv2_imshow(img_segments)

plt.imshow(superpixelated_image)

from skimage.measure import regionprops

regions = regionprops(img_segments)
all_centers=[]
for props in regions:
    cx, cy = props.centroid
    all_centers.append([cx,cy])

 Contrast cue and Spatial cues

import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import math
img2=superpixelated_image
imgarray = img2.reshape((-1, 3)).astype('float32')

'''criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
k = 6
_, labels, centers = cv2.kmeans(imgarray, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)'''

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
K = 10
retval, labels, centers = cv2.kmeans(imgarray, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
unique, counts = np.unique(labels, return_counts=True)
N=len(imgarray)
print(N)


Contrast cue

def find_contrast_cue(N,centers):
  contrast_cue_values=[]
  numclusters=len(centers)
  for i in range(numclusters):
    cur_center=centers[i]
    init_sum=0
    for j in range(numclusters):
      if i != j:
        ni=counts[j]
        other_center=centers[j]
        l2norm=math.sqrt(pow(cur_center[0]-other_center[0],2)+pow(cur_center[1]-other_center[1],2)+pow(cur_center[2]-other_center[2],2))
        init_sum=init_sum + ((ni*l2norm)/N)

    contrast_cue_values.append(init_sum)

  return contrast_cue_values

contrast_cue=find_contrast_cue(N,centers)
print(contrast_cue)
contrast_cue_img=np.zeros((img2.shape[0],img2.shape[1]))
masked_image_contrast = contrast_cue_img.reshape((-1))
for i in range(N):
  masked_image_contrast[i]=contrast_cue[labels[i][0]]
masked_image_contrast = masked_image_contrast.reshape((img2.shape[0],img2.shape[1]))
from matplotlib import pyplot as plt
plt.imshow(masked_image_contrast,cmap='Greys', interpolation='nearest')

Spatial cue

def arrangeclusterwise(orig_image,flat_image,labels):
  all_clusters=[]
  unique, counts = np.unique(labels, return_counts=True)
  for i in range(len(unique)):
    all_clusters.append([])
  for i in range(len(flat_image)):
    all_clusters[labels[i][0]].append([flat_image[i], [i // orig_image.shape[0], i % orig_image.shape[1]]])
  return all_clusters

all_clusters=arrangeclusterwise(img2,imgarray,labels)

def find_spatial_cue(all_clusters,image):
  spatial_cue_values=[]
  #center_pixel=image[int(image.shape[0]/2),int(image.shape[1]/2)]
  center_pixel=[image.shape[0]//2,image.shape[1]//2]
  sigma=math.sqrt(image.shape[0]**2+image.shape[1]**2)
  for j in range(len(all_clusters)):
    init_sum=0
    for i in range(len(all_clusters[j])):
        cur_pixel_xy=all_clusters[j][i][1]
        l2norm=(cur_pixel_xy[0]-center_pixel[0])**2 + (cur_pixel_xy[1]-center_pixel[1])**2
        expterm=math.exp(-l2norm/sigma)
        init_sum=init_sum+expterm
    #init_sum=init_sum/len(all_clusters[j])
    init_sum=init_sum
    
    spatial_cue_values.append(init_sum)

  return spatial_cue_values

spatial_cue=find_spatial_cue(all_clusters,img2)
print(spatial_cue)

def rescale_vals(inp_array):
  new_array=[]
  min_val=min(inp_array)
  max_val=max(inp_array)
  for i in inp_array:
    newval = ((i - min_val)*(255/(max_val-min_val)))
    new_array.append(newval)
  return new_array

rescaled_spat=rescale_vals(spatial_cue)
print(rescaled_spat)

spatial_cue_img=np.zeros((img2.shape[0],img2.shape[1]))
masked_image_spatial = spatial_cue_img.reshape((-1))
for i in range(N):
  #masked_image_spatial[i]=spatial_cue[labels[i][0]]
  masked_image_spatial[i]=rescaled_spat[labels[i][0]]
masked_image_spatial = masked_image_spatial.reshape((img2.shape[0],img2.shape[1]))
from matplotlib import pyplot as plt
plt.imshow(masked_image_spatial,cmap='Greys', interpolation='nearest')

final_cue=[]
for i in range(len(all_clusters)):
  final_cue.append(contrast_cue[i]*spatial_cue[i])
final_cue

final_cue_img=np.zeros((img2.shape[0],img2.shape[1]))
masked_image_final = final_cue_img.reshape((-1))
for i in range(N):
  masked_image_final[i]=final_cue[labels[i][0]]
masked_image_final = masked_image_final.reshape((img2.shape[0],img2.shape[1]))
from matplotlib import pyplot as plt
plt.imshow(masked_image_final,cmap='Greys', interpolation='nearest')

## Question-3

Calculation of Special measure (Phi)

def otsu_algorithm(grayimage):
  total_pixels=grayimage.size
  histogram=cv2.calcHist([grayimage],[0],None,[256],[0,256])
  histogram=histogram.astype(int)
  intra_class_array=[]
  #so histogram has 255 values 
  for i in range(1,len(histogram)):
    part1,part2=np.split(histogram,[i])
    w1=(np.sum(part1))/total_pixels
    w2=(np.sum(part2))/total_pixels
    #weights of both classes done, now calculate variance of both the classes
    mean1=np.sum([x*y for x,y in enumerate(part1)])/(np.sum(part1)+0.00001)  #here x is the pixel value and y is the number of pixels for that value
    mean2=np.sum([x2*y2 for x2,y2 in enumerate(part2)])/(np.sum(part2)+0.0001)
    variance1=np.sum([(x-mean1)**2*y for x,y in enumerate(part1)])
    variance2=np.sum([(x-mean2)**2*y for x,y in enumerate(part2)])
    variance1=np.nan_to_num(variance1)
    variance2=np.nan_to_num(variance2)
    mean1=np.nan_to_num(mean1)
    mean2=np.nan_to_num(mean2)
    intra_class=(variance1)+(variance2)
    intra_class_array.append(intra_class)

  min_value=np.argmin(intra_class_array)
  return min_value

def findDfandDb(fgmean,bgmean,fgstd,bgstd,z):
  Df=math.exp(-1*((z-fgmean)/fgstd)**2)/(fgstd*(math.sqrt(2*3.14)))
  Db=math.exp(-1*((z-bgmean)/bgstd)**2)/(bgstd*(math.sqrt(2*3.14)))
  return [Df,Db]

def findLs(fgmean,bgmean,fgstd,bgstd,z_1,z_2):
  term1=0
  z_1=int(z_1)
  z_2=int(z_2)
  for dz in range (0,z_1+1):
    curdf=findDfandDb(fgmean,bgmean,fgstd,bgstd,z_1)
    curdf=curdf[0]
    term1=term1+curdf

  term2=0
  for dz in range (z_2,2):
    curdb=findDfandDb(fgmean,bgmean,fgstd,bgstd,z_2)
    curdb=curdb[1]
    term2=term2+curdb

  return term1+term2

def separation_measure(img):
  threshold=otsu_algorithm(np.float32(img))
  if threshold==0:
    threshold=threshold+30
  print(threshold)
  histogram=cv2.calcHist([np.float32(img)],[0],None,[256],[0,256])
  histogram=histogram.astype(int)
  part1,part2=np.split(histogram,[threshold])
  total_pixels=img.size
  w1=(np.sum(part1))/total_pixels
  w2=(np.sum(part2))/total_pixels
  #weights of both classes done, now calculate variance of both the classes
  mean1=np.sum([x*y for x,y in enumerate(part1)])/(np.sum(part1)+0.00001)  #here x is the pixel value and y is the number of pixels for that value
  mean2=np.sum([x2*y2 for x2,y2 in enumerate(part2)])/(np.sum(part2)+0.0001)
  variance1=np.sum([((x-mean1)**2)*y for x,y in enumerate(part1)])
  variance2=np.sum([((x-mean2)**2)*y for x,y in enumerate(part2)])
  variance1=np.nan_to_num(variance1)
  variance2=np.nan_to_num(variance2)
  mean1=np.nan_to_num(mean1)
  mean2=np.nan_to_num(mean2)

  #foreground = part1 - Pixels with intensity less than threshold
  #background = part2 - Pixels with intensity greater than threshold.
  fg=part1
  bg=part2
  fgmean=mean1
  bgmean=mean2
  fgstd=math.sqrt(variance1)
  bgstd=math.sqrt(variance2)

  fg_zstar=0
  term1=((bgmean*(fgstd**2))-(fgmean*(bgstd**2)))/((fgstd**2)-(bgstd**2))
  term2=(fgstd*bgstd)/((fgstd**2)-(bgstd**2))
  term3=math.sqrt(((fgmean-bgmean)**2)-(2*((fgstd**2)-(bgstd**2))*((math.log(bgstd)-math.log(fgstd)))))

  zstar_1=term1+(term2*term3)
  zstar_2=term1-(term2*term3)

  op=findLs(fgmean,bgmean,fgstd,bgstd,zstar_1,zstar_2)

  phi=1/(1+math.log(1+(255*op),10))

  return phi




phi_contrast=separation_measure(masked_image_contrast)
print("Phi Contrast is:" , phi_contrast)
phi_spatial=separation_measure(masked_image_spatial)
print("Phi spatial is:" , phi_spatial)


final_cue=[]
for i in range(len(all_clusters)):
  final_cue.append((phi_contrast*contrast_cue[i])+(phi_spatial*rescaled_spat[i]))
final_cue

final_cue_img=np.zeros((img2.shape[0],img2.shape[1]))
masked_image_final = final_cue_img.reshape((-1))
for i in range(N):
  masked_image_final[i]=final_cue[labels[i][0]]
masked_image_final = masked_image_final.reshape((img2.shape[0],img2.shape[1]))
from matplotlib import pyplot as plt
plt.imshow(masked_image_final,cmap='Greys', interpolation='nearest')