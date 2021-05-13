import numpy as np
from PIL import Image
import cv2
image = Image.open('hw1.png')
data = np.asarray(image)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import math

contours, hierarchy = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

def finddistance(p1,p2):
  dist=math.sqrt(math.pow(p2[1]-p1[1],2)+math.pow(p2[0]-p1[0],2))
  return dist

def presentincircle(point,circle):
  center=circle[0]
  radius=circle[1]
  distance=finddistance(point,center)
  if distance > radius:
    return False
  else:
    return True

def welzl(P,R):

  if not P or len(R)==3:
    if len(R)==1:
      radius=0;
      center=R[0]
      circle=[center,radius]
      return circle

    elif len(R)==2:
      point1=R[0]
      point2=R[1]
      cx=(point1[0]+point2[0])/2
      cy=(point1[1]+point2[1])/2
      radius=finddistance([cx,cy],point1)
      circle=[]
      circle.append([cx,cy])
      circle.append(radius)
      return circle

    elif len(R)==3:
      point1=R[0]
      point2=R[1]
      point3=R[2]
      temp1=(point2[0]-point1[0])*(point2[0]+point1[0])*0.5 + (point2[1]-point1[1])*(point2[1]+point1[1])*0.5
      temp2=(point3[0]-point1[0])*(point3[0]+point1[0])*0.5 + (point3[1]-point1[1])*(point3[1]+point1[1])*0.5
      temp3=(point2[0]-point1[0])*(point3[1]-point1[1]) - (point2[1]-point1[1])*(point3[0]-point1[0])+0.2
      #print(temp1,temp2,temp3)
      centerx=(((point3[1]-point1[1])*temp1 - (point2[1]-point1[1])*temp2)/temp3)
      centery=(((point1[0]-point3[0])*temp1 + (point2[0]-point1[0])*temp2)/temp3)

      radius=int(math.sqrt(math.pow(point1[1]-centery,2)+math.pow(point1[0]-centerx,2)))
      #circle=[(centerx,centery),radius]
      circle=[]
      circle.append([centerx,centery])
      circle.append(radius)
      return circle
    else:
      return None

  else:
    #not trivial so remove a value randomly from P
    index=random.randint(0,len(P)-1)
    point=P[index]
    Pcopy=P.copy()
    Pcopy.remove(point)
    D=welzl(Pcopy,R)

    if D is not None and presentincircle(point,D):
      return D

    else:
      Rcopy=R.copy()
      Rcopy.append(point)
      return welzl(Pcopy,Rcopy)

final_answers=[]
datacopy=np.asarray(image)
for i in range(len(contours)):
  a=contours[i].tolist()
  circleFound=False
  b=[]
  if i==7 :
    limit=len(a)-100
  else:
    limit=len(a)
  for j in range(limit):
    b.append(a[j][0])
  q=[]
  temp=[]
  for iter in range (10):
    circle = welzl(b, q)
    temp.append(circle)

  cur_min=999999999
  cur_center=[]
  for m in range(len(temp)):
    cur_circle=temp[m]
    cur_radius=cur_circle[1]
    if cur_radius<cur_min:
      cur_min=cur_radius
      cur_center=cur_circle[0]
  cur=[]
  center=cur_center
  radius=int(cur_min)
  cur.append(center)
  cur.append(radius)
  final_answers.append(cur)
  centerx=int(center[0])
  centery=int(center[1])
  datacopy=cv2.circle(datacopy, (centerx,centery), radius, (255, 0, 0), thickness=1, lineType=8, shift=0)
  print("1")

cv2_imshow(datacopy)

df=pd.DataFrame(np.asarray(final_answers))

def countobjects(image):
    data=image.copy()
    rows = len(data)
    cols =len(data[0])
    all_objects=[]
    count = 0
    for i in range(rows):
        for j in range(cols):
            if data[i][j] == 255:
                count += 1
                current_object=[]
                current_object.append([i,j])
                stack = [(i,j)]
                while stack:
                    ii, jj = stack.pop()
                    if 0<=ii<rows and 0<=jj<cols and data[ii][jj] == 255:
                        current_object.append([ii,jj])
                        data[ii][jj] = 0
                        stack.extend([(ii+1,jj),(ii-1,jj),(ii,jj-1),(ii,jj+1)])
                
                all_objects.append(current_object)
                
    return all_objects


all_objects=countobjects(data)
all_objects_masks=[]
for i in range(len(all_objects)):
  black=np.zeros(data.shape)
  current_object=all_objects[i]
  for i in range(len(current_object)):
    black[current_object[i][0],current_object[i][1]]=255

  all_objects_masks.append(black)

cv2_imshow(all_objects_masks[7])

all_circle_masks=[]
for j in range(len(final_answers)):
  cur_circlea=final_answers[j]
  xc=int(cur_circlea[0][0])
  yc=int(cur_circlea[0][1])
  radiuss=int(cur_circlea[1])
  mask = np.zeros(data.shape)
  mask = cv2.circle(mask, (xc,yc), radiuss, (255,255,255), -1)
  print(mask)
  #cv2_imshow(mask)
  all_circle_masks.append(mask)


all_circle_masks.reverse()

def jaccard(image1,image2):
  intersection=0
  union=0
  for i in range(image1.shape[0]):
    for j in range(image1.shape[1]):
      if image1[i][j]==255 and image2[i][j]==255:
        intersection=intersection+1
        union=union+1
      elif image1[i][j]==255 and image2[i][j]==0:
        union=union+1
  
  score=intersection/union
  return score


jaccard_scores=[]
for n in range(len(all_objects)):
  cur_score=jaccard(all_circle_masks[n],all_objects_masks[n])
  jaccard_scores.append(cur_score)

da=pd.DataFrame(jaccard_scores)
np.savetxt('jaccard.txt',da.values,fmt='%s')


