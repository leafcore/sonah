import cv2
import numpy as np
import sys
import pytesseract
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['savefig.dpi'] = 1000
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

reload(sys)
sys.setdefaultencoding('utf-8')


# Read image with opencv
img = cv2.imread("R61.jpg")

#width, height = cv2.GetSize(src)

print img.size
#height , width,channel = img.shape
#print width,height




# Read image with opencv
img = cv2.imread("T.jpg")



#print img.size
#width , height = img.size

height , width,channel = img.shape
img = cv2.resize(img,(3*width, 3*height), interpolation = cv2.INTER_LINEAR)
img = cv2.bitwise_not(img)

# Convert BGR to HSV
#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#img_cropped = cv2.cvtColor(img_cropped,cv2.COLOR_BGR2GRAY)

# define range of blue color in HSV
lower_lightblack = np.array([80,80,80])
upper_grey = np.array([255,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(img, lower_lightblack, upper_grey)
mask = cv2.bitwise_not(mask)

mask = cv2.GaussianBlur(mask,(5,5),0)
ret3,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

height=3*height
width=3*width


# Bitwise-AND mask and original image
#res = cv2.bitwise_and(img,img, mask= mask)


#image = cv2.imread("D:/newapproach/B&W"+str(i)+".png")

#height,width = image.shape[:2]
#gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY) # grayscale
_,thresh = cv2.threshold(mask,150,255,cv2.THRESH_BINARY_INV) # threshold
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
_, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

img = np.zeros([height, width, 3], dtype=np.uint8)
img =cv2.bitwise_not(img)

# for each contour found, draw a rectangle around it on original image
index=0
for contour in contours:
	# get rectangle bounding contour
	[x,y,w,h] = cv2.boundingRect(contour)
	if h<w and h<110 : 
	   cv2.drawContours(mask, contour, -1, (255,255,255), 5) 
	   continue	
	   # discard areas that are too large
	
	if h>300 and w>300:
	   cv2.drawContours(mask, contour, -1, (255,255,255), 5)
	   continue

		# discard areas that are too small
	if h<40 or w<40:
	   cv2.drawContours(mask, contour, -1, (255,255,255), 5)
	   continue
	   #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2) 
	    	
	#img=mask		
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2) 	
	
	
	index+=1
	

	
# draw rectangle around contour on original image
# cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
	
#img = cv2.bitwise_and(image, img)	
# remove the contours from the image and show the resulting images
#img = cv2.bitwise_not(img)
		
#cv2.imshow("Blank.jpg", img)
#img = cv2.bitwise_not(img)
cv2.imwrite('D:/newapproach/SelectedText.jpg', img)
cv2.imwrite('D:/newapproach/beforeCrop.jpg', mask)
#mask =cv2.bitwise_not(mask)
#cv2.imwrite('D:/newapproach/Afterinvert.jpg', mask)

# write original image with added contours to disk  




#print width,height
ichl1=int(round(height*(.3009)))
ichu1=int(round(height*(.5555)))
icwl1=int(round(width*(.1770)))
icwu1=int(round(width*(.3125)))

ichl2=int(round(height*(.6481)))
ichu2=int(round(height*(.8981)))
icwl2=int(round(width*(.1770)))
icwu2=int(round(width*(.3125)))


icwl3=int(round(width*(.4687)))
icwu3=int(round(width*(.5312)))

icwl4=int(round(width*(.5520)))
icwu4=int(round(width*(.625)))

icwl5=int(round(width*(.6458)))
icwu5=int(round(width*(.7187)))

icwl6=int(round(width*(.75)))
icwu6=int(round(width*(.8229)))

icwl7=int(round(width*(.8437)))
icwu7=int(round(width*(.9166)))

icwl14=int(round(width*(.4687)))
icwu14=int(round(width*(.9166)))

image_crop_hlb1=int(round(height*(.3009)))
image_crop_hub1=int(round(height*(.5555)))

image_crop_hlb2=int(round(height*(.6481)))
image_crop_hub2=int(round(height*(.8981)))

image_crop_wlb=int(round(width*(.4687)))
image_crop_wub1=int(round(width*(.9375)))



ImgList = []


img_cropped1 = mask[ichl1:ichu1, icwl1:icwu1]           #crop_img(img, 0.64)
ImgList.insert(0,img_cropped1)

img_cropped2 = mask[ichl2:ichu2, icwl2:icwu2] 
ImgList.insert(1,img_cropped2)



img_croppedscore1 = mask[image_crop_hlb1:image_crop_hub1, image_crop_wlb:image_crop_wub1]           #crop_img(img, 0.64)
img_croppedscore2 = mask[image_crop_hlb2:image_crop_hub2, image_crop_wlb:image_crop_wub1] 
"""
img_cropped3 = mask[ichl1:ichu1, icwl3:icwu3]   
ImgList.insert(2,img_cropped3)

img_cropped4 = mask[ichl1:ichu1, icwl4:icwu4]  
ImgList.insert(3,img_cropped4)  
img_cropped5 = mask[ichl1:ichu1, icwl5:icwu5]   
ImgList.insert(4,img_cropped5) 
img_cropped6 = mask[ichl1:ichu1, icwl6:icwu6]   
ImgList.insert(5,img_cropped6) 
img_cropped7 = mask[ichl1:ichu1, icwl7:icwu7]        #crop_img(img, 0.64)
ImgList.insert(6,img_cropped7) 

img_cropped8 = mask[ichl2:ichu2, icwl14:icwu14] 
ImgList.insert(7,img_cropped8) 
img_cropped9 = mask[ichl2:ichu2, icwl3:icwu3]   
ImgList.insert(8,img_cropped9) 
img_cropped10 = mask[ichl2:ichu2, icwl4:icwu4]    
ImgList.insert(9,img_cropped10) 
img_cropped11 = mask[ichl2:ichu2, icwl5:icwu5] 
ImgList.insert(10,img_cropped11)   
img_cropped12 = mask[ichl2:ichu2, icwl6:icwu6] 
ImgList.insert(11,img_cropped12)    
img_cropped13 = mask[ichl2:ichu2, icwl7:icwu7]   
ImgList.insert(12,img_cropped13) 

ImgList.insert(13,img_croppedscore1)
ImgList.insert(14,img_croppedscore2)
"""
ImgList.insert(2,img_croppedscore1)
ImgList.insert(3,img_croppedscore2)
#img_cropped14 = img[ichl1:ichu1, icwl14:icwl14] 


print len(ImgList)

#print os.path.join(os.path.expanduser('~'),'uploads',"B&W"+"1"+".png")

def extract(ilist):
	i=1 
	for img_cropped in ilist:
		
    	
		img_cropped = cv2.bitwise_not(img_cropped)
		#s=settings.MEDIA_ROOT
		cv2.imwrite("D:/newapproach/B&W"+str(i)+".png", img_cropped)
		img_cropped=mpimg.imread("D:/newapproach/B&W"+str(i)+".png")
		
		plt.axis('off')
		plt.imshow(img_cropped,cmap='gray')
		h,w= (img_cropped.shape)
		h=round(h/1000)
		w=round(w/1000)
		print plt.get_backend()
		plt.savefig('D:/newapproach/B&W'+str(i)+'.jpg',dpi=1000.0,format='jpg',bbox_inches='tight')
		


		im=Image.open("D:/newapproach/B&W"+str(i)+".jpg")
		im.save("D:/newapproach/B&W"+str(i)+".jpg",dpi=(300,300))
		
		


	  
		result = pytesseract.image_to_string(Image.open("D:/newapproach/B&W"+str(i)+".jpg"),lang="eng")
		print result
		f= open("D:/newapproach/B&W"+str(i)+".txt","w")
		f.write(result)
		f.close()
		i+=1

		
extract(ImgList)

