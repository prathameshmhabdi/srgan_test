import cv2
frame=[]

for i in range(1000):
	img=cv2.imread("high_res_result_image_"+str(i)+".png")
	frame.append(img)


out=cv2.VideoWriter("video.avi",cv2.VideoWriter_fourcc(*'DIVX'),30,(384,384))

for i in range(1000):
	out.write(frame[i])

out.release()
