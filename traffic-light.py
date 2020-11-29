import numpy as np
import matplotlib.pyplot as plt
import cv2


#helper function to mask the image based on color range
def mask_image(image, hsv_image, low_color, high_color, color):
	low_color = np.array(low_color)
	high_color = np.array(high_color)
	color_mask = cv2.inRange(hsv_image, low_color, high_color)

	color_image = cv2.bitwise_and(image, image, mask=color_mask)
	cv2.imshow(color, color_image)

	return color_image


#helper function that returns lists of intensity of masked image based on intensity (number of non black pixels) in the x, y directions
def make_histogram(color_image):
	y_direction_intensity = []
	x_direction_intensity = []

	mask = np.all(color_image != [0,0,0], axis=-1).astype(int)
	y_direction_intensity = np.sum(mask, axis=1)
	x_direction_intensity = np.sum(mask, axis=0)

	return (x_direction_intensity, y_direction_intensity)


#helper function converts an rgb color (list) to an hsv color range
def rgb_to_hsv_color_range(rgb_color):
	rgb_color = np.uint8([[rgb_color]])
	hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)

	upper = [hsv_color[0][0][0]+10 if hsv_color[0][0][0]+10<=179 else 179, 255,200]
	lower = [hsv_color[0][0][0]-10 if hsv_color[0][0][0]-10>=0 else 0,40,20]
	return (upper, lower)


#detect status of traffic signal in an image
#takes image file of traffic light (cropped) as parameter, returns status ("GO", "SLOW DOWN", "STOP")
#assume that the image of the traffic light is closely cropped and only 3 lights per image
def read_traffic_light(image_file):

	image = cv2.imread(image_file)
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	#mask based on color (hsv)
	#mask red
	(upper_red, lower_red) = rgb_to_hsv_color_range([255,0,0])

	red = mask_image(image, hsv_image, lower_red, upper_red, "red")
	(red_x, red_y) = make_histogram(red)

	#mask amber/yellow
	(upper_amber, lower_amber) = rgb_to_hsv_color_range([255,165,0])
	amber = mask_image(image, hsv_image, lower_amber, upper_amber, "amber")
	(amber_x, amber_y) = make_histogram(amber)

	#mask green -
	# (upper_green, lower_green) = rgb_to_hsv_color_range([0,255,0])
	green = mask_image(image, hsv_image, [30, 40, 20], [90, 255, 255], "green")
	(green_x, green_y) = make_histogram(green)

	left = red_x.shape[0]//4
	right = red_x.shape[0] - red_x.shape[0]//4

	#plot histogram for each mask
	plt.subplot(2,1,1)
	plt.plot(red_x[left:right], label="red")
	plt.plot(amber_x[left:right], label="amber")
	plt.plot(green_x[left:right], label="green")
	plt.legend()

	plt.subplot(2,1,2)
	plt.plot(red_y, label="red")
	plt.plot(amber_y, label="amber")
	plt.plot(green_y, label="green")
	plt.legend()

	plt.show()
	cv2.waitKey(0)

	x_plots = {
		"red": red_x,
		"amber": amber_x,
		"green": green_x
	}

	y_plots = {
		"red": red_y,
		"amber": amber_y,
		"green": green_y
	}


	#check max for each color - check that peak is roughly in the middle of x direction (for vertically oriented traffic lights)
	left = red_x.shape[0]//4
	right = red_x.shape[0] - red_x.shape[0]//4

	brightest = 0
	brightest_color = ""

	for color in x_plots:
		max_color = np.amax(x_plots[color][left:right]).astype(int)
		max_position = np.argmax(x_plots[color][left:right]).astype(int)

		#position must be roughly in the center in x direction
		if max_color > brightest:
			brightest = max_color
			brightest_color = color

		#for higher accuracy, also take into account peaks in the y direction and check if they align with expected (red: peak at top of image, yellow: middle, green: bottom). If not, check the second brightest.



	if brightest_color == "red":
		print("STOP")
		return "STOP"
	elif brightest_color=="amber":
		print("SLOW DOWN")
		return "SLOW DOWN"
	elif brightest_color=="green":
		print("GO")
		return "GO"




read_traffic_light('traffic-light.jpg')
read_traffic_light('traffic-light-amber.jpg')
read_traffic_light('traffic-light-green.jpg')
read_traffic_light('traffic-light-green2.jpg')
