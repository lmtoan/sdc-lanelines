def draw_square(img, vertices):
	ysize = img.shape[0]
	xsize = img.shape[1]
	region = np.copy(img)

	left_bottom = vertices[0]
	right_bottom = vertices[1]
	left_top = vertices[2]
	right_top = vertices[3]

	# Fit lines (y=Ax+B) to identify the  3 sided region of interest
	# np.polyfit() returns the coefficients [A, B] of the fit
	fit_left = np.polyfit((left_bottom[0], left_top[0]), (left_bottom[1], left_top[1]), 1)
	fit_right = np.polyfit((right_bottom[0], right_top[0]), (right_bottom[1], right_top[1]), 1)
	fit_top = np.polyfit((left_top[0], right_top[0]), (left_top[1], right_top[1]), 1)
	fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

	# Find the region inside the lines
	XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
	region_thresholds = ((YY > (XX*fit_left[0] + fit_left[1])) & \
	                    (YY > (XX*fit_right[0] + fit_right[1])) & \
	                    (YY < (XX*fit_bottom[0] + fit_bottom[1])) & \
	                    (YY > (XX*fit_top[0] + fit_top[1]))

	region[region_thresholds] = [255, 0, 0]
	plt.figure()
	plt.imshow(region_select)
	plt.show()