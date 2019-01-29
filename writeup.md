# Advanced Lane Finding Project

author liangtaohy@gmail.com

[//]: # (Image References)

[image1]: ./output_images/calibration2.png "Calibration2"
[image2]: ./output_images/thresholded_binary.png "Thresholded Binary"
[image3]: ./output_images/warped_image.png "Warped Image"
[image4]: ./output_images/histogram.png "Histogram"
[image5]: ./output_images/warped_image_lane_lines.png "Lane line pixels"
[image6]: ./output_images/unwarp_boundaries.png "Unwarp Boundaries"
[image7]: ./output_images/lane_line_pipeline.png "Final Pipeline"

---

## Goal

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Camera Calibration

A function estimates the intrinsic camera parameters and extrinsic parameters for each of the views.
An object with a known geometry and easily detectable feature points is called a calibration rig or calibration pattern. OpenCV has built-in support for a chessboard as a calibration rig.

For more detail, see [camera_calibration](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)


We use chessboard images to estimates the intrinsic camera parameters.

```
def camera_calibration(corners=(9,6), camera_cal_dir='camera_cal'):
    """
    load camera chessboard images and calibrate camera, get mtx and dst parameters for camera calibration
    """
    nx = corners[0]  # corners on x axis
    ny = corners[1]  # corners on y axis

    # prepare object points
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(camera_cal_dir, '*.jpg'))
    imgs = []
    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            #corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners,ret)
            imgs.append(img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs, imgs
```

### Undistortion Example

After mtx and dist are estimated, we can use them to undistort an example image.

```
img = mpimg.imread('camera_cal/calibration2.jpg')
undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
draw_images(img, undistorted_img, title="undistorted image")
```

![alt text][image1]

---


## To create a thresholded binary image

I use color space and gradient methods learned from the lesson to create a threholded binary image.

* 1) change color space into gray
* 2) get Sobel Gradient on aixs X. Thresh is (20, 200).
* 3) compute Direction Gradient. Thresh is \[30, 90\].
* 4) RGB selection for the yellow line.
* 5) hls select. `s` thresh is (100, 255). `l` thresh is (120, 255).
* 6) apply region of interest select

For function, please see [AdvanceLaneLine]('AdvanceLaneLine.ipynb') function `get_thresholded_binary`.

### Thresholded Binary Image Example

```
image = mpimg.imread("test_images/straight_lines1.jpg")

binary_output = get_thresholded_binary(image)
draw_images(image, binary_output, title="Combined Binary Image", cmap='gray')
```

![alt text][image2]

---

## Perspective Transform

I select source points mannually. The dst points should be keep line as straight. Is there a good way to find source points? I just try them again and again.

The following is the code snippet.

```
src_points = ...
dst_points = ...

def unwarp_image(image, src_points, dst_points):
    img_size = (image.shape[1], image.shape[0])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    p_img = cv2.warpPerspective(image, M, dsize=img_size)
    return p_img
```

### Example

![alt text][image3]

---

## Find Lane Line Pixels And Fit The Line

The most methods used to find lane line pixels are histogram, sliding window search (brute search), polynomial fit.

### Histogram

Through the following image, we can clearly see the two lane line areas. The peak of the left and right halves of the histogram will be the starting point for the left and right lane lines.

```
def histogram(img):
    img = img/255
    bottom_half = img[img.shape[0]//2:,:]
    h = np.sum(bottom_half, axis=0)
    return h

hist = histogram(warped_image)
plt.plot(hist)
```

![alt text][image4]

### Sliding Window Search

It's a brute search method.

* 1) Take a histogram of the bottom half of the image
* 2) Find the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines
* 3) HYPERPARAMETERS
    * Choose the number of sliding windows (10)
    * Set the width of the windows +/- margin (100)
    * Set minimum number of pixels found to recenter window (100)
    * Set height of windows - based on nwindows above and image shape ((image height) / (number windows))
    * Step through the windows one by one

For code detail, please see [Sliding-Window-Search]('AdvanceLaneLine.ipynb#Sliding-Window-Search')

![alt text][image5]

---

## Determine the curvature of the lane and vehicle position with respect to center

To determine the curvature of a line, 2 degree polynomial fit is a choice for us.

```
# Implement the calculation of R_curve (radius of curvature)
line_curverad = ((1 + (2*plotyfit[0]*np.max(ploty) + plotyfit[1])**2)**1.5) / np.absolute(2*plotyfit[0])
```

For code detail, please see [Curvature]('AdvanceLaneLine.ipynb#Determine-the-curvature-of-the-lane-and-vehicle-position-with-respect-to-center').

---

## Warp the detected lane boundaries back onto the original image

```
out_img = np.dstack((warped_image, warped_image, warped_image))*255
y_points = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
left_line_window = np.array(np.transpose(np.vstack([h_l_x, ploty])))
right_line_window = np.array(np.flipud(np.transpose(np.vstack([h_r_x, ploty]))))
line_points = np.vstack((left_line_window, right_line_window))
cv2.fillPoly(out_img, np.int_([line_points]), [0,255, 0])
M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
img_size = (image.shape[1], image.shape[0])
unwarped = cv2.warpPerspective(out_img, M_inv, img_size, flags=cv2.INTER_LINEAR)
result = cv2.addWeighted(image, 1, unwarped, 0.3, 0)
```

![alt text][image6]

## Final Pipeline

* 1) Get thresholded binary image
* 2) Get Perspective view image
    * perspective transform matrix and inverse matrix
* 3) find lane line pixels through sliding window search
* 4) prediction left/right lane line x
* 5) Determine the curvature of the lane
* 6) Warp the detected lane boundaries back onto the original image

![alt text][image7]

## Test On Video

* [project_video_output.mp4](https://github.com/liangtaohy/CarND-Advanced-Lane-Lines/tree/master/project_video_output.mp4)