
# Vehicle Detection Project

The goals / steps of this project are the following:

[//]: # (Image References)
[image1a]: ./examples/car_hog.png
[image1b]: ./examples/noncar_hog.png
[image2]: ./examples/hog_explore.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/search_sliding_windows.png
[image4a]: ./examples/heatmap_bounding_box.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---
### README

#### 1. Provide a README that includes all the rubric points and how you addressed each one.   

Here I will consider the rubric points individually and describe how I addressed each point in my implementation. All my code is in the jupyter notebook "P5.ipynb", and all code cell reference are from that notebook.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The first code cell in "P5" notebook has the function `get_hog_features()` to extract HOG features which uses `skimage.features.hog`.  There is a helper function `convert_color()` to change color space of the image.  The first code cell also reads the **some** of the file names of all the `vehicle` and `non-vehicle` images. 

The second cell has function `show_hog_features_colorspace()` that I used to explore different color spaces.

I grabbed random images from each of the two classes and displayed them to get a feel for what the  output looks like.  I eventually decided to use `YCrCb` color space. `YUV` also looks quite similar to `YCrCb`, `YUV` gave me a runtime error, and quite clustered HOG results.  The `H` channel of both `HSV` and `HLS` did not quite mark out the car or the non-car that well.  The exploration results depicted below bear these out:

**Here is an example color space exploration for a random car image:**
![alt text][image1a]

**Here is an example color space exploration for a random non-car image:**
![alt text][image1b]

#### 2. Explain how you settled on your final choice of HOG parameters.

For the other HOG parameters for `skimage.features.hog()` function, I tried various combinations of HOG parameters, as depicted below.  This code for this is in fourth code cell, helper function `show_hog_features()` is used for visualisation.  The first two columns show variations of `orientation` bins keeping values of `pix_per_cell` as 8 and `cell_per_block` of 2. Third column shows variation of `pix_per_cell` keeping `orientation` constant as 9, and `cell_per_block` as 2. Fourth column show variations of `cell_per_block` with other two parameters defaulted. Last column shows various combinations of the three parameters.  HOG visulaisation is for all the first channels of `YCrCb` color space.

![alt text][image2]

I also looked up the findings presented in the [HOG presentation](https://www.youtube.com/watch?v=7S5qXET179I) and [HOG paper](http://vc.cs.nthu.edu.tw/home/paper/codfiles/hkchiu/201205170946/Histograms%20of%20Oriented%20Gradients%20for%20Human%20Detection.pdf) by Naveen Dalal et al, which exlored HOG for finding humans in images.  The paper used a portrait rectangular block to scan the images for humans standing upright.  The vehicles don't have moving limbs.  The nearby vehicles would have ther side-view visible and appear more like a rectangle in landscape mode.  The vehicles further away will have their rear (or front for oncoming vehicles) view visible and will be more like squares.

- `orientation` bins: 9 as per findings in the presentation.  Given that cars don't have moving limbs, fewer bins would have been sufficient.  The presentaion mentioned performance decreasing for more than 9 bins.
- `pix_per_cell`: (8, 8): Decided to use square block which neatly dvides the training data image sizes of 64x64.  Bigger blocks lose the detail of image and smaller blocks don't generalise enough.
- `cells_per_block`: (2, 2): This corresponds to 25% overlap given block size of 8x8 for normalising features.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code in 5th cell has functions
- `bin_spatial()` for computing binned color features with down sampling.
- `color_hist()` for computing color histograms.
- `extract_features_img()` that extracts features from a single image, optionally calling color features functions.
- Finally `extract_features()` for extraction of features for a set of file names.

The code in 6th cell trains a linear SVM using `sklearn.svm.LinearSVC`. It first extracts features from all `vehicles` and `non-vehicles` image data supplied by Udacity for this project.  All three channels of `YCrCb` color space are used with spatial size of (32x32), and 32 color histogram bins and HOG parameters 8 pix_per_cell, 2 cell_per_block.  This results in feature vection of 8460 elements. Training and test data split of 90%:10% was used and it acheived a test acuuracy of 99.29%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Code in 7th cell implements the sliding window. Heler function `slide_window_1()` calculates the window bounds.  Function `slide_window()` subdivides area of interest to two regions for two scales. I decided to search window positions with two scales scaling with Y-axis search limited to 400 to 656 pixel range, with 75% overlap of window size of (64x64).

The two different ranges of areas of interest for the two scales is shown below.  The blue rectangles depict areas of 64x64 (further central regions) and the yellow rectangles show areas at double scale (128x128) window size (sideways and nearby regions), as per the perspective scaling expected to be seen from camera's view.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided following result on test images.  Code in the 10th cell has function `search_windows()` that performs the search using the sliding windows as described above.

![alt text][image4]
---



### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4).  Tht thick blue boxes are cars identified after filtering some false positives as described below.  There are thin cyan boxes drawn too for all the raw detections (to aid visual debugging).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video.

Code in 13th cell defines class `Vehicles` that is used to track detections over last N frames and uses the threshold over heapmap from all the prior N frames' detection to reduce false positives.

### Here are nine frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all nine frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initial difficulty in this assignment was on finguring out the correct features (hog/channels) to train the claissfier.  I took guidance from the material in the lessons and depicted results of various combinations.

Next set of difficulties were in getting the cars to be recognised, I eventually settled on 2 scales with 75% overlap in sliding windows.  I had tried 3 scales, but didn't get enough time to check if that will help improve detection.  I eventually limited the windows by limiting not only overall area of interest, but further limiting the search area for each of the two scales.  There were some false positives, and the run time per frame turned out to be very slow.

I used the Hog sub-sampling approach from the lesson to with scale of 1.5. It resulted in more false positives and there is also a region of image where it misses car detection.  Whilst the the filtering approach descibed above helps with false positives, the missed detection can be addressed by making use of previous detection, tracking relative motion of the car and projecting/predicting the region the car would be in next frame and resample that region.  One could use the lane detection from approach and further limit the region of interest and make the processing faster.


```python

```
