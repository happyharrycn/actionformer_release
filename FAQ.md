## FAQ

The following questions are taken from GitHub issues. We thank our users for raising these good questions.
##

**Q:** Errors related to nms_1d_cpu

**A:** Please refer to the second part of INSTALL.md, and make sure that the C++ implementation of NMS is properly compiled.
##

**Q:** How to convert the absolute time in seconds (in the annotations) into the temporal feature grids (used by the model)?

**A:** The conversion uses the following equation.

*feature_grid = (timestamp * FPS - 0.5 * window_size) / feature_stride*

Here we assume that each video clip feature is extracted from a temporal window (oftentimes with even number of frames) and the feature grids lie in center of these local windows.

To illustrate this, it is probably easier to consider the following example, assuming FPS=1, sequence length=6, a local window size of 4 frames and a stride of 2 frames for feature extraction. This video lasts for 6 seconds and has 6 frames with the following layout. Just like pixels in an image, those frames are center aligned in a video. For example, the 0th frame starts at time step 0.0 and ends at 1.0, centered at time step 0.5.
<pre>
Time:              |0---1---2---3---4---5---6|
Frame index:       |--0---1---2---3---4---5--|
</pre>

With a local window size=4 and a feature stride of 2, there will be 2 clip features from this video.
<pre>
Time:              |0---1---2---3---4---5---6|
Frame index:       |--0---1---2---3---4---5--|
Local window 1:    |--0---1---2---3--|
Local window 2:            |--2---3---4---5--|
Feature grids:     |--------0-------1--------|
</pre>

Namely, 0.0 and 1.0 on the feature grids correspond to frame index 1.5 and 3.5, and time step 2.0 and 4.0, respectively.  Now if we look at a time step e.g., 0.0, it lies on -1.0 (=(0.0*1 - 0.5 * 4) / 2) of the feature grids.
##

**Q:** Why masking out action instances with label ID 4 when training on THUMOS14?

**A:** This is no longer needed in the code.
##

**Q:** Why some of the videos in the training / validation set of ActivityNet are not considered?
**A:** The original release of ActivityNet contains links to youtube videos. Over the time, some of the videos are no longer available and thus their pre-extracted features might not exist. Videos on ActivityNet are filtered out if (1) they don’t have video features (or the length of video features does not match the video duration); or (2) they don’t have any actions that last longer than 0.01 seconds (this is only done for training set).

##

**Q:** How to use I3D models to extract video features?

**A:** We refer to [this repo](https://github.com/Finspire13/pytorch-i3d-feature-extraction) for extracting video features using I3D models.

##
