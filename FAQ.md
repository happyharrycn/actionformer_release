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

**A:**  In THUMOS14 dataset, the action category "Cliff Diving" (ID: 4) is a subset of the category "Diving" (ID: 7). Namely, all "Cliff Diving" instances are also annotated as "Diving", yet not all "Diving" instances are marked as "Cliff Diving". This brings an issue for the implementation of label assignment during training. Specifically, our current implementation assumes that a moment can be matched to at most one ground truth (GT) action instance, and thus could not handle the case where there are two GT action instances with exactly the same temporal boundaries (yet different labels).

Our solution, somewhat hacky, is to remove all instances from the action category "Cliff Diving" during training by masking out those instances. Note that (1) all instances from "Diving" still remain; and (2) this is only done during training. At test time, our model is still evaluated on all categories. The downside of this solution is that our model will not be able to recognize "Cliff Diving" on the test set, as it never saw an training sample from this category. Part of this discrepancy is resolved by fusing external classification scores. To verify this, you can disable score fusion and print the mAP scores per category, and the mAP for "Cliff Diving" (ID: 4) should be close to 0.
##
