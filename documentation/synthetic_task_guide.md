
# Synthetic task guide

All self-supervised tasks must extend and implement SelfSupTask, defined in
[nnood/self_supervised_task/self_sup_task.py](../nnood/self_supervised_task/self_sup_task.py) as having an `apply`, `loss`,  and `calibrate` function (details in docstrings of
methods).

To aid in experimenting with patch-blending tasks we have decomposed the existing tasks, making it is easier to swap 
out individual parts of the task to understand which are contributing the most to performance.

The classes of component are:
 - `PatchShapeMaker` - returns a mask which is used to extract the source patch
 - `PatchTransforms` - applies a transform to the extracted patch, which can be either spatial or altering the content
  of the patch. A list of these is used to define the task.
 - `PatchBlender` - integrates the source patch into the target image at the given location.
 - `PatchLabeller` - given the original image and the altered image, compute the pixel-wise label for the image.

These components are combined using `nnood/self_supervised_task/patch_ex.patch_ex`, which also has a number of other
parameters such as the number of anomalies introduced (see the docstring for full details).

Here are some example tasks reimplemented using this framework:
 - [Foreign Patch Interpolation](../nnood/self_supervised_task/fpi.py)
 - [CutPaste](../nnood/self_supervised_task/cutpaste.py)
 - [Poisson Image Interpolation](../nnood/self_supervised_task/pii.py)
 - [Natural Synthetic Anomalies](../nnood/self_supervised_task/nsa.py) (both source and mixed gradient variants).
