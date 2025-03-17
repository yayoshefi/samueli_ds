# Part II:
    - Goal: Describe 2 approaches to find the L3 from CT scan

## General (assumptions)
I'm not sure how does dcm files look like.  
I Imagine it as a series of images (slices) of the body aligned by the vertical axis (each slice is a Horizontal plane)  
- From what I know about CT, it's actually a 3d volume described by voxel representation, so in practice there is a way to rotate the scan to the required axis.  

![human axis](./images/human_axis.png "human axis")
<!-- {width: "200px" height:"150px"} -->

As stated in the question, each dcm file is an axial slice of the body.
Answers are given under the assumption that we actually get multiple horizontal planes. by taking into account the full voxel representation and other slices, I also gave some variant to each approach - using other planes or 3D representation.



## Approach A: Classical
Identify bones as a binary mask per image $=>$  aggregate slices to create vertebrae $=>$ Identify known point in our axes (*anchor*) $=>$ Find L3 $=>$ return median slice for L3.

- Create a binary mask of the bones in the slice using a combination of edge detector (e.g. `sobel filter` ) and morphology operations on binary image. the vertebrae should be the largest bone in the images, which can be identified using `cv2.findCountours` to the largest "blob"
- Deciding 2 adjacent slices actually forms the same vertebrae, can be done by measuring IoU between them
  - IoU should be small in case of bones imperfections or spine angles
  - for the nominal case the slices should be aligned and approximately the same size
  - THe intervertebral  disc should create area of discontinuity effectively separating between the vertebras
- To create an "*anchor*" to our spine I used the Sacrum as it's the largest section and has unique shape the can be identified
  - assuming we know where the scan has started we can also use this as a prior knowledge
- To find the specific L3, we can count from our know point on our axis and count up.
    - Count 3 vertebras up from the Sacrum
    - For the middle section take half point from the start and end of the vertebrae
#### Variant
    In case we are not bounded to vertical slice (dcm files)
    We can identify L3 by counting vertebrae front to the spinal coord from a side view,
    we can find the Z-plane and choose the correct slice to return.
![side view](./images/spice_sagital.png "side view")
    


## Approach B: Data Driven
Use machine learning model to identify the required vertebrae. This approach is useful when we think there is enough data in each slice to actually know what vertebrae it is without context.

- For the simplest option (yet not sure it will work) classify each slice for a single class: L3 or not 
- This assumes that L3 slices are identified with no context of other joints (need to ask domain experts)
- Train the data to create the classifier model.
- In inference time, run the classifier on each dcm file.
- Take the largest streak of L3 classified and the the median slice as output. We can run a smoothing filter to avoid outliers.

#### Variant
1. If we are not bounded to horizontal planes (dcm files slice by slices), we can use a frontal view and annotate all vertebrates on the same image, using a multi labels (heat maps or detection), this method should be easier as we see more vertebras in our receptive field and actually learn more from the multi labels.
2. Use 3D architectures (use voxels instead pixels) - this can work similar to pose estimation networks where we actually predict different heatmaps for each vertebrae, and preform NMS to find the exact coordinate of the center of mass of the object.



# Reference Images
![Frontal Spine](./images/spine_frontal_plane.jpg "Frontal plane")
![all planes](./images/spine_all_planes.jpg "all planes")
![Horizontal](./images/vertebra_horizontal_plane.jpg "Horizontal plane")