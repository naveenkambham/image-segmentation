# image-segmentation
Image segmentation is the process of partitioning a digital image into multiple segments (sets of pixels, also known as image objects). The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze. Image segmentation is typically used to locate objects and boundaries (lines, curves, etc.) in images. More precisely, image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain characteristics.

## Edge Detection
Edge detection is useful in identifying the boundaries of objects within images. It works by detecting discontinuities in brightness. Various weight matrices such as Sobel and Laplace can be very useful in identifying the edges of objects.
In: 

![alt text](https://github.com/naveenkambham/image-segmentation/blob/master/saskatoon.png)

Out: 
![alt text](https://github.com/naveenkambham/image-segmentation/blob/master/saskatoon_laplace_edges_detection.png)


## Thresholding
The simplest method of image segmentation is called the thresholding method. This method is based on a clip-level (or a threshold value) to turn a gray-scale image into a binary image. The key of this method is to select the threshold value (or values when multiple-levels are selected). 

In:
![alt text](https://github.com/naveenkambham/image-segmentation/blob/master/mining.jpg)
Out:
![alt text](https://github.com/naveenkambham/image-segmentation/blob/master/mine_image_output.png)
