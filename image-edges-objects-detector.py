from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import KMeans

def overview_image(file):
    # to get a glance of the image

    #reading and converting to gray 
    img = plt.imread(file)
    print(img.shape)
    
    # converting to gray image
    gray_img = rgb2gray(img)
    plt.imshow(gray_img,cmap='gray')
    # plt.imshow(img)
    plt.show()
    return gray_img

def apply_img_threshold(img):
    # Applying Threshold Segmentation
    img_reshaped = img.reshape(img.shape[0]*img.shape[1])
    print(img_reshaped.shape)
    
    for i in range(img_reshaped.shape[0]):
        if img_reshaped[i] > img_reshaped.mean():
            img_reshaped[i] = 3
        
        elif img_reshaped[i] > 0.5 :
             img_reshaped[i] = 2

        elif img_reshaped[i] > 0.25 :
             img_reshaped[i] = 1
        
        else:
             img_reshaped[i] = 0


    img_reshaped = img_reshaped.reshape(img.shape[0],img.shape[1])
    plt.imshow(img_reshaped,cmap='gray')
    plt.show()

def detect_edges(img):
    #method to detect the edges of the objects

    #applying sobel horizontal and vertical filters
    filter_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
    filter_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
    
    # laplace filter to identify both vertical and horizontal edges
    filter_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
    
    out_horizontal_edges_image = ndimage.convolve(img,filter_horizontal,mode='reflect')
    out_vertical_edges_image =ndimage.convolve(img,filter_vertical,mode='reflect')
    out_img_edges = ndimage.convolve(img,filter_laplace,mode='reflect')
    plt.imshow(out_horizontal_edges_image,cmap='gray')
    plt.show()

    plt.imshow(out_vertical_edges_image,cmap='gray')
    plt.show()

    plt.imshow(out_img_edges,cmap='gray')
    plt.show()

def segment_by_clustering(img_file):
    # detecting the objects based on similar pixel values
    img = plt.imread(img_file)/255 
    img_reshaped = img.reshape(img.shape[0]*img.shape[1],3)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(img_reshaped)
    out_img = kmeans.cluster_centers_[kmeans.labels_]
    labeled_img = out_img.reshape(img.shape[0], img.shape[1], img.shape[2])
    plt.imshow(labeled_img)
    plt.show()

# Use Case 1: Identifying different objects inside an image
img = overview_image(r'C:\codebase\image-segmentation\mining.jpg')
apply_img_threshold(img)

# Use Case 2: Detecting the edges
img = overview_image(r'C:\codebase\image-segmentation\saskatoon.png')
detect_edges(img)

# Use Case 3: Image Segmentation by Cluster Analysis
segment_by_clustering(r'C:\codebase\image-segmentation\mining.jpg')