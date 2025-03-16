import numpy as np
import cv2
from skimage.feature import hog
from skimage.filters import sobel
from skimage.measure import moments_hu

# For updates here to take effect, the entire notebook needs to be reloaded.
# downside is that you will lose all stored data that has been
# processed. So the current work around that I am going to use for this is to
# only write core functions here that will be used everywhere else,
# and try to avoid writing code here that will need a lot of modifications to get working.


class ImageData:
    def __init__(self, dataArrays):
        self.dataImages = dataArrays['images']
        self.dataLabels = dataArrays['labels']
        self.features = (2159,)

        flattenedDataImages = self.dataImages.reshape(self.dataImages.shape[0], -1)
        self.dataImages = flattenedDataImages
        # we should have 70K images with an with and height of 28 pixels.
        # so array shape should be 70,000 entires with each entry bing 28x28 = 784
        ## so (70000,784) since we are processing a single image,
        ## we have an array of shape (1,784)
        print ("loaded dataset from file. Shape of Images data set:", self.dataImages.shape)


    def PreProcessImage(self, img):
        "Preprocess image: Normalize, remove noise, and binarize."
        img = img / 255.0  # Normalize pixel values to range [0,1] for consistency across models.
        img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur to reduce small noise artifacts. (3,3): K-size need to be tested. IMPORTANT!!! keep these numbers odd and grater than or equal to 0
        ret, img = cv2.threshold(img, 0.5, 1.0, cv2.THRESH_BINARY)  # Convert to binary format to simplify features. [2]
        # print(ret)
        return img



    # Extracts Histogram of Oriented Gradients from a given image.
    def Histogram_of_Oriented_Gradients(self, img):
        # Size in pixels for each cell block
        pixelsPerCell = (4,4)
        # Cells per block to look at
        cellsPerBlock = (2,2) 
        # Return the data as a vector
        featureVector = True

        # We are telling hog that each cell is 4 by 4 per cell,
        # and we want hog to normalize an area of 2 by 2 cell blocks.
        # This totals 4x2 by 4x2 :> 16 total pixels looked at to calculate the
        # vector. This helps reduce noise and "smooth" out the vector a little.

        # TODO:: look into this a little more as I feel that I am not 100% sure I fully understand
        ## this step.
        return hog(img, pixels_per_cell=pixelsPerCell, cells_per_block=cellsPerBlock, feature_vector=featureVector)



    def Sobel_Edge_Detection(self, img):
        # enhance edge contrast, making digit contours clearer for feature extraction
        return sobel(img).flatten()



    def Extract_features(self, img):
        # Use HOG to identify differences in digits based on stroke patterns.
        #TODO:: Look into this more for a better understanding, at the moment
        ## I am just doing this because it is recommended by the guide.
        hog_features = self.Histogram_of_Oriented_Gradients(img)

        # Edge detection using sobel filter
        edge_features = self.Sobel_Edge_Detection(img)

        # Split image into 4 by 4 regions and calculate pixel densities
        zones = []
        for i in range(0,28, 7):
            for j in range (0, 28, 7):
                zone = img[i:i+7, j:j+7]
                zones.append(np.sum(zone))
        
        # Projection Features - compute horizontal and vertical projections
        # helps identify stroke distributions, such as differentiating between narrow nad wide digits.
        hor_project = np.sum(img, axis = 1)
        ver_project = np.sum(img, axis = 0)

        # we can help better distinguish between digits with similar structures such as 3 and 8 by using Hu moments.
        # and help eliminate distortion created by the image being rotated, scaled, or flipped.
        hu_moments = moments_hu(img)

        # combine multiple features extractions techniques to improve classification performance at the expense of speed.
        # concatenates all 1D extracted components vectors into a 1D array.
        features = np.hstack([hog_features, edge_features, zones, hor_project, ver_project, hu_moments])
        return features