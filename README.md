# UNO-Card-Recognition
### The project's goal is to recognize cards from the game UNO. The program takes input from a file or a camera and identify cards in an image. The project includes creating the image dataset on which the recognition procedure will be performed. The dataset folder must be imported into the Jupyter notebook. 

##### The following libraries need to be imported. 

•	glob
•	cv2 
•	os
• argparse
•	matplotlib.pyplot 
•	numpy as np
•	random

# Step A: UNO Number Detection

In this section the first step was to load the dataset, then crop out the card, followed by normalizing the image size and splitting the dataset to training 80% and testing 20%

The next step consists of detecting the number using the training data: - 
-The first step was to convert the images to grayscale and do edge detection.
-The next step was to carry out closing.  Closing is dilation followed by erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.
- The dataset is then shown (black image with yellow boarders around edges)
- The next step is to find external contours in an image, and then fill them to create a silhouette image. 
-  In the next step the numbers in the centre of the image are cropped out. 
- The next step is to carry out local histogram. 


Test against testing data 
-	The image is converted to grayscale, followed by edge detection. The contours are then found
A silhouette image is created and finally the number is the centre is cropped out. 





# Step B: UNO Colour Detection

 Load dataset 
-	Load the images from DATASET/* (folder where the dataset is stored).
-	The libraries were imported 
-	Analyse the filename to extract the label for each image.
-	The dataset was loaded 
-	 Parse image filename for label
-	The data set is then shown with the name of the card (e.g. [‘G’, ‘draw’])

Crop out card
-	The region of interest (ROI) for each image is found 
-	Adaptive threshold is carried out to find edges 
-	 Apply a gaussian blur to make sure it doesn't pick up noisy edges.
-	Dilatation us applied to close any contour and when cropping. When finding the contours, we get two contours around the card. We pick the most inner one. The dilatation process will make sure the inner contour is exclusively inside the card.
-The biggest external contour area is found 
-The oriented bounding box around card is found
-The card width and height are estimated. 
-If else statement is placed if the card is straight or on its side 
-The data set is shown.

 Split the dataset 
-	80 % of the dataset is used for training and 20% is used for testing. 

Colour detection on training data
-The training images are converted to HSV 
-The pixels are counted. Each range corresponds to a colour (e.g. ['B', 'skip'] [562   5   2 23]).

Test against testing data
-From the previous results of the pixels of each image, a hue channel is picked. 
- For example, for channel 0, we get Colour testing: GOOD 11/BAD 0/TOTAL 11



# Step C: UNO Combination of Colour and Number

-The dataset is loaded 

-Find _Cards 

-An image is picked.

-A Gaussian blur is applied, followed by an adaptive threshold to find the edges.

-Dilation is applied to the image to close any contours. 

- Estimate the card width and height 

- Compute perspectives transform matrix and get wrapped image which is only the card. 

-Find _Colour

-Get _Number

-Find card for each image 

-Normalize image size 

-Train the colour and number 

-Test with data




# Step D: CW2_Final (Final-Combined-Webcam)

The previous stages are repeated in this stage. 
For the camera to detect the cards quickly and precisely, the setup background must be the same (the cards were placed on a black background).

When utilizing the camera, the command cap = cv. VideoCapture(0). Whereby selecting 1 this will launch the webcam, while 0 will launch the connected camera (phone).

For improved detection, the camera has been inverted/flipped.



# Step E: Webcam + DATASET - Python Script

>This script consists of the following arguments.

•	'--images', help="Input dataset"

•	'--camdevice',  help="Camera number"

•	'--mode', choices= ['livecam', 'staticimg'], default='livecam', help="Mode"

•	'--input', help="Input image for detection"

> These arguments will allow the user to choose which image source to use (webcam "livecam" OR DATASET "staticimg")

>> All image names in the dataset must be spelled the same 



### To run the python script using static images (DATASET):

python Final-cam-data.py --mode staticimg --images DATASET --input DATASET/nine_R.jpg


### To run the python script using the live cam (Webcam): 

python Final-cam-data.py --mode livecam --camdevice 1 --images DATASET


