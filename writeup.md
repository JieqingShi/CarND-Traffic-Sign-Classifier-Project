# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[example-signs]: ./images-for-writeup/example-signs.jpg "example-signs"
[hist-signs]: ./images-for-writeup/hist-signs.jpg "hist-signs"
[bumpy_road]: ./examples/downloaded_signs/bumpy_road.jpg
[end_of_all_limits]: ./examples/downloaded_signs/end_of_all_limits.jpg
[no_entry]: ./examples/downloaded_signs/no_entry.jpeg
[no_vehicles]: ./examples/downloaded_signs/no_vehicles.jpg
[priority_road]: ./examples/downloaded_signs/priority_road.jpeg
[right_of_way_until_next_intersection]: ./examples/downloaded_signs/right_of_way_until_next_intersection.png
[road_narrows_to_the_right]: ./examples/downloaded_signs/road_narrows_to_the_right.jpg
[speed_limit_120]: ./examples/downloaded_signs/speed_limit_120.jpg
[speed_limit_20]: ./examples/downloaded_signs/speed_limit_20.jpg
[stop]: ./examples/downloaded_signs/stop.jpg
[straight_or_left]: ./examples/downloaded_signs/straight_or_left.jpg
[misclassified]: ./examples/misclassified.png
[top5]: ./examples/top5.png

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

I looked at the dataset in terms of:
- how many examples per class there are in the training set
- how the examples look like

**Histogram of number of samples per class**

The number of samples per class can be obtained from the `Counter` class from the `collections` package. (or `groupby` and `count` from `pandas` )
Then I plotted a barplot using `seaborn`. 

![Histogram][hist-signs]

We can see that the dataset is a bit unbalanced. For some classes there are almost 10x as many images than for other classes.
This could pose a potential problem if images from the underrepresented classes are misclassified more frequently because the model was not exposed to enough examples.

**Visualization of sample signs**

First I obtained the indices of images in the training dataset for each class and summarized that in a pandas dataframe. Then I iterate through each class, grab a random index
from that class and plot together with the name of the sign.

![Example-Signs][example-signs]

We can see some images with bad contrast where it's really difficult to see anything due to the darkness. 
As a possible preprocessing step one could pick out those dark images and work with the contrast to make it lighter. Or one could simply drop those images

**More things to do for data exploration**
Here are some ideas for further data exploration
- look at the distribution of number of samples per class in the validation and test set and compare them to training set (maybe there are a lot of pictures of one class in the test set which are underrepresented in the training set?)
- inspect images of validation and test set and see if they're similar to training set (as in: they come from a similar distribution - maybe training set consists mostly of clear images while the test set is dominated by dark pictures?)
- clean possibly corrupted samples (corrupted = e.g. very dark/very bright, traffic signs from other countries, no traffic signs at all)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

It makes sense to start with a baseline model to see which data processing techniques to apply for further improvement.

As a first step, I decided to use the original LeNet which I adapted from the lecture. I used all three color channels and I normalized the images using the suggestion from the notebook (`(pixel-128)/128`). The only modifications I made to the LeNet architecture was to change the number of input channels of my convolutional layers (from 1 to 3) and to change the number of output classes (from 10 to 43). 


The training and validation accuracy of that model was surprisingly low, ~80% and ~73%, respectively.


I also tried the min-max scaling suggested in the lecture and I normalized the images to a [0.1, 0.9] range. This had a minimal effect on the accuracy.

I deduced from this that I was underfitting which is why I did not use any image augmentation techniques to create more training data (as more data does not help underfitting). Instead I decided to increase the complexity of the model. (see next sections)

I also experimented with converting the pictures to grayscale, which is also the approach described in the original LeNet paper (intuitively it also makes sense because the color of a traffic sign may not be its most characteristic attribute). However contrary to the paper results, I was unable to observe any improvements by using grayscale images, which is why I stuck to the original three color-channels.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution     	    | 5x5x12, stride = 1, valid padding 	        |
| RELU					|												|
| Max pooling	      	| 2x2x12 stride = 2, same padding				|
| Convolution    	    | 5x5x24, stride = 1, valid padding				|
| RELU					|												|
| Max pooling	      	| 2x2x24 stride = 2, same padding               |
| Fully connected		| 600x120      									|
| RELU					|												|
| Dropout			    | keep_prob=0.6									|
| Fully connected		| 120x84      									|
| RELU					|												|
| Dropout			    | keep_prob=0.6									|
| Softmax			    | 84x43								|

As to why I chose this architecture, see sections below. 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model using the following parameters

| Parameter         	|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Number of epochs         		| 10 							| 
| Batch size | 128 |
| Learning rate | 0.001 | 
| Optimizer | Adam |

The weights were initialized using tensorflows `truncated_normal()` with a mean of 0 and a std of 1
The Adam parameters for beta1, beta2 etc. were left at their default values. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The following changes to the original LeNet architecture brought me the most significant improvements:

- increasing the feature map size of both convolutional layers to 12 and 24, respectively
- normalize images to a [0,1] range simply by dividing the pixel values by 255

which got me to ~93% on the validation set. However I saw that I was now overfitting since my training set accuracy was ~99.9%, which is why I added dropout layers to both fully connected layers with a dropout rate of `p=0.4` （i.e. `keep_prob`=0.6 in tensorflow). I turn off dropout in evaluation mode by setting a `keep_prob` of 1.0.

My final model results were:
* training set accuracy of 99.4%
* validation set accuracy of 96.2% 
* test set accuracy of 94.3%

I also printed the confusion matrix of the test set. (see notebook/html export) It did not seem that the underrepresented classes in the training set
were particularly frequently misclassified which is positive.


The validation and test set accuracy are quite similar which shows that the model works. However it still shows signs of overfitting (training set accuracy very high). This could be mitigiated by some regularization techniques like weight decay (l1/l2 regularization), higher dropout rates, batch-normalization, early-stopping etc.


Although looking at the metric to diagnose over-/underfitting is good enough, the technically correct way would be to look at the training and validation losses (instead of the accuracy). This is because an improvement in the loss does not necessarily imply an improvement in the metric (sometimes an increase in loss from one epoch to the next is accompanied by a increase in accuracy). 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I downloaded 11 images of german traffic signs. After that, I cropped them by manually using a bounding box to center out the actual traffic sign in the images.


![][bumpy_road] ![][end_of_all_limits] ![][no_entry]
![][no_vehicles] ![][priority_road] ![][right_of_way_until_next_intersection]
![][road_narrows_to_the_right] ![][speed_limit_120] ![][speed_limit_20] 
![][stop] ![][straight_or_left]

6 of them occur quite often in the training dataset. (*Priority road, No vehicles, Stop, No entry, Speed limit 120, Right-of-way at next intersection*)



I also chose an additional 5 images specifically because they weren't that represented in the training set, because I wanted to see how the model performs on these rather rare images. (*Bumpy road, Speed limit 20, Go straight or left, Road narrows on the right, End of all speed and passing limits*)


Afterwards the pipeline to get the pictures into the correct format is very simple:
    
    - shape the image to 32x32
    - then normalize it to [0,1] using the quick_normalize() function

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

My model was able to classify 9 out of these 11 pictures which gives an accuracy of 81.8%. These two images were incorrectly classified

![][misclassified]

| True Label			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road narrows to the right      		| Priority road   									| 
| Speed limit 120    			| Speed limit 30										|

The accuracy is lower than the test set, maybe because the hand-picked images from Google images have a different distribution than that of the original dataset. However since we're only using 11 images for testing it's impossible to tell. Also it completely depends on which images are chosen for testing. When I choose perfectly centered, front-facing, high-resolution stock images of traffic signs I should expect different results than when I choose images which are shot with a low-res camera at night.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![][top5]

I created a table out of the true label of the eleven images + the top 5 predicted labels and their corresponding probabilities.
The table indicates that the model is quite confident in its predictions (almost all top-1 probabilities at 99%)
For the two predictions it got wrong, the model is also very confident. Furthermore the actual real label is not within the top 5 predictions.


As to why the model got these wrong I can only speculate (first picture is taken at an extreme angle, second picture is slightly cropped and not centered?)



### Future improvements
The following things can be done for further improvement.

- collect additional images of underrepresented classes or use image augmentation techniques (random rotation, changes in lighting, zooming etc.) in order to get more balanced classes
- counter overfitting through regularization techniques (weight decay aka l1/l2 regularization, more dropout, early stopping, batch-norm etc.)
- look at training and validation losses instead of the metric (loss may increase although accuracy increases!)

I don't think that using transfer learning with a model like Resnet-50 etc. would help with the generalization capabilities. Since I'm already overfitting a little bit, using an even more complex model would not help in this case. Instead the overfitting should be mitigated through regularization.
