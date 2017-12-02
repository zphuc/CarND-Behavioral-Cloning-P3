# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures/track1.png "Track1"
[image2]: ./pictures/track2.png "Track2"
[image3]: ./pictures/all_center_org.png "Center Image"
[image4]: ./pictures/all_left_org.png "Left Image"
[image5]: ./pictures/all_right_org.png "Right Image"
[image6]: ./pictures/pHis.png "Error loss Image"

---
## Target Tracks

This project works for the two tracks.

#### Track ONE
![alt text][image1]

#### Track TWO
![alt text][image2]

---
## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator, drive.py and my model.h5 file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
## Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

#### Model architecture
My Network has been finally designed based on [the NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with some modification.

  * The model consists of three 2D convolution layers with 5x5 filter sizes and depths of 24, 36 and 48, two convolution layers with 3x3 filter sizes and depths of 64 (model.py lines 115-119).
  * These convolution layers also includes the RELU activation to introduce nonlinearity.

#### Dataset
  * The image dataset is normalized in the model using a Keras lambda layer (model.py line 112).
  * A Cropping2D layer is used to remove the useless information at the top and bottom portion of images captures trees, hills, sky and the hood of the car, respectively (code line 113).

####  2. Attempts to reduce overfitting in the model

In order to reduce overfitting in the model,
  * The entire datasets were split into 80% for training datasets and 20% for validation datasets to ensure that the model was trained and validated on different datasets(model.py line 205).
  * The model contains four Dropout layers with different rates (model.py lines 120-127).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 225).

#### 4. Appropriate training data

Training dataset was chosen to keep the vehicle driving on the road.   
I used the steering angle, and images of the center, left and right cameras of following dataset for training and validation.
  * Dataset provided by Udacity.
  * Dataset driving forward of the track ONE (two claps)
  * Dataset driving forward of the track TWO (two claps)

For details about how I created the training data, see the next section.

---
## Model Architecture and Training Strategy in Details

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the [LeNet model](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb), which was introduced in the "More Networks" class. But, the car went out of road in a moment.

Therefore, I used the [Nvidia model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which was introduced as a powerful network in the "Even More Powerful Network" class. The car run more well, but went out of road later. Then, I recorded two laps on the track ONE to augment more dataset. After training, I found that the car could run well for the track ONE.  

I also try to run for the track TWO using the training result, but the car obviously went out of road in a moment.

I then recorded two laps on the track TWO. Using overall dataset (Udacity data, two claps of track ONE, two laps of track TWO), I did the training again. After training, I found that the car fell off for both track ONE and TWO in a few spots.

I thought that the model might be overfitting. To improve the driving behavior, I modified the model using the some Dropout layers. Herein, I also did the training with the following hyperparameters for tunning. See in detail and the final values in the below section (**Final Model Architecture and hyperparameters**).

  * The cropping region in Cropping2D layer to remove the useless information at the top and bottom portion of images.
  * The **correction** parameter for estimation of the steering angles of images of the left and right cameras.
  * The rates of the Dropout layers.

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road.

#### 2. Final Model Architecture and hyperparameters

  * The final model architecture (model.py lines 107-130) consisted of a convolution neural network with the following layers.

| Layer        	|  Description                                     |    Output    |
|:-------------:|:------------------------------------------------:|:------------:|
| Lambda        |  Normalization                                   |    3@160x320  |
| Cropping2D    |  Cropping region ((50,25),(0,0))                 |    3@85x320   |
| Convolution2D |  5x5 (filter: 24, strides: 2x2, activation: ELU) |   24@41x158   |
| Convolution2D |  5x5 (filter: 36, strides: 2x2, activation: ELU) |   36@19x77    |
| Convolution2D |  5x5 (filter: 48, strides: 2x2, activation: ELU) |   48@8x37     |
| Convolution2D |  3x3 (filter: 64, strides: 1x1, activation: ELU) |   64@3x35     |
| Convolution2D |  3x3 (filter: 64, strides: 1x1, activation: ELU) |   64@1x33     |
| Dropout       |  rate:0.5                                        |      -        |
| Flatten       |                                                  |    2112       |
| Dense         |  Fully connected: 100 neurons                    |    100        |
| Dropout       |  rate:0.25                                       |      -        |
| Dense         |  Fully connected:  50 neurons                    |     50        |
| Dropout       |  rate:0.25                                       |      -        |
| Dense         |  Fully connected:  10 neurons                    |     10        |
| Dropout       |  rate:0.25                                       |      -        |
| Dense         |  Fully connected:   1 neurons                    |      1        |


  * The final **correction** parameter is 0.2 (model.py lines 36), which is used for images of the left and right cameras

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first used the dataset provided by Udacity.

To augment the dataset, I recorded two laps on each track (the track ONE and track TWO).   

For each data, I used image of the center camera using for the center lane driving.  
Here is an example image of the center camera:

![alt text][image3]

I also used images of the side cameras. Herein, their steering angles were adjusted by the **correction** parameter using the method suggested in the "Using Multiple Cameras" class.  
Here is an example image of the left and right camera:

![alt text][image4]
![alt text][image5]

After the collection process, I had 68,652 number of data points (images, steering angles).  

I finally randomly shuffled the data set and put 20% of the data into a validation set.
I used this training data for training the model. The validation set helped determine if the model was over or under fitting.  

I used an adam optimizer so that manually training the learning rate wasn't necessary.

The number of epochs was 5. Here is "mean squared error loss" of the training:

![alt text][image6]

---
## Video results

Here are the videos recording of the vehicle driving in autonomous mode for the following tracks.

  * Track ONE : [track1.mp4](track1.mp4)
  * Track TWO : [track2.mp4](track2.mp4)
