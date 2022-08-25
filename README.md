# Linear Regression and Homography
Part of the assignment for ENPM 673 - Perception for Autonomous Robots
 
## Required Packages:
* OpenCV - version = "4.1.0"
* Numpy - version = "1.21.2"
* Matplotlib - version = "3.5.0"
* Random
 
## Instructions to Run the Code:
### Question 2:
use the command ```python Question2.py``` or ```python3 Question2.py```
 		
Make sure that the videos - ball_video1.mp4 and ball_video2.mp4	are in the same path from where the python file (Question2.py) is running.	
 		
### Question 3:
use the command ```python Question3.py``` or ```python3 Question3.py```
 		
Make sure that the file ENPM673_hw1_linear_regression_dataset-Sheet1.csv is in the same path from where the python file (Question3.py) is running.
 		
### Question 4:
Use the command ```python Question4.py``` or ```python3 Question4.py```

## Results:
### Standard Least Squares Curve Fitting
Perfect Data  |  Result |
:-------------------------:|:-------------------------:
<img src= "https://user-images.githubusercontent.com/40200916/186565699-5bc5fcf2-5893-4cef-bf0b-b5b601f5cab6.png"> | <img src="https://user-images.githubusercontent.com/40200916/186565735-65481a2b-dfbe-45dc-a642-edee44d48f5a.png" width="100%"> 

Noisy Data   |  Result |
:-------------------------:|:-------------------------:
<img src= "https://user-images.githubusercontent.com/40200916/186565857-c7f151c2-5e01-4397-ac97-a834fee2624b.png"> | <img src="https://user-images.githubusercontent.com/40200916/186565894-3f1c1dfe-2858-4084-936f-ae565d6cd4c4.png" width="100%"> 

### LS, TLS, RANSAC, Covariance
Least Squares  |  Total Least Squares | Covariance |
:-------------------------:|:-------------------------:|:-------------------------:
<img src= "https://user-images.githubusercontent.com/40200916/186566445-f28041fd-3e77-4e49-82c7-f00abe367dab.png"> | <img src="https://user-images.githubusercontent.com/40200916/186566488-7ea4e812-56bd-4fc1-b45e-30ea6a9d7b22.png"> | <img src="https://user-images.githubusercontent.com/40200916/186566551-e5cd40d0-8585-4303-9946-80d430772858.png" width="100%"> 

<p align="center">
  <img src="https://user-images.githubusercontent.com/40200916/186567257-7ab757a5-883f-49cb-a556-af57fa220186.png" width="50%">
</p>

## Note 
* The outputs are displayed in the terminal for this question. A, S, Vt, U and H matrices are printed on the terminal.
* The report for the Homework is in the report folder.
