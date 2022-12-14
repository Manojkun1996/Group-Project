# Group-Project

An experiment is performed with Baxter robot and the actions of the robot is recorded and a huge dataset is generated. The videos contained in the dataset have already been labeled, that is each video has been analysed and have inserted manual labels (in the image metadata) indicating beginning and end of actions performed by the participants. 

# Aim of the project

1. Writing a script which extracts the timestamps corresponding to each action, according to the action label. 
2. To synchronize the action timestamps with the timestamps of the accelerometer data, obtaining a precise correspondence between each action and its resulting            acceleration pattern. 
3. To Find all the actions in the metadata of Images and extracting the information and finding its start and end labels of its data and synchronize it with along the    required sensory output.

## Information about the dataset

Regarding the files, you will see that each folder is structured as follows:
- a /frames sub-folder, containing the sequence of images which make up the video of the experiment
- a timestamp file (in txt format), specifiying the timestamp of each frame contained in the aforementioned subfolder.
- 4 txt files containing the data of wearable accelerometers worn by the person during the experiment. In particular, each person was wearing one accelerometer on the back of the hand and another one on the wrist, both for the left and right arm. The data in the txt file also contain the timestamps, specifying the instant in which each sample was acquired.

## Extraction of Metadata

Download **exiftool.exe** from google.
For more information regarding installation and usage follow the tutorial : https://exiftool.org/

Save the output of the metadata as **out.csv**

## Program to extract the timestamps corresponding to each action, according to the action label and To synchronize the action timestamps with the timestamps of the accelerometer data, obtaining a precise correspondence between each action and its resulting acceleration pattern
Clone the repo 

Navigate to Group-Project/datasets/s01/

Launch the python file **Meta_data.py** 

Enter the file location in the script
example : FILE_LOCATION = r"G:\datasets\s01\out.csv"

RUN the program.

We get the synchronised output as shown below files.

my_data_left_backPose.csv,
my_data_left_wristPose.csv,
my_data_right_backPose.csv,
my_data_right_wristPose.csv

## output Video

https://user-images.githubusercontent.com/88244126/197787644-902773e6-747e-4a82-b411-e84c78dba4d6.mp4

## Program to get the information regarding each particular actions

Navigate to Group-Project/datasets/s01/

Launch the python file **Meta_data_actions.py** 

Enter the file location in the script
example : FILE_LOCATION = r"G:\datasets\s01\out.csv"

RUN the program.

We get the required output of all the respected actions individually, which u can see inside the folders : 
data_left_backPose,
data_left_wristPose,
data_right_backPose,
data_right_wristPose 

Location on the repository :
Group-Project/datasets/s01/data_left_backPose/

Group-Project/datasets/s01/data_left_wristPose/

Group-Project/datasets/s01/data_right_backPose/

Group-Project/datasets/s01/data_right_wristPose/


The output video is shown below

https://user-images.githubusercontent.com/88244126/202874571-c8271798-b64d-4490-9c35-89fcb66ae6ec.mp4
