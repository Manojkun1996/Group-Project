# Group-Project

An experiment is performed with Baxter robot and the movement of the robot is recorded and a huge dataset is generated. The videos contained in the dataset have already been labeled, that is each video has been analysed and have inserted manual labels (in the image metadata) indicating beginning and end of actions performed by the participants. 

# Aim of the project

1. Writing a script which extracts the timestamps corresponding to each action, according to the action label. 
2. To synchronize the action timestamps with the timestamps of the accelerometer data, obtaining a precise correspondence between each action and its resulting acceleration pattern. 

## Information about the dataset

Regarding the files, you will see that each folder is structured as follows:
- a /frames sub-folder, containing the sequence of images which make up the video of the experiment
- a timestamp file (in txt format), specifiying the timestamp of each frame contained in the aforementioned subfolder.
- 4 txt files containing the data of wearable accelerometers worn by the person during the experiment. In particular, each person was wearing one accelerometer on the back of the hand and another one on the wrist, both for the left and right arm. The data in the txt file also contain the timestamps, specifying the instant in which each sample was acquired.

## Extraction of Metadata

Download **exiftool.exe** from google.
For more information regarding installation and usage follow the tutorial : https://exiftool.org/

Save the output of the metadata as **out.csv**

## Program to extract the timestamps corresponding to each action, according to the action label

Launch the python file **Meta.py** 
Enter the file location in the script
example : FILE_LOCATION = r"G:\dataset\s01\out.csv"
RUN the program.

## To synchronize the action timestamps with the timestamps of the accelerometer data, obtaining a precise correspondence between each action and its resulting acceleration pattern

Launch the python file **Extraction.py**

We get the synchronised output as shown below files.

my_data_left_backPose.csv,
my_data_left_wristPose.csv,
my_data_right_backPose.csv,
my_data_right_wristPose.csv

