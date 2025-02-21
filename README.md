# UR3 ROBOT ARM SIMULATION

This repo contains code files that determine the forward and inverse kinematics equations, and dynamics equations for UR3 robot, and simulates and plots the robots motion while drawing a custom-shape made of a semi-circle and part of rectangle. One of the code files recreates the same simulation with the additional constraint that the robot applies & maintains 5Nm of force on the wall during the drawing. The robot dynamics equations were used to determine the equired joint torques for this requirement. 

## Code Files

There are 4 python code files, 1 Text file containg Matlab code, and 2 PDF reports:

-   `hw2_problem1_code.py`: Prints the end-effector's homogeneous transform matrix equation,then a prompt will display asking the user if they would like to
    perform the validation of these equations by entering joint angles. If the user chooses yes, then they will be prompted to enter six joint angles. 
-   `hw2_problem2_code.py`: Computes the Jacobian Matrix and prints it out to the terminal
-   `hw2_problem3_code.py`: Uses the foward and inverse kinematics equations to simulate the UR3 drawing the custome-shape, then displays a plot of the drawing.
-   `hw3_code.py`: Computes the joint torques required to maintain 5Nm on the wall, while the shape is being drawn. Prints-out statements listing the minimum, maximum and mean of each joint torque during the 200 seconds drawing will display in the terminal.The gravity matrix will also display in the terminal. Displays the plot of the drawing, then when that plot window is closed, the plot of each of the joint angle torques with respect to time will appear one at a time, after each plot window is closed.
-   `robotics_toolbox_matlab_code.txt`: code to be copied into Matlab to use the robotics toolbox
-   `hw2_report.pdf`: Describes how the DH frames & DH table was set and how the foward and velocity kinematics equations (including the Jaboian) were calculated, then used to simulate the UR3 drawing the circle. 
-   `hw3_report.pdf`: Describes how the foward & inverse kinematics were used along with the robot dynamics equations in the hw3_code.py file, and also contains output plots.

## Usage

Each code file can be run separately

1.  Ensure that the dependencies are installed.
2.  Download the desired file, then run it. Follow any input prompt instructions.


## Dependencies

### Software Packages

-   Python 3.8 or later
-   Peter Corke Matlab Robotics Toolbox

### Libraries

-   sympy
-   numpy
-   matplotlib.pyplot