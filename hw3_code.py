#!/usr/bin/env python3
import sympy as sp
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

PEN_MASS = 50/1000  # kg (assuming a 50g pen, converted to kg)
PEN_LENGTH = 4.5/100  # m (4.5 cm)
R = (10/100)/2 # radius of the circle in meters. diameter = 10cm or 0.1 m
NUMBER_OF_SEGMENTS = 5
TOTAL_TIME = 200 # seconds
SEGMENT_DURATION = TOTAL_TIME/NUMBER_OF_SEGMENTS # 40 seconds for each segment so that (4sec/segements)*5segments = 200 sec
NUM_OF_POINTS = 1000 # number of points to be used for the numerical integration
TIME_DELTA = SEGMENT_DURATION/NUM_OF_POINTS # deltat t (seconds)
VERTICAL_DISTANCE = 5/100 # 5 cm coverted to meter. segment 2 vertical line
HORIZONTAL_DISTANCE = 10/100 # 10 cm converted to meter. segment 3 horizontal line

all_xe_points, all_ye_points, all_ze_points = [], [], [] # lists to collect points along the segments.
tau1_points, tau2_points, tau3_points, tau4_points, tau5_points, tau6_points = [], [], [], [], [], [] # lists to collect joint angle torques
time_list = [] # list to collect time points along the segments

# Symbolic variables
x, y, z = sp.symbols("x, y, z") # x,y,z symbols used for parameteric equation of semi-circle's first half.
theta1, theta2, theta3, theta4, theta5, theta6 = sp.symbols("theta1:7") #joint angles
jacobian, inv_jacobian = sp.symbols("jacobian, inv_jacobian") # jacobian matrix
wx, wy, wz, w_ee = sp.symbols("wx, wy, wz, w_ee")# end-effector tool angular velocities
# end-effector linear velocities
x_dot, y_dot, z_dot, Xee_dot, V_ee = sp.symbols("x_dot, y_dot, z_dot, Xee_dot, V_ee")
# t for time in seconds
t = sp.symbols("t")

# Denavit-Hartenberg Table for UR3 robot with pen attached to end-effector(found using Spong's Method)
dh_table = ((theta1+sp.pi/2, sp.pi/2, 0, 0.1833),
            (theta2+sp.pi/2, 0, 0.73731, 0),
            (theta3, 0, 0.3878, 0),
            (theta4+sp.pi/2, sp.pi/2, 0, -0.0955),
            (theta5, sp.pi/2, 0, 0.1155),
            (theta6, 0, 0, 0.1218))

# Mass of Each links
M1, M2, M3, M4, M5, M6 = 2, 3.42, 1.26, 0.8, 0.8, 0.35 # kg
# Mass of Each Links, with mass of pen added to link 6
M1, M2, M3, M4, M5, M6_PLUS_PEN = 2, 3.42, 1.26, 0.8, 0.8, 0.35+PEN_MASS # kg

# Center-of-mass coordinates without pen
COM_WITHOUT_PEN = {'link1': [0, -0.02, 0],
                   'link2': [-0.13, 0, -0.1157],
                   'link3': [-0.05, 0, -0.0238],
                   'link4': [0, 0, 0.01],
                   'link5': [0, 0, 0.01],
                   'link6': [0, 0, -0.02]} # all in meters
# Updated COMs, With the last DH frame (dh frame 6 moved to the end of the tool, instead of the end-effector)
COM_PEN = [0, 0, -PEN_LENGTH/2] # this is w.r.t to DH frame 6
COM_LINK6 = [0, 0, -0.02-PEN_LENGTH] # this is w.r.t to DH frame 6
# Combining link 6 and pen COM
combined_COM_x = (COM_LINK6[0] * M6 + COM_PEN[0] * PEN_MASS)/ (M6 + PEN_MASS)
combined_COM_y = (COM_LINK6[1] * M6 + COM_PEN[1] * PEN_MASS)/ (M6 + PEN_MASS)
combined_COM_z = (COM_LINK6[2] * M6 + COM_PEN[2] * PEN_MASS)/ (M6 + PEN_MASS)

COM_WITH_PEN = {'link1': [0, -0.02, 0],
                'link2': [-0.13, 0, -0.1157],
                'link3': [-0.05, 0, -0.0238],
                'link4': [0, 0, 0.01],
                'link5': [0, 0, 0.01],
                'link6_plus_pen': [combined_COM_x, combined_COM_y, combined_COM_z]}

def calculate_local_homogeneous_transform(link: tuple):
    """Calculate the homogeneous transformation matrix for a given link using DH parameters.

    This function implements the standard Denavit-Hartenberg transformation matrix
    calculation for robot kinematics.

    Args:
        link (tuple): DH parameters in the order (theta, alpha, a, d)
            theta: Joint angle in radians
            alpha: Link twist angle in radians
            a: Link length in meters
            d: Link offset in meters

    Returns:
        sp.Matrix: A 4x4 homogeneous transformation matrix representing the link's pose
    """
    theta, alpha, a, d = link  # Unpack parameters to be used in matrix
    
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0, sp.sin(alpha), sp.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def calculate_homogeneous_transforms_wrt_base():
    """Calculate the homogeneous transformation matrices for each DH frame w.r.t
    base frame, using the DH table.

    Returns:
        tuple: 6 elements consisting of:
            - sp.Matrix: 4x4 end-effector homogeneous transformation matrix
    """
    # Calculate transformation matrices for the different frames
    trans_matrix_1_0 = calculate_local_homogeneous_transform(dh_table[0])
    trans_matrix_2_0 = trans_matrix_1_0 * calculate_local_homogeneous_transform(dh_table[1])
    trans_matrix_3_0 = trans_matrix_2_0 * calculate_local_homogeneous_transform(dh_table[2])
    trans_matrix_4_0 = trans_matrix_3_0 * calculate_local_homogeneous_transform(dh_table[3])
    trans_matrix_5_0 = trans_matrix_4_0 * calculate_local_homogeneous_transform(dh_table[4])
    trans_matrix_6_0 = trans_matrix_5_0 * calculate_local_homogeneous_transform(dh_table[5])
    return trans_matrix_1_0, trans_matrix_2_0, trans_matrix_3_0, trans_matrix_4_0, trans_matrix_5_0, trans_matrix_6_0

def calculate_jacobian_matrix(transforms_wrt_base):
    """Calculate the end-effector transformation matrix and geometric Jacobian.
    
    Computes the geometric Jacobian matrix for the UR3 robot
    using the transformation matrices w.r.t base frame. The Jacobian includes both linear and angular velocity
    components.
    Args:
        tuple: 6 elements made of:
            - sp.Matrix: 4x4 homogeneous transformation matrices

    Returns:
        Jacobian: 6x6 sp.Matrix
    """
    # Calculate top part of Jacobian (linear velocity) by taking the partial derivative w.r.t theta1 to theta6
    j11 = sp.diff(transforms_wrt_base[5][:3, -1], theta1)
    j12 = sp.diff(transforms_wrt_base[5][:3, -1], theta2)
    j13 = sp.diff(transforms_wrt_base[5][:3, -1], theta3)
    j14 = sp.diff(transforms_wrt_base[5][:3, -1], theta4)
    j15 = sp.diff(transforms_wrt_base[5][:3, -1], theta5)
    j16 = sp.diff(transforms_wrt_base[5][:3, -1], theta6)
    
    j_top = sp.Matrix.hstack(j11, j12, j13, j14, j15, j16)
    
    # Compute bottom part of the Jacobian (angular velocity) by taking the Z axis components.
    j21 = transforms_wrt_base[0][:3, -2]
    j22 = transforms_wrt_base[1][:3, -2]
    j23 = transforms_wrt_base[2][:3, -2]
    j24 = transforms_wrt_base[3][:3, -2]
    j25 = transforms_wrt_base[4][:3, -2]
    j26 = transforms_wrt_base[5][:3, -2]

    j_bottom = sp.Matrix.hstack(j21, j22, j23, j24, j25, j26)
    # return ee homogeneous transform and Jacobian (after stacking top and bottom components)
    return sp.Matrix.vstack(j_top, j_bottom)

def create_circle_parametric_equations(radius: float, start_theta: float, omega: float, 
                    circle_center: tuple, rotation_direction: int, plane: str):
    """Create parametric equations for a circle or circular arc in 3D space.
    
    Args:
        radius (float): Circle radius in meters
        start_theta (float): Initial angle in radians
        omega (float): Angular velocity in radians/second
        circle_center (tuple): Center coordinates (x, y, z) in meters
        rotation_direction (int): Direction of rotation:
            1 for counter-clockwise
            -1 for clockwise
        plane (str): Plane of the circle:
            'xy': Circle in XY plane
            'xz': Circle in XZ plane
            'yz': Circle in YZ plane

    Returns:
        tuple[sp.Expr, sp.Expr, sp.Expr]: Parametric equations (x(t), y(t), z(t))
            representing the circle's position as a function of time
    """
    center_x, center_y, center_z = circle_center # coordinate of circle center

    if plane == 'yz':
        x = center_x
        y = center_y + radius*sp.cos(start_theta + omega*t)
        z = center_z + rotation_direction*radius*sp.sin(start_theta + omega*t)
    if plane == 'xy':
        x = center_x + radius*sp.cos(start_theta + omega*t)
        y = center_y + rotation_direction*radius*sp.sin(start_theta + omega*t)
        z = center_z
    if plane == 'xz':
        x = center_x + radius*sp.cos(start_theta + omega*t)
        y = center_y 
        z = center_z + rotation_direction*radius*sp.sin(start_theta + omega*t)

    return x, y, z

def create_line_parametric_equations(start: tuple, velocity: tuple):
    """Generate parametric equations for a 3D line segment.

    Args:
        start (tuple): Starting point coordinates (x, y, z) in meters
        velocity (tuple): Constant velocity components (vx, vy, vz) in meters/second

    Returns:
        tuple[sp.Expr, sp.Expr, sp.Expr]: Parametric equations (x(t), y(t), z(t))
            representing the line's position as a function of time
    """
    start_x, start_y, start_z = start
    velocity_x, velocity_y, velocity_z = velocity

    x = start_x + velocity_x*t
    y = start_y + velocity_y*t
    z = start_z + velocity_z*t

    return x, y, z

def calculate_ee_velocity_vector(parametric_equations: tuple, angular_velocities: tuple):
    """Create the end-effector velocity vector combining linear and angular components.

    Args:
        parametric_equations (tuple): Position equations (x(t), y(t), z(t)) as SymPy expressions
        angular_velocities (tuple): Angular velocity components (wx, wy, wz) in rad/s

    Returns:
        sp.Matrix: 6x1 velocity vector where:
            - Top 3 elements: Linear velocities (dx/dt, dy/dt, dz/dt)
            - Bottom 3 elements: Angular velocities (wx, wy, wz)
    """
    ee_x, ee_y, ee_z = parametric_equations
    ee_wx, ee_wy, ee_wz = angular_velocities
    # End-effector linear velocities
    ee_x_dot = sp.diff(ee_x, t)
    ee_y_dot = sp.diff(ee_y, t)
    ee_z_dot = sp.diff(ee_z, t)
    ee_V = sp.Matrix([[ee_x_dot], [ee_y_dot], [ee_z_dot]]) # linear velocities vector
    ee_w = sp.Matrix([[ee_wx], [ee_wy], [ee_wz]]) # angular velocities vector
    # Stack them vertically and return
    return sp.Matrix.vstack(ee_V, ee_w)

def integrate_trajectory_segment(time_start: float, joint_angles: np.ndarray, 
                           initial_jacobian: np.ndarray, initial_ee_velocity: np.ndarray,
                           transform_func, ee_velocity_func, jacobian_func, tau_func):
    """Calculate end-effector positions using Euler numerical integration.
    
    This function implements forward kinematics and numerical integration to compute
    the end-effector trajectory points for a segment of the overall path.

    Args:
        time_start (float): Initial time for the segment in seconds
        joint_angles (np.ndarray): Initial joint angles in radians
        initial_jacobian (np.ndarray): Initial Jacobian matrix
        initial_ee_velocity (np.ndarray): Initial end-effector velocity vector
        transform_func (callable): Function to compute end-effector transformation matrix
        ee_velocity_func (callable): Function to compute end-effector velocity vector
        jacobian_func (callable): Function to compute Jacobian matrix
        tau_func (callable): Function to compute the vector containing the torque at each joint

    Returns:
        tuple: A pair containing:
            - float: Final time at segment end in seconds
            - np.ndarray: Final joint angles at segment end in radians

    Note:
        Updates global lists all_xe_points, all_ye_points, and all_ze_points
        with the computed trajectory points.
    """
    current_time = time_start
    current_joints = joint_angles
    current_jacobian = initial_jacobian
    current_ee_velocity = initial_ee_velocity

    for _ in range(NUM_OF_POINTS):
        # Calculate pseudo-inverse of Jacobian
        jacobian_inverse = np.linalg.pinv(current_jacobian)
        
        # Numerical integration for joint angles
        new_joints = current_joints + (jacobian_inverse @ current_ee_velocity) * TIME_DELTA
        joint_values = np.array(new_joints).astype(float).flatten()
        
        # Calculate new transformation matrix
        new_transform = np.float64(transform_func(*joint_values))
        x_ee, y_ee, z_ee = new_transform[:3, 3] # end-effector coordinates from the last column

        # Calculate torque matrix with new joint angles
        new_tau = tau_func(*joint_values)

        # Store end-effector coordinates
        all_xe_points.append(x_ee)
        all_ye_points.append(y_ee)
        all_ze_points.append(z_ee)

        # Store Torque values
        tau1_points.append(new_tau[0,-1])
        tau2_points.append(new_tau[1,-1])
        tau3_points.append(new_tau[2, -1])
        tau4_points.append(new_tau[3,-1])
        tau5_points.append(new_tau[4, -1])
        tau6_points.append(new_tau[5, -1])
        
        # Update for next iteration
        current_time += TIME_DELTA
        time_list.append(current_time) # add time to time tracking list
        current_jacobian = jacobian_func(*joint_values)
        current_ee_velocity = ee_velocity_func(current_time)
        current_joints = new_joints
    
    return current_time, joint_values

def plot_yz_trajectory():
    """Plot the end-effector trajectory in the YZ plane.
    
    Creates a 2D scatter plot showing the robot's end-effector path
    using the collected points from the simulation. The plot includes:
        - YZ plane projection of the trajectory
        - Axis labels in meters
        - Equal aspect ratio
        - Grid lines
    """
    plt.figure(figsize=(10, 8))
    plt.plot(all_ye_points, all_ze_points, 'o', markersize=2)
    plt.xlabel('Y Position [m]')
    plt.ylabel('Z Position [m]')
    plt.title('End-Effector Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.autoscale(True)
    plt.show()

def plot_joint_angle_torques():
    """ Plots each of the joint angle torque with respect to time """
    #------- Tau1 Plot------------------------------------------------
    # plt.figure(figsize=(10, 8))
    plt.plot(time_list, tau1_points, 'o', markersize=2)  # 'o' creates a scatter plot
    # axis labels
    plt.xlabel('time [sec]')
    plt.ylabel('Tau1 [Nm]')
    #Axis Title
    plt.title('2-D Plot of Torque at Joint1 vs Time')
    # Set Grid
    plt.grid(True)
    # set aspect ratio and axis limits
    # plt.axis('equal')  # Equal aspect ratio
    plt.xlim(0, 210)
    plt.ylim(-30, 30)
    # plt.autoscale(True)
    plt.show()
    #------- Tau2 Plot------------------------------------------------
    plt.plot(time_list, tau2_points, 'o', markersize=2)  # 'o' creates a scatter plot
    # axis labels
    plt.xlabel('time [sec]')
    plt.ylabel('Tau2 [Nm]')
    #Axis Title
    plt.title('2-D Plot of Torque at Joint2 vs Time')
    # Set Grid
    plt.grid(True)
    # set aspect ratio and axis limits
    # plt.axis('equal')  # Equal aspect ratio
    plt.xlim(0, 210)
    plt.ylim(-30, 30)
    # plt.autoscale(True)
    plt.show()
    #------- Tau3 Plot------------------------------------------------
    plt.plot(time_list, tau3_points, 'o', markersize=2)  # 'o' creates a scatter plot
    # axis labels
    plt.xlabel('time [sec]')
    plt.ylabel('Tau3 [Nm]')
    #Axis Title
    plt.title('2-D Plot of Torque at Joint3 vs Time')
    # Set Grid
    plt.grid(True)
    # set aspect ratio and axis limits
    # plt.axis('equal')  # Equal aspect ratio
    plt.xlim(0, 210)
    plt.ylim(-30, 30)
    # plt.autoscale(True)
    plt.show()
    #------- Tau4 Plot------------------------------------------------
    plt.plot(time_list, tau4_points, 'o', markersize=2)  # 'o' creates a scatter plot
    # axis labels
    plt.xlabel('time [sec]')
    plt.ylabel('Tau4 [Nm]')
    #Axis Title
    plt.title('2-D Plot of Torque at Joint4 vs Time')
    # Set Grid
    plt.grid(True)
    # set aspect ratio and axis limits
    # plt.axis('equal')  # Equal aspect ratio
    plt.xlim(0, 210)
    plt.ylim(-30, 30)
    # plt.autoscale(True)
    plt.show()
    #------- Tau5 Plot------------------------------------------------
    plt.plot(time_list, tau5_points, 'o', markersize=2)  # 'o' creates a scatter plot
    # axis labels
    plt.xlabel('time [sec]')
    plt.ylabel('Tau5 [Nm]')
    #Axis Title
    plt.title('2-D Plot of Torque at Joint5 vs Time')
    # Set Grid
    plt.grid(True)
    # set aspect ratio and axis limits
    # plt.axis('equal')  # Equal aspect ratio
    plt.xlim(0, 210)
    plt.ylim(-30, 30)
    # plt.autoscale(True)
    plt.show()
    #------- Tau6 Plot------------------------------------------------
    plt.plot(time_list, tau6_points, 'o', markersize=2)  # 'o' creates a scatter plot
    # axis labels
    plt.xlabel('time [sec]')
    plt.ylabel('Tau6 [Nm]')
    #Axis Title
    plt.title('2-D Plot of Torque at Joint6 vs Time')
    # Set Grid
    plt.grid(True)
    # set aspect ratio and axis limits
    # plt.axis('equal')  # Equal aspect ratio
    plt.xlim(0, 210)
    plt.ylim(-30, 30)
    # plt.autoscale(True)
    plt.show()

def main():
    """Simulate a UR3 robot drawing a composite shape.
    
    The shape consists of:
        1. First half of a semi-circle
        2. Vertical line segment (5 cm)
        3. Horizontal line segment (10 cm)
        4. Second vertical line segment (5 cm)
        5. Second half of the semi-circle
    
    The simulation:
        - Calculates forward kinematics and Jacobian matrices
        - Performs numerical integration for trajectory generation
        - Plots the resulting end-effector path
    """
    transforms_wrt_base = calculate_homogeneous_transforms_wrt_base()
    transM_6_0 = transforms_wrt_base[5]
    jacobian = calculate_jacobian_matrix(transforms_wrt_base)
    # Printing the jacobian matrix
    # print("the jacobian matrix in symbolic form is: ")
    # sp.pprint(jacobian)
    
    # Adding 1 at the end of each COM coordinate to convert them into homogeneous form.
    for link in COM_WITH_PEN.keys():
        COM_WITH_PEN[link].append(1)

    # Homogeneous center-of-mass coordinates, converted to column vectors w.r.t the base frame
    r1_0h = transforms_wrt_base[0] * sp.transpose(sp.Matrix([COM_WITH_PEN['link1']]))
    r2_0h = transforms_wrt_base[1] * sp.transpose(sp.Matrix([COM_WITH_PEN['link2']]))
    r3_0h = transforms_wrt_base[2] * sp.transpose(sp.Matrix([COM_WITH_PEN['link3']]))
    r4_0h = transforms_wrt_base[3] * sp.transpose(sp.Matrix([COM_WITH_PEN['link4']]))
    r5_0h = transforms_wrt_base[4] * sp.transpose(sp.Matrix([COM_WITH_PEN['link5']]))
    r6_0h = transforms_wrt_base[5] * sp.transpose(sp.Matrix([COM_WITH_PEN['link6_plus_pen']]))
    # Taking the x,y,z coordinates by taking the top 3 elements
    R1_0 = r1_0h[0:3,-1]
    R2_0 = r2_0h[0:3,-1]
    R3_0 = r3_0h[0:3,-1]
    R4_0 = r4_0h[0:3,-1]
    R5_0 = r5_0h[0:3,-1]
    R6_0 = r6_0h[0:3,-1]

    # gravity
    gravity = sp.symbols("gravity")
    gravity = sp.Matrix([[0.0], [0.0], [9.81]])
    gravity_T = sp.transpose(gravity)
    # Potential Energy  potential_energy = (m*g^T)*r_cm
    potential_energy = sp.symbols("potential_energy")
    potential_energy = M1*gravity_T*R1_0 + M2*gravity_T*R2_0 + M3*gravity_T*R3_0 + M4*gravity_T*R4_0 + M5*gravity_T*R5_0 + M6_PLUS_PEN*gravity_T*R6_0
    gravity_matrix = sp.Matrix([[sp.diff(potential_energy, theta1)], [sp.diff(potential_energy, theta2)], [sp.diff(potential_energy,theta3)],
                    [sp.diff(potential_energy, theta4)], [sp.diff(potential_energy, theta5)], [sp.diff(potential_energy,theta6)]])

    # force vector F = [Fx, Fy, Fz, Tx, Ty, Tz]
    force = sp.symbols("force")
    force = sp.Matrix([[5.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    #Torque
    torque = sp.symbols("torque")
    torque = gravity_matrix - sp.transpose(jacobian)*force

    ############# SEGMENT 1: FIRST HALF OF SEMI-CIRCLE ################################################################
    # INITIAL CONDITIONS
    t_0 = 0.0
    tht1, tht2, tht3, tht4, tht5, tht6 = 0.0, 0.0, np.pi/2, -np.pi/2, 0.0, 0.0
    q0 = np.array([[tht1],[tht2],[tht3],[tht4],[tht5],[tht6]], dtype=float)
    # Calculate the initial end-effector transformation matrix using q0
    init_transM_6_0 = sp.matrix2numpy(transM_6_0.subs({theta1: q0[0,-1], theta2:q0[1, -1], theta3: q0[2, -1], theta4: q0[3, -1], 
                                                       theta5: q0[4, -1], theta6:q0[5, -1]}), dtype=float)
    # Taking the initial end-effector position coordinates from the last column of the transformation matrix
    init_x_ee, init_y_ee, init_z_ee = init_transM_6_0[:3, 3]  # (-0.2173, -0.3878, 1.03611)
    # center of circle wrt to base frame, and its radius "R"
    C_x, C_y, C_z = init_x_ee, init_y_ee, init_z_ee-R # center coordinate, derived from end-effector coord
    omega = (np.pi/2)/SEGMENT_DURATION  # rad/s. constant angular speed of the tool
    # parametric equation of a circle centered at C_x,C_y,C_z, with radius, with start point at pi/2 rad
    # x = C_x ; y = C_y + R*sp.cos(omega*t +np.pi/2); z = C_z + R*sp.sin(omega*t +np.pi/2)
    x, y, z = create_circle_parametric_equations(R, np.pi/2, omega, (C_x, C_y, C_z), 1, 'yz')

    # # end-effector angular velocity = 0 for this problem
    wx, wy, wz = 0.0, 0.0, 0.0
    # End-effector vector with linear and angular velocities 6x1 vector
    Xee_dot = calculate_ee_velocity_vector((x,y,z), (wx, wy, wz))

    # Calculate the initial Jacobian matrix
    init_jac = sp.matrix2numpy(jacobian.subs({theta1: tht1, theta2: tht2, theta3: tht3, theta4: tht4, theta5: tht5, theta6: tht6}), 
                               dtype=float)
    # calculate the initial end-effector velocity vector
    init_Xee_dot = sp.matrix2numpy(Xee_dot.subs({t:t_0}), dtype=float)
    # Initializing torque 
    tau_init = sp.matrix2numpy(torque.subs({theta1: q0[0,-1], theta2: q0[1, -1], 
    theta3: q0[2, -1], theta4: q0[3, -1], theta5: q0[4, -1], theta6: q0[5, -1]}), 
    dtype=float)
    # set the starting time to t_0 to be used in the loop
    t_nu = t_0

    # Update segment point collection lists with initial end-effector coordinate points
    all_xe_points.append(init_x_ee)
    all_ye_points.append(init_y_ee)
    all_ze_points.append(init_z_ee)

    time_list.append(t_nu)

    #updating Tau_pts lists with their initial values
    tau1_points.append(tau_init[0,-1])
    tau2_points.append(tau_init[1,-1])
    tau3_points.append(tau_init[2, -1])
    tau4_points.append(tau_init[3,-1])
    tau5_points.append(tau_init[4, -1])
    tau6_points.append(tau_init[5, -1])

    # Creating Lambdifying functions to improve efficieny in the numerical integration for loop
    transM_6_0_fnc = sp.lambdify([theta1, theta2, theta3, theta4, theta5, theta6], transM_6_0, modules=["numpy"])
    jacobian_fnc = sp.lambdify([theta1, theta2, theta3, theta4, theta5, theta6], jacobian, modules=["numpy"])
    Xee_dot_fnc = sp.lambdify(t, Xee_dot, modules=["numpy"])
    tau_func = sp.lambdify([theta1, theta2, theta3, theta4, theta5, theta6], torque, modules=["numpy"])


    ############################# SEGMENT2: FIRST VERTICLE LINE ###########################################################
    distance = VERTICAL_DISTANCE  # 5 cm verticle segment
    velocity = distance/SEGMENT_DURATION

    # Parametric equations for the vertical line
    # x2 = C_x; y2 = all_ye_points[-1]; z2 = all_ze_points[-1] - velocity * t
    x2, y2, z2 = create_line_parametric_equations ((C_x, all_ye_points[-1], all_ze_points[-1]), (0, 0, -velocity))
    # set velocity vector
    Xe2_dot = calculate_ee_velocity_vector((x2,y2,z2), Xee_dot[:3, -1])

    # Create a new function for Xe2_dot
    Xe2_dot_fnc = sp.lambdify(t, Xe2_dot, modules=["numpy"])

    # INITIAL CONDITIONS
    # calculate points along segment 1, then set initial time & joint angles qu_values to values at the end of the last segment
    t2_0, qu_values = integrate_trajectory_segment(t_nu, q0, init_jac, init_Xee_dot, transM_6_0_fnc, Xee_dot_fnc, jacobian_fnc, tau_func)
    thet1, thet2, thet3, thet4, thet5, thet6 = qu_values
    q2_0 = np.array([[thet1], [thet2], [thet3], [thet4], [thet5], [thet6]], dtype=float)
    # Calculate the initial Jacobian matrix
    init_jac2 = sp.matrix2numpy(
            jacobian.subs(
                {theta1: thet1, theta2: thet2, theta3: thet3, theta4: thet4, theta5: thet5, theta6: thet6}
            ), 
            dtype=float)
    # Calculate the initial end-effector velocity vector
    init_Xe2_dot = Xe2_dot_fnc(t2_0)

    ############################# SEGMENT3: HORIZONTAL LINE ###########################################################
    distance = HORIZONTAL_DISTANCE  # 10 cm horizontal line
    velocity = distance/SEGMENT_DURATION

    # Parametric equations for the vertical line
    # x3 = C_x; y3 = all_ye_points[-1] + velocity * t; z3 = all_ze_points[-1]
    x3, y3, z3 = create_line_parametric_equations((C_x, all_ye_points[-1], all_ze_points[-1]), (0, velocity, 0))
    # New End-effector 6x1 velocity vector
    Xe3_dot = calculate_ee_velocity_vector((x3,y3,z3), Xee_dot[:3, -1])

    # Create a new function for Xe3_dot
    Xe3_dot_fnc = sp.lambdify(t, Xe3_dot, modules=["numpy"])

    # INITIAL CONDITIONS
    # calculate points along segment 2, then set initial time & joint angles qu_values to values at the end of the last segment
    t3_0, qu2_values = integrate_trajectory_segment(t2_0, q2_0, init_jac2, init_Xe2_dot, transM_6_0_fnc, Xe2_dot_fnc, jacobian_fnc, tau_func)
    tet1, tet2, tet3, tet4, tet5, tet6 = qu2_values
    q3_0 = np.array([[tet1], [tet2], [tet3], [tet4], [tet5], [tet6]], dtype=float)
    # Calculate the initial Jacobian matrix
    init_jac3 = sp.matrix2numpy(
                    jacobian.subs(
                        {theta1: tet1, theta2: tet2, theta3: tet3, theta4: tet4, theta5: tet5, theta6: tet6}
                                ), 
                            dtype=float)
    
    # Calculate the initial end-effector velocity vector
    init_Xe3_dot = Xe3_dot_fnc(t3_0)

    ############################# SEGMENT4: VERTICAL LINE ###########################################################
    distance = VERTICAL_DISTANCE  # second 5cm verticle line, which has the same measurment as the 1st verticle segment
    velocity = distance/SEGMENT_DURATION

    # Parametric equations for the vertical line
    # x4 = C_x; y4 = all_ye_points[-1]; z4 = all_ze_points[-1] + velocity * t
    x4, y4, z4 = create_line_parametric_equations((C_x, all_ye_points[-1], all_ze_points[-1]), (0, 0, velocity))

    # New End-effector 6x1 velocity vector
    Xe4_dot = calculate_ee_velocity_vector((x4,y4,z4), Xee_dot[:3, -1])

    # Create a new function for Xe4_dot
    Xe4_dot_fnc = sp.lambdify(t, Xe4_dot, modules=["numpy"])

    # INITIAL CONDITIONS
    # calculate points along segment 3, then set initial time & joint angles qu_values for segment 4 to values at the end of the last segment
    t4_0, qu3_values = integrate_trajectory_segment(t3_0, q3_0, init_jac3, init_Xe3_dot, transM_6_0_fnc, Xe3_dot_fnc, jacobian_fnc, tau_func)
    # t4_0 = t_nu
    teta1, teta2, teta3, teta4, teta5, teta6 = qu3_values
    q4_0 = np.array([[teta1], [teta2], [teta3], [teta4], [teta5], [teta6]], dtype=float)
    # Calculate the initial Jacobian matrix
    init_jac4 = sp.matrix2numpy(
                    jacobian.subs(
                        {theta1: teta1, theta2: teta2, theta3: teta3, theta4: teta4, theta5: teta5, theta6: teta6}
                        ), dtype=float)
    # Calculate the initial end-effector velocity vector
    init_Xe4_dot = Xe4_dot_fnc(t4_0)

    ############################# SEGMENT5: SECOND HALF OF SEMI-CIRCLE ###########################################################
    # omga = (np.pi/2)/SEGMENT_DURATION # rad/s. constant angular speed along arc

    # parametric equation of for the second half of the semi circle using the same Omega as the first 1/2 semi-circle.
    # x5 = C_x ; y5 = (all_ye_points[-1]-R) + R*sp.cos(omga*t + 0); z5 = all_ze_points[-1] + R*sp.sin(omga*t + 0)
    x5, y5, z5 = create_circle_parametric_equations(R, 0, omega, (C_x, all_ye_points[-1]-R, all_ze_points[-1]), 1, 'yz')
    # # new End-effector 6x1 velocity vector
    Xe5_dot = calculate_ee_velocity_vector((x5,y5,z5), Xee_dot[:3, -1])
    # Create a new function for Xe5_dot
    Xe5_dot_fnc = sp.lambdify(t, Xe5_dot, modules=["numpy"])

    # INITIAL CONDITIONS
    # calculate points along segment 4, then set initial time & joint angles qu_values for segment5 to values at the end of the last segment
    t5_0, qu4_values = integrate_trajectory_segment(t4_0, q4_0, init_jac4, init_Xe4_dot, transM_6_0_fnc, Xe4_dot_fnc, jacobian_fnc, tau_func)
    tta1, tta2, tta3, tta4, tta5, tta6 = qu4_values # set initial angle the same as the last angles from previous segment end
    q5_0 = np.array([[tta1], [tta2], [tta3], [tta4], [tta5], [tta6]], dtype=float)
    # Calculate the initial Jacobian matrix
    init_jac5 = sp.matrix2numpy(
            jacobian.subs(
                {theta1: tta1, theta2: tta2, theta3: tta3, theta4: tta4, theta5: tta5, theta6: tta6}
            ), dtype=float
        )
    # Calculate the initial end-effector velocity vector
    init_Xe5_dot = Xe5_dot_fnc(t5_0)
    # calculate points along segment 5, the last segment
    _ = integrate_trajectory_segment(t5_0, q5_0, init_jac5, init_Xe5_dot, transM_6_0_fnc, Xe5_dot_fnc, jacobian_fnc, tau_func)

    # printing summary of data from joint angle torques
    print(f'The max torque at joint1 is {round(max(tau1_points), 2)} Nm, the minimum torque: {round(min(tau1_points), 2)} Nm, and the mean: {round(st.mean(tau1_points), 4)} Nm')
    print(f'The max torque at joint2 is {round(max(tau2_points), 2)} Nm, the minimum torque: {round(min(tau2_points), 2)} Nm, and the mean: {round(st.mean(tau2_points), 4)} Nm')
    print(f'The max torque at joint3 is {round(max(tau3_points), 2)} Nm, the minimum torque: {round(min(tau3_points), 2)} Nm, and the mean: {round(st.mean(tau3_points), 4)} Nm')
    print(f'The max torque at joint4 is {round(max(tau4_points), 2)} Nm, the minimum torque: {round(min(tau4_points), 2)} Nm, and the mean: {round(st.mean(tau4_points), 4)} Nm')
    print(f'The max torque at joint5 is {round(max(tau5_points), 2)} Nm, the minimum torque: {round(min(tau5_points), 2)} Nm, and the mean: {round(st.mean(tau5_points), 4)} Nm')
    print(f'The max torque at joint6 is {round(max(tau6_points), 2)} Nm, the minimum torque: {round(min(tau6_points), 2)} Nm, and the mean: {round(st.mean(tau6_points), 4)} Nm \n')

    print("The gravity matrix g(q) is:", "\n")
    sp.pprint(gravity_matrix)

    plot_yz_trajectory() # Plot all the points to create the drawing.
    plot_joint_angle_torques() # Plot each joint angle torque with respect to time.

if __name__ == "__main__" :
    main()

