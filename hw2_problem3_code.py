import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

R = (10/100)/2 # radius of the circle. diameter = 10cm or 0.1 m
NUMER_OF_SEGMENTS = 5
TOT_TIME = 20 # seconds
segment_duration = TOT_TIME/NUMER_OF_SEGMENTS # 4 seconds for each segment so that (4sec/segements)*5segments = 20 sec
# setting up the number of pts to be used for the numberical intergration
NUM_OF_POINTS = 1000.0 
time_delta = segment_duration/NUM_OF_POINTS 
DISTANCE_SEGMENT2 = 5/100 # 5 cm coverted to meter. segment 2 vertical line
DISTANCE_SEGMENT3 = 10/100 # 10 cm converted to meter. segment 3 horizontal line

x, y, z = sp.symbols("x, y, z")
# joint angles
theta1, theta2, theta3, theta4, theta5, theta6 = sp.symbols("theta1, theta2, theta3, theta4, theta5, theta6")
jacobian, inv_jacobian = sp.symbols("jacobian, inv_jacobian") # jacobian matrix
# end-effector tool angular velocities
wx, wy, wz, w_ee = sp.symbols("wx, wy, wz, w_ee")
# end-effector linear velocities
x_dot, y_dot, z_dot, Xee_dot, V_ee = sp.symbols("x_dot, y_dot, z_dot, Xee_dot, V_ee")
# t for time in seconds
t = sp.symbols("t")

# Denavit-Hartenberg Table for UR3 robot
dh_table = ((theta1+sp.pi/2, sp.pi/2, 0, 0.1833),
            (theta2+sp.pi/2, 0, 0.73731, 0),
            (theta3, 0, 0.3878, 0),
            (theta4+sp.pi/2, sp.pi/2, 0, -0.0955),
            (theta5, sp.pi/2, 0, 0.1155),
            (theta6, 0, 0, 0.1218))

def calc_ht_matrix(link: tuple) -> sp.Matrix:
    """
    Calculates the homogeneous transformation matrix for a given link.

    Parameters
    ----------
    link : tuple
        Link parameters (theta, alpha, a, d) from the DH table.

    Returns
    -------
    sympy.Matrix
        The homogeneous transformation matrix as a SymPy matrix.
    """
    theta = link[0]
    alpha = link[1]
    a = link[2]
    d = link[3]
    
    homogeneous_transform = sp.Matrix([
    [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
    [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
    [0, sp.sin(alpha), sp.cos(alpha), d],
    [0, 0, 0, 1]])

    return homogeneous_transform

#****** Calculate The Transformation Matrices for the different frames ******
transM_1_0 = calc_ht_matrix(dh_table[0])
transM_2_0 = transM_1_0 * calc_ht_matrix(dh_table[1])
transM_3_0 = transM_2_0 * calc_ht_matrix(dh_table[2])
transM_4_0 = transM_3_0 * calc_ht_matrix(dh_table[3])
transM_5_0 = transM_4_0 * calc_ht_matrix(dh_table[4])
transM_6_0 = transM_5_0 * calc_ht_matrix(dh_table[5]) # this is the end-effector's homogeneous transform matrix
#Calculate the top part of the jacobian
# To do this, the end-effector x,y,z coordinates are taken from the homogenous transform transM_6_0,
# then the partial derivatives of those are taken w.r.t to each of the 6 theta joint angles
j11 = sp.diff(transM_6_0[:3, -1], theta1)
j12 = sp.diff(transM_6_0[:3, -1], theta2)
j13 = sp.diff(transM_6_0[:3, -1], theta3)
j14 = sp.diff(transM_6_0[:3, -1], theta4)
j15 = sp.diff(transM_6_0[:3, -1], theta5)
j16 = sp.diff(transM_6_0[:3, -1], theta6)
# Stacking the 3x6 vectors horizontally to form the top part of the jacobian matrix
j_top = sp.Matrix.hstack(j11, j12, j13, j14, j15, j16)
#Getting the bottom part of the jacobian by taking Z from
# each of the transformation matrices, which is the top 3 elements from the 3rd column.
j21 = transM_1_0[:3, -2]
j22 = transM_2_0[:3, -2]
j23 = transM_3_0[:3, -2]
j24 = transM_4_0[:3, -2]
j25 = transM_5_0[:3, -2]
j26 = transM_6_0[:3, -2]
# stacking the 3x6 vectors horizontally to form the top part of the jacobian
j_bottom = sp.Matrix.hstack(j21, j22, j23, j24, j25, j26)
#jacobian matrix is made by stacking j_top onto j_bottom
jacobian = sp.Matrix.vstack(j_top, j_bottom)
# Printing the jacobian matrix
print("the jacobian matrix in symbolic form is: ")
sp.pprint(jacobian)
print('\n')

# INITIAL CONDITIONS
t_0 = 0.0
tht1, tht2, tht3, tht4, tht5, tht6 = 0.0, 0.0, np.pi/2, -np.pi/2, 0.0, 0.0
q0 = np.float64(np.array([[tht1],[tht2],[tht3],[tht4],[tht5],[tht6]]))
# Calculate the initial end-effector transformation matrix using q0
init_transM_6_0 = np.float64(sp.matrix2numpy(transM_6_0.subs({theta1: q0[0,-1], theta2:q0[1, -1], theta3: q0[2, -1], theta4: q0[3, -1], theta5: q0[4, -1], theta6:q0[5, -1]})))
# Taking the initial end-effector position coordinates from the last column of the transformation matrix
init_x_ee, init_y_ee, init_z_ee = init_transM_6_0[:3, 3]  # (-0.2173, -0.3878, 1.03611)
# center of circle wrt to base frame, and its radius "R"
C_x, C_y, C_z = init_x_ee, init_y_ee, init_z_ee-R # center coordinate, derived from end-effector without tool coord
omega = (np.pi/2)/segment_duration  # rad/s. constant angular speed of the tool
# parametric equation of a circle centered at C_x,C_y,C_z, with radius, with start point at pi/2
x = C_x 
y = C_y + R*sp.cos(omega*t +np.pi/2)
z = C_z + R*sp.sin(omega*t +np.pi/2)

# End-effector linear velocities
x_dot = sp.diff(x, t)
y_dot = sp.diff(y, t)
z_dot = sp.diff(z, t)

V_ee = sp.Matrix([[x_dot], [y_dot], [z_dot]])
# end-effector angular velocity = 0 for this problem
wx, wy, wz = 0.0, 0.0, 0.0
w_ee = sp.Matrix([wx, wy, wz])
# end-effector vector with linear and angular velocities
Xee_dot = sp.Matrix.vstack(V_ee, w_ee)

# Calculate the initial Jacobian matrix
init_jac = np.float64(sp.matrix2numpy(jacobian.subs({theta1: tht1, theta2: tht2, theta3: tht3, theta4: tht4, theta5: tht5, theta6: tht6})))
# calculate the initial end-effector velocity vector
init_Xee_dot = sp.matrix2numpy(Xee_dot.subs({t:t_0}))
# set the starting time to t_0 to be used in the loop
t_nu = t_0

# Lists to collect the end-effector coordinate points
all_xe_points = [init_x_ee]
all_ye_points = [init_y_ee]
all_ze_points = [init_z_ee]

# Creating Lambdifying functions to improve efficienC_y in the for loop
transM_6_0_fnc = sp.lambdify([theta1, theta2, theta3, theta4, theta5, theta6], transM_6_0, modules=["numpy"])
jacobian_fnc = sp.lambdify([theta1, theta2, theta3, theta4, theta5, theta6], jacobian, modules=["numpy"])
Xee_dot_fnc = sp.lambdify(t, Xee_dot, modules=["numpy"])

for point in range(int(NUM_OF_POINTS)):
    # Jacobian matrix
    jac0 = init_jac
    jac0= np.float64(jac0)
    # finding the inverse of the Jacobian matrix
    jac0_inv = np.linalg.pinv(jac0)
    Xee_dot_i = init_Xee_dot
    # numberical integration to find the joint angles
    qu = q0 + (jac0_inv @ Xee_dot_i)*time_delta
    qu_values = np.array(qu).astype(float).flatten()  # Flatten Q to a 1D array for function input
    # calcuate the new transformation matrix for the new joint angles qu
    new_transM_6_0 = np.float64(transM_6_0_fnc(*qu_values))
    # finding new x,y,z end-effector coordinates from the last column of the transformation matrix
    newx_e, newy_e, newz_e = new_transM_6_0[:3, 3]
    # updating end-effector lists of coordinates with new x,y,z
    all_xe_points.append(newx_e)
    all_ye_points.append(newy_e)
    all_ze_points.append(newz_e)
    # updating the time
    t_nu = t_nu + time_delta
    # time_points.append(t_nu)
    # Recalculating the Jacobian
    init_jac= jacobian_fnc(*qu_values)
    # Calculating new Xee_dot
    init_Xee_dot = Xee_dot_fnc(t_nu)
    # updating q0 to set it as q_current in next iteration in numberical integration
    q0 = qu
############################# SEGMENT2: FIRST VERTICLE LINE ###########################################################
distance = DISTANCE_SEGMENT2  # 5 cm verticle segment
velocity = distance/segment_duration

# Parametric equations for the vertical line
x2 = C_x
y2 = all_ye_points[-1]
z2 = all_ze_points[-1] - velocity * t

# End-effector linear velocities
x2_dot = sp.diff(x2, t)
y2_dot = sp.diff(y2, t)
z2_dot = sp.diff(z2, t)

V2_e = sp.Matrix([[x2_dot], [y2_dot], [z2_dot]])
Xe2_dot = sp.Matrix.vstack(V2_e, w_ee) # new Xee_dot

# Create a new function for Xe2_dot
Xe2_dot_fnc = sp.lambdify(t, Xe2_dot, modules=["numpy"])

# INITIAL CONDITIONS
t2_0 = t_nu
thet1, thet2, thet3, thet4, thet5, thet6 = qu_values
q2_0 = np.float64(np.array([[thet1], [thet2], [thet3], [thet4], [thet5], [thet6]]))
# Calculate the initial Jacobian matrix
init_jac2 = np.float64(
    sp.matrix2numpy(
        jacobian.subs(
            {theta1: thet1, theta2: thet2, theta3: thet3, theta4: thet4, theta5: thet5, theta6: thet6}
        )
    )
)
# Calculate the initial end-effector velocity vector
init_Xe2_dot = Xe2_dot_fnc(t2_0)

for point in range(int(NUM_OF_POINTS)):
    # Jacobian matrix
    jac2_0 = init_jac2
    jac2_0 = np.float64(jac2_0)
    # Finding the inverse of the Jacobian matrix
    jac2_0_inv = np.linalg.pinv(jac2_0)
    Xe2_dot_i = init_Xe2_dot
    # Numerical integration to find the joint angles
    qu2 = q2_0 + (jac2_0_inv @ Xe2_dot_i) * time_delta
    qu2_values = np.array(qu2).astype(float).flatten()  # Flatten Q to a 1D array for function input
    # Calculate the new transformation matrix for the new joint angles qu
    new2_transM_6_0 = np.float64(transM_6_0_fnc(*qu2_values))
    # Extract new x, y, z end-effector coordinates
    new2_x_e, new2_y_e, new2_z_e = new2_transM_6_0[:3, 3]
    # Update end-effector lists of coordinates with new x, y, z
    all_xe_points.append(new2_x_e)
    all_ye_points.append(new2_y_e)
    all_ze_points.append(new2_z_e)
    # Update the time
    # t_current = t_nu - t2_0
    t_nu += time_delta
    # Recalculate the Jacobian
    init_jac2 = jacobian_fnc(*qu2_values)
    # Calculate new Xee_dot
    # init_Xe2_dot = Xe2_dot_fnc(t_current)
    init_Xe2_dot = Xe2_dot_fnc(t_nu)
    # Update q2_0 for the next iteration
    q2_0 = qu2


############################# SEGMENT3: HORIZONTAL LINE ###########################################################
distance = DISTANCE_SEGMENT3  # 10 cm horizontal line
velocity = distance/segment_duration

# Parametric equations for the vertical line
x3 = C_x
y3 = all_ye_points[-1] + velocity * t
z3 = all_ze_points[-1]

# End-effector linear velocities
x3_dot = sp.diff(x3, t)
y3_dot = sp.diff(y3, t)
z3_dot = sp.diff(z3, t)

V3_e = sp.Matrix([[x3_dot], [y3_dot], [z3_dot]])
Xe3_dot = sp.Matrix.vstack(V3_e, w_ee) # new Xee_dot

# Create a new function for Xe3_dot
Xe3_dot_fnc = sp.lambdify(t, Xe3_dot, modules=["numpy"])

# INITIAL CONDITIONS
t3_0 = t_nu
tet1, tet2, tet3, tet4, tet5, tet6 = qu2_values
q3_0 = np.float64(np.array([[tet1], [tet2], [tet3], [tet4], [tet5], [tet6]]))
# Calculate the initial Jacobian matrix
init_jac3 = np.float64(
    sp.matrix2numpy(
        jacobian.subs(
            {theta1: tet1, theta2: tet2, theta3: tet3, theta4: tet4, theta5: tet5, theta6: tet6}
        )
    )
)
# Calculate the initial end-effector velocity vector
init_Xe3_dot = Xe3_dot_fnc(t3_0)

for point in range(int(NUM_OF_POINTS)):
    # Jacobian matrix
    jac3_0 = init_jac3
    jac3_0 = np.float64(jac3_0)
    # Finding the inverse of the Jacobian matrix
    jac3_0_inv = np.linalg.pinv(jac3_0)
    Xe3_dot_i = init_Xe3_dot  # end-effector velocity vector
    # Numerical integration to find the joint angles
    qu3 = q3_0 + (jac3_0_inv @ Xe3_dot_i) * time_delta
    qu3_values = np.array(qu3).astype(float).flatten()  # Flatten Q to a 1D array for function input
    # Calculate the new transformation matrix for the new joint angles qu
    new3_transM_6_0 = np.float64(transM_6_0_fnc(*qu3_values))
    # Extract new x, y, z end-effector coordinates
    new3_x_e, new3_y_e, new3_z_e = new3_transM_6_0[:3, 3]
    # Update end-effector lists of coordinates with new x, y, z
    all_xe_points.append(new3_x_e)
    all_ye_points.append(new3_y_e)
    all_ze_points.append(new3_z_e)
    # Update the time
    # t_current = t_nu - t3_0
    t_nu += time_delta
    # Recalculate the Jacobian
    init_jac3 = jacobian_fnc(*qu3_values)
    # Calculate new Xee_dot
    # init_Xe3_dot = Xe3_dot_fnc(t_current)
    init_Xe3_dot = Xe3_dot_fnc(t_nu)
    # Update q2_0 for the next iteration
    q3_0 = qu3


############################# SEGMENT4: VERTICAL LINE ###########################################################
distance = DISTANCE_SEGMENT2  # second 5cm verticle line, which has the same measurment as the 1st verticle segment
velocity = distance/segment_duration

# Parametric equations for the vertical line
x4 = C_x
y4 = all_ye_points[-1] 
z4 = all_ze_points[-1] + velocity * t

# new End-effector linear velocities
x4_dot = sp.diff(x4, t)
y4_dot = sp.diff(y4, t)
z4_dot = sp.diff(z4, t)

V4_e = sp.Matrix([[x4_dot], [y4_dot], [z4_dot]])
Xe4_dot = sp.Matrix.vstack(V4_e, w_ee) # new Xee_dot vector

# Create a new function for Xe4_dot
Xe4_dot_fnc = sp.lambdify(t, Xe4_dot, modules=["numpy"])

# INITIAL CONDITIONS
t4_0 = t_nu
teta1, teta2, teta3, teta4, teta5, teta6 = qu3_values
q4_0 = np.float64(np.array([[teta1], [teta2], [teta3], [teta4], [teta5], [teta6]]))
# Calculate the initial Jacobian matrix
init_jac4 = np.float64(
    sp.matrix2numpy(
        jacobian.subs(
            {theta1: teta1, theta2: teta2, theta3: teta3, theta4: teta4, theta5: teta5, theta6: teta6}
        )
    )
)
# Calculate the initial end-effector velocity vector
init_Xe4_dot = Xe4_dot_fnc(t4_0)

for point in range(int(NUM_OF_POINTS)):
    # Jacobian matrix
    jac4_0 = init_jac4
    jac4_0 = np.float64(jac4_0)
    # Finding the inverse of the Jacobian matrix
    jac4_0_inv = np.linalg.pinv(jac4_0)
    Xe4_dot_i = init_Xe4_dot  # end-effector velocity vector
    # Numerical integration to find the joint angles
    qu4 = q4_0 + (jac4_0_inv @ Xe4_dot_i) * time_delta
    qu4_values = np.array(qu4).astype(float).flatten()  # Flatten Q to a 1D array for function input
    # Calculate the new transformation matrix for the new joint angles qu
    new4_transM_6_0 = np.float64(transM_6_0_fnc(*qu4_values))
    # Extract new x, y, z end-effector coordinates
    new4_x_e, new4_y_e, new4_z_e = new4_transM_6_0[:3, 3]
    # Update end-effector lists of coordinates with new x, y, z
    all_xe_points.append(new4_x_e)
    all_ye_points.append(new4_y_e)
    all_ze_points.append(new4_z_e)
    # Update the time
    # t_current = t_nu - t4_0
    t_nu += time_delta
    # Recalculate the Jacobian
    init_jac4 = jacobian_fnc(*qu4_values)
    # Calculate new Xee_dot
    # init_Xe4_dot = Xe4_dot_fnc(t_current)
    init_Xe4_dot = Xe4_dot_fnc(t_nu)
    # Update q2_0 for the next iteration
    q4_0 = qu4


############################# SEGMENT5: SECOND HALF OF SEMI-CIRCLE ###########################################################
omga = (np.pi/2)/segment_duration

# parametric equation of for the second half of the semi circle
x5 = C_x 
y5 = all_ye_points[-1] + R*sp.cos(omga*t + 0)
z5 = all_ze_points[-1] + R*sp.sin(omga*t + 0)

# new End-effector linear velocities
x5_dot = sp.diff(x5, t)
y5_dot = sp.diff(y5, t)
z5_dot = sp.diff(z5, t)

V5_e = sp.Matrix([[x5_dot], [y5_dot], [z5_dot]]) 
Xe5_dot = sp.Matrix.vstack(V5_e, w_ee) # new Xee_dot 6x1 vector

# Create a new function for Xe5_dot
Xe5_dot_fnc = sp.lambdify(t, Xe5_dot, modules=["numpy"])

# INITIAL CONDITIONS
t5_0 = t_nu
tta1, tta2, tta3, tta4, tta5, tta6 = qu4_values # set initial angle the same as the last angles from previous segment end
q5_0 = np.float64(np.array([[tta1], [tta2], [tta3], [tta4], [tta5], [tta6]]))
# Calculate the initial Jacobian matrix
init_jac5 = np.float64(
    sp.matrix2numpy(
        jacobian.subs(
            {theta1: tta1, theta2: tta2, theta3: tta3, theta4: tta4, theta5: tta5, theta6: tta6}
        )
    )
)
# Calculate the initial end-effector velocity vector
init_Xe5_dot = Xe5_dot_fnc(t5_0)

for point in range(0, int(NUM_OF_POINTS)):
    # Jacobian matrix
    jac5_0 = init_jac5
    jac5_0 = np.float64(jac5_0)
    # Finding the inverse of the Jacobian matrix
    jac5_0_inv = np.linalg.pinv(jac5_0)
    Xe5_dot_i = init_Xe5_dot  # end-effector velocity vector
    # Numerical integration to find the joint angles
    qu5 = q5_0 + (jac5_0_inv @ Xe5_dot_i) * time_delta
    qu5_values = np.array(qu5).astype(float).flatten()  # Flatten Q to a 1D array for function input
    # Calculate the new transformation matrix for the new joint angles qu
    new5_transM_6_0 = np.float64(transM_6_0_fnc(*qu5_values))
    # Extract new x, y, z end-effector coordinates
    new5_x_e, new5_y_e, new5_z_e = new5_transM_6_0[:3, 3]
    # Update end-effector lists of coordinates with new x, y, z
    all_xe_points.append(new5_x_e)
    all_ye_points.append(new5_y_e)
    all_ze_points.append(new5_z_e)
    # Update the time
    # t_current = t_nu - t5_0
    t_nu += time_delta
    # Recalculate the Jacobian
    init_jac5 = jacobian_fnc(*qu5_values)
    # Calculate new Xee_dot
    # init_Xe5_dot = Xe5_dot_fnc(t_current)
    init_Xe5_dot = Xe5_dot_fnc(t_nu)
    # Update q2_0 for the next iteration
    q5_0 = qu5


# ************** 2-D PLOTS *******************************************
plt.plot(all_ye_points, all_ze_points, 'o')  # 'o' creates a scatter plot
# axis labels
plt.xlabel('Ye [m]')
plt.ylabel('Ze [m]')
#Axis Title
plt.title('2-D Plot of End-Effector Trajectory')
# Set Grid
plt.grid(True)
# set aspect ratio and axis limits
plt.axis('equal')  # Equal aspect ratio
plt.autoscale(True)
plt.show()
