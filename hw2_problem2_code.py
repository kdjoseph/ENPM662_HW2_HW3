import sympy as sp

theta1, theta2, theta3, theta4, theta5, theta6 = sp.symbols("theta1, theta2, theta3, theta4, theta5, theta6")  # symbols for joint angles
jacobian = sp.symbols("jacobian") # jacobian matrix

# Denavit-Hartenberg Table for UR3 robot
dh_table = ((theta1+sp.pi/2, sp.pi/2, 0, 0.1833),
            (theta2+sp.pi/2, 0, 0.73731, 0),
            (theta3, 0, 0.3878, 0),
            (theta4+sp.pi/2, sp.pi/2, 0, -0.0955),
            (theta5, sp.pi/2, 0, 0.1155),
            (theta6, 0, 0, 0.0768))

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

def main ():
    """ Main function to run the code"""

    #****** Calculate The Transformation Matrices for the different frames ******
    transM_1_0 = calc_ht_matrix(dh_table[0])
    transM_2_0 = transM_1_0 * calc_ht_matrix(dh_table[1])
    transM_3_0 = transM_2_0 * calc_ht_matrix(dh_table[2])
    transM_4_0 = transM_3_0 * calc_ht_matrix(dh_table[3])
    transM_5_0 = transM_4_0 * calc_ht_matrix(dh_table[4])
    transM_6_0 = transM_5_0 * calc_ht_matrix(dh_table[5]) # this is the end-effector's homogeneous transform matrix w.r.t the base frame
    #Calculate the top part of the jacobian
    # To do this, the end-effector x,y,z coordinates are taken from the homogenous transform transM_6_0 (elements in the last column & top 3 rows)
    # then the partial derivatives of those are taken w.r.t to each of the 6 theta joint angles
    j11 = sp.diff(transM_6_0[:3, -1], theta1)
    j12 = sp.diff(transM_6_0[:3, -1], theta2)
    j13 = sp.diff(transM_6_0[:3, -1], theta3)
    j14 = sp.diff(transM_6_0[:3, -1], theta4)
    j15 = sp.diff(transM_6_0[:3, -1], theta5)
    j16 = sp.diff(transM_6_0[:3, -1], theta6)
    # Stacking the 3x1 vectors horizontally to form the top part of the jacobian matrix (3x6)
    j_top = sp.Matrix.hstack(j11, j12, j13, j14, j15, j16)
    #Getting the bottom part of the jacobian by taking Z from
    # each of the transformation matrices, which is the top 3 elements from the 3rd column.
    j21 = transM_1_0[:3, -2]
    j22 = transM_2_0[:3, -2]
    j23 = transM_3_0[:3, -2]
    j24 = transM_4_0[:3, -2]
    j25 = transM_5_0[:3, -2]
    j26 = transM_6_0[:3, -2]
    # stacking the 3x1 vectors horizontally to form the bottom part of the jacobian (3x6)
    j_bottom = sp.Matrix.hstack(j21, j22, j23, j24, j25, j26)
    #jacobian matrix is made by stacking j_top onto j_bottom
    jacobian = sp.Matrix.vstack(j_top, j_bottom)
    # Printing the jacobian matrix
    print("the Jacobian matrix is \n")
    sp.pprint(jacobian)

if __name__ == "__main__":
    main()