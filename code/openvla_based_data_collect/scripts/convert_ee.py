import numpy as np
import math

import modern_robotics as mr
from modern_robotics.core import JacobianSpace, Adjoint, MatrixLog6, se3ToVec, TransInv, FKinSpace


##### MR description #####
# https://github.com/Interbotix/interbotix_ros_toolboxes/blob/main/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/mr_descriptions.py
class wx250s:
    Slist = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.11065, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.36065, 0.0, 0.04975],
                      [1.0, 0.0, 0.0, 0.0, 0.36065, 0.0],
                      [0.0, 1.0, 0.0, -0.36065, 0.0, 0.29975],
                      [1.0, 0.0, 0.0, 0.0, 0.36065, 0.0]]).T

    M = np.array([[1.0, 0.0, 0.0, 0.458325],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.36065],
                  [0.0, 0.0, 0.0, 1.0]])
    

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R, thresh=1e-6):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < thresh

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R, check_error_thresh=1e-6):

    assert(isRotationMatrix(R, check_error_thresh))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def ModifiedIKinSpace(Slist, M, T, thetalist0, eomg, ev, maxiterations=40):
    """
    ModifiedIKinSpace - Inverse Kinematics in the Space Frame
    this exposed the max_iterations parameter to the user

    # original source:
    ModernRobotics/packages/Python/modern_robotics/core.py at master · NxRLab/ModernRobotics
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    Tsb = FKinSpace(M,Slist, thetalist)
    Vs = np.dot(Adjoint(Tsb), \
                se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
          or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(JacobianSpace(Slist, \
                                                          thetalist)), Vs)
        i = i + 1
        Tsb = FKinSpace(M, Slist, thetalist)
        Vs = np.dot(Adjoint(Tsb), \
                    se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
              or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    if err:
        print('IKinSpace: did not converge')
        print('Vs', Vs)
    return (thetalist, not err)

def get_ee(joint, gripper):
    result = []

    T_sb = mr.FKinSpace(wx250s.M, wx250s.Slist, joint)
    # Extract position vector
    xyz = T_sb[:3, 3]
    # Extract rotation matrix
    rot_matrix = T_sb[:3, :3]

    # Convert to euler_angles
    euler_angles = rotationMatrixToEulerAngles(rot_matrix)

    # concatenate xyz with rotation
    result.append(np.concatenate((xyz, euler_angles, gripper)))
    return np.array(result)

def get_joint_and_gripper(xyz, euler_angles, gripper, guess_angle):  
    # guess_angle = puppet_bot_left.dxl.joint_states.position[:6]
    # Initialize result container
    joint_and_gripper = []

    rot_matrix = eulerAnglesToRotationMatrix(euler_angles)

    # Construct the homogeneous transformation matrix T_sb
    T_sd = np.eye(4)
    T_sd[:3, :3] = rot_matrix
    T_sd[:3, 3] = xyz

    # Solve inverse kinematics to find joint positions
    # joint_positions, success  = mr.IKinSpace(wx250s.Slist, wx250s.M, T_sd, np.zeros(len(wx250s.Slist)), eomg=1e-2, ev=1e-2)
    joint_positions, success = ModifiedIKinSpace(wx250s.Slist, wx250s.M, T_sd, guess_angle, eomg=1e-3, ev=1e-4)

    if not success:
        print('KinSpace failed to converge')

    # Append the gripper value to the joint positions
    joint_and_gripper = np.concatenate((joint_positions, gripper))

    return joint_and_gripper