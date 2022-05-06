# x = [x1,x2,x3,a1,a2,a3] -> tip joint
# q_1, q_2, ..., q_n -> other joints, q_i = [x1,x2,x3,a1,a2,a3]
# f_1, f_2, ..., f_6: the function that reflects the relation between q and x, each f corresponds to one dim of x.
# f_m(q_1,q_2,...,qn) -> x_m
from ast import While
from unittest import result
import numpy as np
from numpy import linalg as lng
import math

from sympy import Inverse

# Set Rotation Utils:


def Quat2Conjugate(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.asarray([w, -x, -y, -z])


def GetQuat(Axis, Angle):
    w = math.cos(Angle/2)
    v = Axis * math.sin(Angle/2)
    return np.asarray([w, v[0], v[1], v[2]])


def Hamilton_Product(quat1, quat2):
    w1, w2 = quat1[0], quat2[0]
    x1, x2 = quat1[1], quat2[1]
    y1, y2 = quat1[2], quat2[2]
    z1, z2 = quat1[3], quat2[3]
    return np.asarray([w1*w2-x1*x2-y1*y2-z1*z2,
                       w1*x2+x1*w2+y1*z2-z1*y2,
                       w1*y2-x1*z2+y1*w2+z1*x2,
                       w1*z2+x1*y2-y1*x2+z1*w2])


def SetRotation_Hamilton(Chain, idx, Quat):
    Quat_Conj = Quat2Conjugate(Quat)
    anchor_joint = Chain[idx]
    for joint_idx in range(idx, len(Chain)):
        local_vector = Chain[joint_idx]-anchor_joint
        rotated_vector = np.delete(Hamilton_Product(Hamilton_Product(
            Quat, np.insert(local_vector, 0, 0)), Quat_Conj), 0, 0)
        Chain[joint_idx] = rotated_vector + anchor_joint
    return Chain

# Compute partial derivatives for general rotation link to position target:


def ComputePositionalPartialDerivative(RotationAxis, ToEffector):
    #ToEffector = InChain[InEffectorInd] - InChain[InCurrentInd]
    Out = np.cross(RotationAxis, ToEffector)
    return Out
# Setting target pos closer, hence avoiding sigularity; not necessary in DLS method.


def ClampMag(Vector, D_max):
    VNorm = lng.norm(Vector)
    if VNorm > D_max:
        Vector = Vector / VNorm * D_max
    return Vector
# Make JacobianMatrix.


def CreateJacobianMatrix(InChain, InEffectorInd, RotationAxis_PreComputed):
    # Here we implement the simplified situation: only 1 effector.
    NumCol = len(InChain) - 1
    OutMatrix = np.zeros((3, NumCol))
    for col in range(NumCol):
        # col meanwhile suggests the ind of joint.
        ToEffector = InChain[InEffectorInd] - InChain[col]
        PartialDerivative = ComputePositionalPartialDerivative(
            RotationAxis_PreComputed[col], ToEffector)
        OutMatrix[0, col] = PartialDerivative[0]
        OutMatrix[1, col] = PartialDerivative[1]
        OutMatrix[2, col] = PartialDerivative[2]
    return OutMatrix
# Jacobian Transpose(JT) Method


def JacobianTransposeMethod(InChain, InEffectorInd, TargetPos, InJacobianMatrix):
    JacobianTransposeMatrix = InJacobianMatrix.transpose()
    JacobianSquare = np.matmul(InJacobianMatrix, JacobianTransposeMatrix)
    ToEffector = InChain[InEffectorInd] - InChain[InEffectorInd - 1]
    ToTarget = TargetPos - InChain[InEffectorInd - 1]
    RotationAxis = np.cross(ToEffector, ToTarget)
    # Derivatives:
    EffectorDerivatives = ComputePositionalPartialDerivative(
        RotationAxis, ToEffector).transpose()
    Result = np.matmul(JacobianSquare, EffectorDerivatives)
    # DeltaTheta:
    AlphaBottom = np.inner(Result, Result)
    AlphaUp = np.inner(EffectorDerivatives, Result)
    InOutAngleDerivativeMatrix = (
        AlphaUp / AlphaBottom) * np.matmul(JacobianTransposeMatrix, EffectorDerivatives)
    return InOutAngleDerivativeMatrix
# Jacobian Pseudo-Inverse Damped Least Square(JPIDLS) Method


def JacobianPIDLSMethod(InChain, InEffectorInd, TargetPos, InJacobianMatrix, InDamping):
    JacobianTransposeMatrix = InJacobianMatrix.transpose()
    Damping = InDamping
    DampingIdentityMatrix = np.eye(len(InJacobianMatrix)) * Damping * Damping
    JacobianSquare = np.matmul(InJacobianMatrix, JacobianTransposeMatrix)
    JacobianSquare = JacobianSquare + DampingIdentityMatrix
    #InverseMatrix = lng.inv(JacobianSquare)
    InverseMatrix = lng.pinv(JacobianSquare)
    # Derivatives:
    ToEffector = InChain[InEffectorInd] - InChain[InEffectorInd - 1]
    ToTarget = TargetPos - InChain[InEffectorInd - 1]
    RotationAxis = np.cross(ToEffector, ToTarget)
    EffectorDerivatives = ComputePositionalPartialDerivative(
        RotationAxis, ToEffector).transpose()
    # DeltaTheta:
    InOutAngleDerivativeMatrix = np.matmul(
        JacobianTransposeMatrix, InverseMatrix)
    InOutAngleDerivativeMatrix = np.matmul(
        InOutAngleDerivativeMatrix, EffectorDerivatives)
    return InOutAngleDerivativeMatrix
# JacobianIK


def JacobianIK(InChain, InEffectorInd, TargetPos, MaxIteration, Precision, InDamping=100):
    bReverseChain = False
    if InEffectorInd == 0:
        InChain = InChain[::-1]
        InEffectorInd = len(InChain)-1
        bReverseChain = True
    Distance = lng.norm(TargetPos - InChain[InEffectorInd])
    IterationCount = 0
    while Distance > Precision and IterationCount + 1 < MaxIteration:
        IterationCount += 1
        RotationAxis_PreComputed = []
        for i in range(len(InChain)-1):
            ToEffector = InChain[InEffectorInd] - InChain[i]
            ToTarget = TargetPos - InChain[i]
            InRotationAxis = np.cross(ToEffector, ToTarget)
            InRotationAxis = InRotationAxis / lng.norm(InRotationAxis)
            RotationAxis_PreComputed.append(InRotationAxis)
        # Add Rotation of EndEffector for convenience.
        RotationAxis_PreComputed.append(np.asarray([0, 0, 0]))
        InJacobianMatrix = CreateJacobianMatrix(
            InChain, InEffectorInd, RotationAxis_PreComputed)
        # JT Method:
        AngleDerivativeMatrix = JacobianTransposeMethod(
            InChain, InEffectorInd, TargetPos, InJacobianMatrix)
        # JPIDLS Method:
        #AngleDerivativeMatrix = JacobianPIDLSMethod(InChain, InEffectorInd, TargetPos, InJacobianMatrix, InDamping)
        # Update Chain
        for LinkIndex in range(len(InChain)-2, -1, -1):
            InRotationAxis = RotationAxis_PreComputed[LinkIndex]
            InAngle = AngleDerivativeMatrix[LinkIndex]
            InQuat = GetQuat(InRotationAxis, InAngle)
            InChain = SetRotation_Hamilton(InChain, LinkIndex, InQuat)
        # Update Distance
        Distance = lng.norm(TargetPos - InChain[InEffectorInd])
    if bReverseChain == True:
        InChain = InChain[::-1]
    return InChain


x, y, z = [0., 0., 0., 0., 0.], [1., 2., 3., 4., 5.], [1., 2., 3., 4., 5.]
t1, t2, t3 = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
chain = np.asarray([x, y, z]).transpose()
TargetPos = np.asarray([0, 4.3, 4])
a = JacobianIK(chain, 4, TargetPos, 100, 0.1, InDamping=5)
bones = []
for i in range(1, len(a)):
    bones.append(lng.norm(a[i]-a[i-1]))
print(bones)
print(a)
