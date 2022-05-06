# 3d dependancy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
# 2d-plot related dependancy
import matplotlib.pyplot as plt
# numpy dependancy
import numpy as np
from numpy import linalg as lng
import math

# CCDIK tools:


def Quat2Conjugate(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.asarray([w, -x, -y, -z])


def Quat2Matrix(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    M1 = [1-2*(y**2)-2*(z**2), 2*x*y - 2*z*w, 2*x*z + 2*y*w]
    M2 = [2*x*y + 2*z*w, 1-2*(x**2)-2*(z**2), 2*y*z - 2*x*w]
    M3 = [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1-2*(x**2)-2*(y**2)]
    M = np.asarray([M1, M2, M3])
    return M


def GetQuat(Axis, Angle):
    w = math.cos(Angle/2)
    v = Axis * math.sin(Angle/2)
    return np.asarray([w, v[0], v[1], v[2]])


def SetRotation(Chain, idx, Quat):
    M = Quat2Matrix(Quat)
    # Since the quaternion represents the local rotation, not global, system changing is needed.
    anchor_joint = Chain[idx]
    bones = []
    for i in range(len(Chain)-1):
        bones.append(lng.norm(Chain[i+1]-Chain[i]))
    for joint_idx in range(idx, len(Chain)):
        local_vector = Chain[joint_idx]-anchor_joint
        Chain[joint_idx] = np.dot(M, local_vector) + anchor_joint
    for i in range(len(Chain)-1):
        bones[i] = lng.norm(Chain[i+1]-Chain[i])
    return Chain


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


def ClampAngle(angle, minAngle, maxAngle):
    angle_clamped = max(angle, minAngle)
    angle_clamped = min(angle_clamped, maxAngle)
    return angle_clamped

# CCDIK main:


def CCDIK(EndInd, InOutChain, AngleDelta, TargetPosition, Precision, MaxIteration, bEnableRotationLimit, RotationLimitPerJoints):
    # If drag from the other side:
    if EndInd == 0:
        InOutChain = InOutChain[::-1]
    # Reference vectors for computing the limitation.
    AngleDeltaRef = []
    for idx in range(0, len(InOutChain)-1):
        AngleDeltaRef.append(InOutChain[idx+1] - InOutChain[idx])
    AngleDeltaRef.append(np.asarray([0, 0, 0]))
    # Update Core:

    def UpdateChainLink(Chain, InAngleDelta, LinkIndex, TargetPos, bInEnableRotationLimit, InRotationLimitPerJoints):
        TipBoneLinkIndex = len(Chain) - 1
        CurrentLink = Chain[LinkIndex]
        # Update new tip pos:
        TipPos = Chain[TipBoneLinkIndex]
        ToEnd = TipPos - CurrentLink
        ToTarget = (TargetPos - CurrentLink)
        ToEnd, ToTarget = ToEnd / \
            lng.norm(ToEnd), ToTarget / lng.norm(ToTarget)
        # a = lng.norm(ToEnd-ToTarget) #debug line
        if lng.norm(ToEnd-ToTarget) < 1e-8:
            updated = False
            return Chain, updated, InAngleDelta
        RotationLimitPerJointInRadian = math.radians(
            InRotationLimitPerJoints[LinkIndex])
        Angle = math.acos(np.dot(ToEnd, ToTarget))
        if bInEnableRotationLimit:  # Clamp if Rotation Limit exists
            Angle = ClampAngle(
                Angle, -RotationLimitPerJointInRadian, RotationLimitPerJointInRadian)
        # AngleDelta[LinkIndex] represents "CurrentLink.CurrentAngleDelta"
        bCanRotate = bInEnableRotationLimit == False or RotationLimitPerJointInRadian > InAngleDelta[
            LinkIndex]
        if bCanRotate:
            if bInEnableRotationLimit:
                if RotationLimitPerJointInRadian < Angle + InAngleDelta[LinkIndex]:
                    # Limitation - CurrentAngle -> Toward Maximum Angle
                    Angle = RotationLimitPerJointInRadian - \
                        InAngleDelta[LinkIndex]
            InAngleDelta[LinkIndex] += Angle
            RotationAxis = np.cross(ToEnd, ToTarget)
            # Normalize to Unit Vector.
            RotationAxis = RotationAxis / lng.norm(RotationAxis)
            DeltaRotation = GetQuat(RotationAxis, Angle)
            NewRotation = DeltaRotation  # * CurrentLinkTransform.GetRotation() #
            NewRotation = NewRotation / lng.norm(NewRotation)
            # Chain = SetRotation(Chain, LinkIndex, NewRotation) # Rotation Matrix Version
            # Hamilton Product Version
            Chain = SetRotation_Hamilton(Chain, LinkIndex, NewRotation)
            updated = True
        else:
            updated = False
        return Chain, updated, InAngleDelta

    # Iteration:
    NumChainLinks = len(InOutChain)
    TipBoneLinkIndex = NumChainLinks - 1
    bLocalUpdated = False
    # check how far:
    TargetPos = TargetPosition
    TipPos = InOutChain[TipBoneLinkIndex]
    Distance = lng.norm(TargetPos - TipPos)
    IterationCount = 0
    while Distance > Precision and IterationCount+1 < MaxIteration:
        # UE4 originally iterates from TipIdx-1 -> 1, we iterate till 0 due to the different definition of transform.
        IterationCount += 1
        for LinkIndex in range(TipBoneLinkIndex-1, -1, -1):
            InOutChain, updated, AngleDelta = UpdateChainLink(
                InOutChain, AngleDelta, LinkIndex, TargetPos, bEnableRotationLimit, RotationLimitPerJoints)
            bLocalUpdated = bLocalUpdated or updated
        Distance = lng.norm(InOutChain[TipBoneLinkIndex]-TargetPosition)
        if not bLocalUpdated:
            break
    if EndInd == 0:
        InOutChain = InOutChain[::-1]
    return InOutChain, AngleDelta


class Point_Move:
    showverts = True
    offset = 0.01  # Joint-selection precision

    def __init__(self):
        # 3D Figure & Axes
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        # Title
        self.ax.set_title('Click and drag a point to move it')
        # Axes Setup
        self.ax.set_xlim((-2, 6))
        self.ax.set_ylim((-2, 6))
        self.ax.set_zlim((-2, 6))
        # camera setup
        self.elev, self.azim = 0, 0
        self.azim_adjust = 0
        self.ax.view_init(self.elev, self.azim)
        # Bone chain
        self.x = [0., 0., 0., 0., 0.]
        self.y = [1., 2., 3., 4., 5.]
        self.z = [1., 2., 3., 4., 5.]
        self.chain = np.asarray([self.x, self.y, self.z]).transpose()
        self.angleDelta = np.zeros_like(self.x)
        self.RotationLimitPerJoints = np.ones_like(self.angleDelta) * 90.
        # line3D
        self.line3d = self.ax.plot(self.x, self.y, self.z, color='g')
        self.points3d = self.ax.scatter(self.x, self.y, self.z, marker='o')

        # Current Index selected
        self._ind = None
        # Canvas Setup
        canvas = self.fig.canvas
        canvas.mpl_connect('draw_event', self.draw_callback_3d)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event',
                           self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas = canvas

        # projected coordinates
        self.canvas.draw()
        self.proj_x, self.proj_y, _ = proj3d.proj_transform(
            self.x, self.y, self.z, self.ax.get_proj())
        print(self.proj_x, self.proj_y)
        plt.grid()
        plt.show()

    # Use "<-" "->" keys to rotate the axes.
    def on_key_press(self, event):
        if event.key == 'left':
            self.azim_adjust -= 1
        if event.key == 'right':
            self.azim_adjust += 1
        self.reset_ax(self.azim_adjust)
        self.line3d = self.ax.plot(self.x, self.y, self.z, color='g')
        self.points3d = self.ax.scatter(
            self.x, self.y, self.z, marker='o', color='b')
        self.proj_x, self.proj_y, _ = proj3d.proj_transform(
            self.x, self.y, self.z, self.ax.get_proj())

    def getz(self, x, y):
        # store the current mousebutton
        b = self.ax.button_pressed
        # set current mousebutton to something unreasonable
        self.ax.button_pressed = -1
        # get the coordinate string out
        s = self.ax.format_coord(x, y)
        # set the mousebutton back to its previous state
        self.ax.button_pressed = b
        x_out, y_out, z_out = '', '', ''
        for i in range(s.find('x')+2, s.find('y')-2):
            x_out = x_out+s[i]
        if len(x_out) == 7:
            x_out = float(x_out[1:]) * -1
        for i in range(s.find('y')+2, s.find('z')-2):
            y_out = y_out+s[i]
        if len(y_out) == 7:
            y_out = float(y_out[1:]) * -1
        for i in range(s.find('z')+2, len(s)):
            z_out = z_out+s[i]
        if len(z_out) == 7:
            z_out = float(z_out[1:]) * -1
        x_out, y_out, z_out = float(x_out), float(y_out), float(z_out)
        return x_out, y_out, z_out

    # reset axes
    def reset_ax(self, azim):
        self.ax.clear()
        self.ax.set_title('Click and drag a point to move it')
        self.ax.set_xlim((-2, 6))
        self.ax.set_ylim((-2, 6))
        self.ax.set_zlim((-2, 6))
        self.elev, self.azim = 0, azim
        self.ax.view_init(self.elev, self.azim)

    # update new position
    def draw_callback_3d(self, event):
        # self.reset_ax()
        # self.line3d = self.ax.plot(self.x, self.y, self.z, color = 'g')
        # self.points3d = self.ax.scatter(self.x, self.y, self.z, marker = 'o', color='b')
        # self.proj_x,self.proj_y,_ = proj3d.proj_transform(self.x,self.y,self.z,self.ax.get_proj())
        pass

    def get_ind_under_point_3d(self, event):
        'get the index of the vertex under point if within epsilon tolerance'
        xt, yt = np.asarray(self.proj_x), np.asarray(self.proj_y)
        d = np.sqrt((xt-event.xdata)**2 + (yt-event.ydata)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind] >= self.offset:
            ind = None
        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts:
            return
        if event.inaxes == None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point_3d(event)  # 3d
        print("joint idx:", self._ind)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def motion_notify_callback(self, event):
        'on mouse movement'
        # freeze rotation
        self.elev, self.azim = 0, self.azim_adjust
        self.ax.view_init(self.elev, self.azim)

        if not self.showverts:
            return
        if self._ind is None:
            return
        if self._ind != 0 and self._ind != len(self.x)-1:
            print("Please select tip.")
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        x_out, y_out, z_out = self.getz(event.xdata, event.ydata)
        # x_out = 0 # 2D Version
        self.chain, self.angleDelta = CCDIK(self._ind, self.chain, self.angleDelta, np.asarray(
            [x_out, y_out, z_out]), 0.01, 10, False, self.RotationLimitPerJoints)
        chain_out = self.chain.transpose()
        self.x, self.y, self.z = chain_out[0].tolist(
        ), chain_out[1].tolist(), chain_out[2].tolist()

        self.reset_ax(self.azim_adjust)
        self.line3d = self.ax.plot(self.x, self.y, self.z, color='g')
        self.points3d = self.ax.scatter(
            self.x, self.y, self.z, marker='o', color='b')
        self.proj_x, self.proj_y, _ = proj3d.proj_transform(
            self.x, self.y, self.z, self.ax.get_proj())


if __name__ == '__main__':
    Point_Move()
    print('closed')
