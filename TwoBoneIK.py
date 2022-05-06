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


def TwoBoneIK(joints, end_ind, Effector, JointTarget):
    # Get Initial Status:
    root_ind = len(joints) - end_ind - 1
    RootPos = joints[root_ind]
    JointPos = joints[1]
    EndPos = joints[end_ind]
    LowerLimbLength = lng.norm(EndPos - JointPos)
    UpperLimbLength = lng.norm(JointPos - RootPos)
    MaxLimbLength = LowerLimbLength + UpperLimbLength
    # Vector Preparation:
    DesiredPos = Effector
    DesiredDelta = DesiredPos - RootPos
    DesiredLength = lng.norm(DesiredDelta)
    DesiredDir = DesiredDelta / DesiredLength  # normalized
    # Get Joint Target Plane:
    JointTargetDelta = JointTarget - RootPos
    # save for zero-checking(TODO)
    JointTargetLengthSqr = math.sqrt(lng.norm(JointTargetDelta))
    # perpendicular to plane formed by 2 vectors.
    JointPlaneNormal = np.cross(DesiredDir, JointTargetDelta)
    JointPlaneNormal = JointPlaneNormal / \
        lng.norm(JointPlaneNormal)  # normalized
    JointBendDir = JointTargetDelta - \
        (np.dot(JointTargetDelta, DesiredDir) * DesiredDir)
    JointBendDir = JointBendDir / lng.norm(JointBendDir)  # normalized
    # Get output:
    OutEndPos = DesiredPos
    OutJointPos = JointPos
    if DesiredLength >= MaxLimbLength:
        OutEndPos = RootPos + (MaxLimbLength * DesiredDir)
        OutJointPos = RootPos + (UpperLimbLength * DesiredDir)
    else:
        TwoAB = 2 * UpperLimbLength * DesiredLength
        CosAngle = (UpperLimbLength**2 + DesiredLength**2 -
                    LowerLimbLength**2) / TwoAB  # law of cosine
        SinAngle = math.sin(math.acos(CosAngle))
        # whether the upper arm points the opposite way to DesiredDir
        bReverseUpperBone = (CosAngle < 0)
        JointLineDist = UpperLimbLength * SinAngle
        ProjJointDist = math.sqrt(UpperLimbLength**2 - JointLineDist**2)
        if bReverseUpperBone:
            ProjJointDist *= -1
        OutJointPos = RootPos + \
            (ProjJointDist * DesiredDir) + (JointLineDist * JointBendDir)
        # print(lng.norm(OutEndPos-OutJointPos),lng.norm(OutJointPos-RootPos))
    return OutEndPos, OutJointPos


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
        self.ax.set_xlim((-2, 4))
        self.ax.set_ylim((-2, 4))
        self.ax.set_zlim((-2, 4))
        # camera setup
        self.elev, self.azim = 0, 0
        self.azim_adjust = 0
        self.ax.view_init(self.elev, self.azim)
        # Bone chain
        self.x = [1., 2., 3.]
        self.y = [1., 2., 3.]
        self.z = [1., 2., 3.]
        self.joints = np.asarray([self.x, self.y, self.z]).transpose()
        self.JointTarget = np.asarray([1, 2, 3])
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
        self.ax.set_xlim((-2, 4))
        self.ax.set_ylim((-2, 4))
        self.ax.set_zlim((-2, 4))
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
        if self._ind == 1:
            print("Out joint idx:", self._ind)
        else:
            print("Effector joint idx:", self._ind,
                  "Root joint idx:", 2-self._ind)

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
        if self._ind == 1:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        x_out, y_out, z_out = self.getz(event.xdata, event.ydata)
        Effector = np.asarray([x_out, y_out, z_out])
        OutEndPos, OutJointPos = TwoBoneIK(
            self.joints, self._ind, Effector, self.JointTarget)
        x_end, y_end, z_end = OutEndPos[0], OutEndPos[1], OutEndPos[2]
        x_joint, y_joint, z_joint = OutJointPos[0], OutJointPos[1], OutJointPos[2]
        self.x[self._ind] = x_end
        self.y[self._ind] = y_end
        self.z[self._ind] = z_end
        self.x[1] = x_joint
        self.y[1] = y_joint
        self.z[1] = z_joint
        self.joints = np.asarray([self.x, self.y, self.z]).transpose()

        self.reset_ax(self.azim_adjust)
        self.line3d = self.ax.plot(self.x, self.y, self.z, color='g')
        self.points3d = self.ax.scatter(
            self.x, self.y, self.z, marker='o', color='b')
        self.proj_x, self.proj_y, _ = proj3d.proj_transform(
            self.x, self.y, self.z, self.ax.get_proj())


if __name__ == '__main__':
    Point_Move()
    print('closed')
