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


def FABRIK(EndInd, InOutChain, TargetPosition, Precision, MaxIterations):
    # If drag from the other side:
    if EndInd == 0:
        InOutChain = InOutChain[::-1]
    # Checking Status Boolean, saved for further usage.
    bBoneLocationUpdated = False
    NumChainLinks = len(InOutChain)
    # Compute max reach of bone chain:
    MaximumReach = 0
    BonesLength = [0]  # Put Root-Root Bone inside for easier reference.
    for idx in range(1, NumChainLinks):
        bone = lng.norm(InOutChain[idx]-InOutChain[idx-1])
        BonesLength.append(bone)
        MaximumReach += bone
    RootToTargetDistSq = np.sum((InOutChain[0]-TargetPosition)**2)
    # Bone translation calculation
    # If the effector is further away than the distance from root to tip:
    if RootToTargetDistSq > MaximumReach**2:
        for idx in range(1, NumChainLinks):
            direction = (
                TargetPosition - InOutChain[idx-1]) / lng.norm(TargetPosition - InOutChain[idx-1])
            InOutChain[idx] = InOutChain[idx-1] + direction * BonesLength[idx]
        bBoneLocationUpdated = True
    else:  # Effector is within reach: Standard Calculation
        TipBoneLinkIndex = NumChainLinks - 1
        Slop = lng.norm(InOutChain[TipBoneLinkIndex]-TargetPosition)
        if (Slop > Precision):
            InOutChain[TipBoneLinkIndex] = TargetPosition
            IterationCount = 0
            while (Slop > Precision) and (IterationCount+1 < MaxIterations):
                IterationCount += 1
                # "Forward Reaching" stage - adjust bones from end effector:
                for LinkIndex in range(TipBoneLinkIndex-1, 0, -1):
                    direction = (InOutChain[LinkIndex] - InOutChain[LinkIndex + 1]) / lng.norm(
                        InOutChain[LinkIndex] - InOutChain[LinkIndex + 1])
                    InOutChain[LinkIndex] = InOutChain[LinkIndex +
                                                       1] + direction * BonesLength[LinkIndex+1]
                # "Backward Reaching" stage - adjust bones from root:
                for LinkIndex in range(1, TipBoneLinkIndex):
                    direction = (InOutChain[LinkIndex] - InOutChain[LinkIndex - 1]) / lng.norm(
                        InOutChain[LinkIndex] - InOutChain[LinkIndex - 1])
                    InOutChain[LinkIndex] = InOutChain[LinkIndex -
                                                       1] + direction * BonesLength[LinkIndex]
                # Recheck Slop: the distance between tip location and effector location
                    # Since tip is already kept at effector location, check with its parent bone:
                Slop = np.abs(BonesLength[TipBoneLinkIndex] - lng.norm(
                    InOutChain[TipBoneLinkIndex - 1]-TargetPosition))
            # Place tip:
            direction = (InOutChain[TipBoneLinkIndex] - InOutChain[TipBoneLinkIndex - 1]) / \
                lng.norm(InOutChain[TipBoneLinkIndex] -
                         InOutChain[TipBoneLinkIndex - 1])
            InOutChain[TipBoneLinkIndex] = InOutChain[TipBoneLinkIndex -
                                                      1] + direction * BonesLength[TipBoneLinkIndex]
            bBoneLocationUpdated = True
    # If drag from the other side, reverse it back to original order:
    if EndInd == 0:
        InOutChain = InOutChain[::-1]
    return InOutChain


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
        self.x = [1., 2., 3., 4., 5.]
        self.y = [1., 2., 3., 4., 5.]
        self.z = [1., 2., 3., 4., 5.]
        self.chain = np.asarray([self.x, self.y, self.z]).transpose()
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
        # FABRIK:
        chain_out = FABRIK(self._ind, self.chain, np.asarray(
            [x_out, y_out, z_out]), 0.001, 5)
        self.chain = chain_out
        chain_out = chain_out.transpose()
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
