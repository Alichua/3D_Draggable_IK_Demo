'''
2D version of matplot-based draggable UI
'''
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class Point_Move:
    showverts = True
    offset = 0.1

    def __init__(self):
        # Initialize the figure and axes
        self.fig, self.ax = plt.subplots()
        # Title
        self.ax.set_title('Click and drag a point to move it')
        # Set the range of axes, feel free to modify.
        self.ax.set_xlim((-2, 4))
        self.ax.set_ylim((-2, 4))
        # Initialize the position of points.
        self.x = [1, 2, 3]
        self.y = [1, 2, 3]
        # Visualize the lines between points
        self.line = Line2D(self.x, self.y, ls="-",
                           marker='o', markerfacecolor='r',
                           animated=True)
        self.ax.add_line(self.line)
        # Signal initialized as None
        self._ind = None
        # Set the canvas, responsible for further reaction of events.
        canvas = self.fig.canvas
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event',
                           self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas
        plt.grid()
        plt.show()

    # Re-painting event
    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within tolerance'
        xt, yt = np.array(self.x), np.array(self.y)
        d = np.sqrt((xt-event.xdata)**2 + (yt-event.ydata)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]
        # return None if no tolerable matching detected.
        if d[ind] >= self.offset:
            ind = None
        return ind

    # Mouse click event: return index of nearest vertex.
    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts:
            return
        if event.inaxes == None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    # Mouse release: clean
    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    # Mouse dragging event: update & repainting
    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        # Updating
        x, y = event.xdata, event.ydata
        self.x[self._ind] = x
        self.y[self._ind] = y

        # Re-painting with updated data.
        self.line = Line2D(self.x, self.y, ls="-",
                           marker='o', markerfacecolor='r',
                           animated=True)
        self.ax.add_line(self.line)
        # restore the background.
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


if __name__ == '__main__':
    Point_Move()
