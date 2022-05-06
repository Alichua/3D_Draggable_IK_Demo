# 3D_Draggable_IK_Demo
A naive implementation of basic IK algorithm &amp; visualization via interactive matplot demonstration.

Despite the intuitive names of each file, here's a simple list of explanation over all the files:

1. The 2D_drag.py and 3D_drag.py are two templates of visualizing 2D/3D interactive draggable points. The points can be controlled by mouse-clicking and dragging, and you can rotate the plot with left/right bottom in the 3D version. 

2. All the files containing "IK" suggest the implementation of corresponding IK algorithm, based on 3D_drag.py. The IK algorithms are the simplified version of those implemented in UE4. The CCDIK_2D.py, for example, indicates my failure of realizing the complete function in 3D scenario, for which I build this sudo-3D version to comfort myself.

3. The JacobUtils.py serves the JacobianIK.py, biensur.

4. All those files we discussed above can be run by simply "python something", no argv required. 

5. Use "pip install -r requirements.txt" if necessary. 
