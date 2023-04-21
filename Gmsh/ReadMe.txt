How to use gmsh (.geo) files.

In order to obtain an xml file for fenics use:
1) Enter the directory containing the .geo file
2) Enter into the command-line the following command to convert the .geo file
   to a msh file. If there are errors, this process might fail, or might work
   and fail at step 3)
  Name.geo -format msh2 -3

  2.1) As a further point, this method is used to adjust mesh resolution.
       Anywhere the following syntax is used to define a point, the first three
       entries define the x, y, and z coordinates and the last a mesh resolution
       parameter. (smaller value = more fine mesh)

       Point(#) = {0, 0, 0, lcSphere};

3) Enter the following command to convert the msh file to an xml file to be
   read in fenics.
  dolfin-convert Name.msh Name.xml
