// Benchmark existing on GMSH tutorials

//Delete All;
Mesh.Algorithm = 2;
Mesh.Algorithm3D = 1;

RSphere = 1.;
lcSphere = 0.2;

// Define all points of the circle
Point(1) = {0,         0,       0, lcSphere};
Point(2) = {RSphere,   0,       0, lcSphere};
Point(3) = {0,         RSphere, 0, lcSphere};
Point(4) = {-RSphere,  0,       0, lcSphere};
Point(5) = {0,        -RSphere, 0, lcSphere};
Point(6) = {0,         0,       -RSphere,lcSphere};
Point(7) = {0,         0,       RSphere,lcSphere};

Circle(1) = {2,1,3};
Circle(2) = {3,1,4};
Circle(3) = {4,1,5};
Circle(4) = {5,1,2};
Circle(5) = {3,1,6};
Circle(6) = {6,1,5};
Circle(7) = {5,1,7};
Circle(8) = {7,1,3};
Circle(9) = {2,1,7};
Circle(10) = {7,1,4};
Circle(11) = {4,1,6};
Circle(12) = {6,1,2};

Line Loop(13) = {2,8,-10};
Surface(14) = {13};
Line Loop(15) = {10,3,7};
Surface(16) = {15};
Line Loop(17) = {-8,-9,1};
Surface(18) = {17};
Line Loop(19) = {-11,-2,5};
Surface(20) = {19};
Line Loop(21) = {-5,-12,-1};
Surface(22) = {21};
Line Loop(23) = {-3,11,6};
Surface(24) = {23};
Line Loop(25) = {-7,4,9};
Surface(26) = {25};
Line Loop(27) = {-4,12,-6};
Surface(28) = {27};

Surface Loop(29) = {28,26,16,14,20,24,22,18};
Volume(1111) = {29};
