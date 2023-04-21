#------------------------------------------------------------------------------
# Swelling of Unit Sphere (Modeled in 3D)
#------------------------------------------------------------------------------
# Based on the formulation in 2015 paper "Effect of solvent diffusion on
# crack-tip fields and driving force for fracture of hydrogels" which simplifies
# the free energy formulation in a prior 2015 paper, "A nonlinear, transient FE
# method for coupled solvent diffusion and large deformation of hydrogels"

from dolfin import *                    # Dolfin module
from mshr import *
from ufl import cofac, rank
import matplotlib.pyplot as plt         # Module matplotlib for plotting
import numpy as np
import time

start_time = time.time()

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

# Solver parameters: Using PETSc SNES solver
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "symmetric": True,
                          "snes_solver": {"maximum_iterations": 100,
                                          "report": True,
                                          "line_search": "bt",
                                          "linear_solver": "mumps",
                                          "method": "newtonls",
                                          "absolute_tolerance": 1e-9,
                                          "relative_tolerance": 1e-9,
                                          "error_on_nonconvergence": True}}

# Defining Classes
#------------------------------------------------------------------------------
# Initial condition (IC) class for displacement and chemical potential
class InitialConditions(UserExpression):
    def eval(self, values, x):
        # Displacement u0 = (values[0], values[1], values[2])
        values[0] = (l0-1)*x[0]
        values[1] = (l0-1)*x[1]
        values[2] = (l0-1)*x[2]
        values[3] = ChemIni        # Initial Chemical potential: mu0
    def value_shape(self):
         return (4,)

# Full boundary for chemical potential bc
class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Center point for fixed displacement bc
class PinPoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0, tol) and near(x[1], 0.0, tol) and near(x[2], 0.0, tol)

# Right boundary point for fixed displacement bcs (x-axis)
class PinPointRight(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0, tol) and near(x[1], 0.0, tol) and near(x[2], 0.0, tol)

# Front boundary point for fixed displacement bcs (z-axis)
class PinPointTop(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0, tol) and near(x[1], 1.0, tol) and near(x[2], 0.0, tol)

PinPoint = PinPoint()
PinPointRight = PinPointRight()
PinPointTop = PinPointTop()

def PowSpace(Start, Stop, Power, Num):
    Start = np.power(Start, 1/float(Power))
    Stop = np.power(Stop, 1/float(Power))
    return np.power( np.linspace(Start, Stop, num=Num), Power)

# Model parameters
#------------------------------------------------------------------------------
# Set the user parameters, can be parsed from command line
parameters.parse()
UserPar = Parameters("user")
UserPar.add("chi", 0.3)
UserPar.add("GMax", 0.5)
UserPar.add("l0", 2)
UserPar.add("IniTime", 10**(-3))
UserPar.add("MidTime", 10**(-1))
UserPar.add("FinTime", 10**(3))
UserPar.add("TotSurfSteps", 20)
UserPar.add("TotEqm1", 50)
UserPar.add("TotChemSteps", 20)
UserPar.add("TotEqm2", 50)
UserPar.parse()

# Parse from command line if given
tol = 1E-12
chi = UserPar["chi"]           # Flory Parameter
l0 = UserPar["l0"]             # Initial Stretch (lambda_o)
n = 10**(-3)                   # Normalization Parameter (N Omega)
# Global stepping, chemical stepping, and surface stepping parameters
TotSurfSteps = UserPar["TotSurfSteps"]
TotChemSteps = UserPar["TotChemSteps"]
# Number of steps to reach equilibrium for stress or chemical ramping case
TotEqm1, TotEqm2 = UserPar["TotEqm1"], UserPar["TotEqm2"]
# Total number of steps
TotSteps = TotSurfSteps + TotEqm1 + TotChemSteps + TotEqm2
# Body force per unit volume and Traction force on the boundary
B = Constant((0.0, 0.0, 0.0))
T = Constant((0.0, 0.0, 0.0))
# Maximum surface tension
GMax = UserPar["GMax"]
# Chemical potential initial and maximum value
ChemIni = np.log((l0**3-1)/l0**3) + 1/l0**3 + chi/l0**6 + n*(1/l0-1/l0**3)
ChemMid = np.log((l0**3-1)/l0**3) + 1/l0**3 + chi/l0**6 + n*(1/l0-1/l0**3) + n*GMax*2/l0
ChemMax = 0.0

# Set up all stepping arrays
# Surface parameter ramping
SurfStep1 = np.linspace(0, GMax ,TotSurfSteps)
SurfStep2 = np.linspace(GMax, GMax, TotEqm1+TotChemSteps+TotEqm2)
SurfSteps = np.concatenate([SurfStep1, SurfStep2])

# Chemical potential BC ramped from initial to maximum value
ChemStep1 = np.linspace(ChemIni, ChemIni, TotSurfSteps)
ChemStep2 = np.linspace(ChemIni, ChemMid, TotEqm1)
ChemStep3 = np.linspace(ChemMid, ChemMax, TotChemSteps)
ChemStep4 = np.linspace(ChemMax, ChemMax, TotEqm2)
ChemSteps = np.concatenate([ChemStep1, ChemStep2, ChemStep3, ChemStep4])
ChemBC = Expression(("ChemVal"), ChemVal=ChemIni, Step=0, degree=1)

IniTime = UserPar["IniTime"]
MidTime = UserPar["MidTime"]
FinTime = UserPar["FinTime"]
TimeStep1 = PowSpace(IniTime, MidTime, 2, TotSurfSteps + TotEqm1)
TimeStep2 = PowSpace(IniTime, MidTime, 2, TotChemSteps)
TimeStep3 = PowSpace(MidTime, FinTime, 2, TotEqm2)
TimeSteps = np.concatenate([TimeStep1, TimeStep2, TimeStep3])

DT = Expression("dt", dt=0, degree=0)
Gamma = Expression("gamma", gamma=0, degree=0)

# Name of file
SimPar1 = "S_%.0f" % (TotSteps)
SimPar2 = "_chi_%.1e" % (chi)
SimPar3 = "_G_%.0e" % (GMax)
SimPar4 = "_l0_%.1e" % (l0)
SaveDir = "../Results/Sphere/" + SimPar1 + SimPar2 + SimPar3 + SimPar4 + "/"
# write the parameters to file
File(SaveDir + "/parameters.xml") << UserPar

# Define mesh
#------------------------------------------------------------------------------
# Create spherical domain with radius of 1.0
mesh = Mesh("../Gmsh/3DSpherePoints.xml")
# Create subdomains, boundaries, points
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
points = MeshFunction("size_t", mesh, mesh.topology().dim() - 3)

# Mark points of interest
points.set_all(0)
PinPoint.mark(points, 1)
PinPointTop.mark(points, 2)
PinPointRight.mark(points, 3)
File = XDMFFile(SaveDir + "/" + "points.xdmf")
File.write(points)

# Measures/redefinition for dx and ds according to subdomains and boundaries
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)

# Define mixed function space
#------------------------------------------------------------------------------
# Tensor space for projection of stress
T_DG0 = TensorFunctionSpace(mesh,'DG',0)
DG0 = FunctionSpace(mesh,'DG',0)
# Define Taylor-Hood Elements
# Second order quadratic interpolation for displacement (u)
V_CG2 = VectorFunctionSpace(mesh, "Lagrange", 2)
# First order linear interpolation for chemical potential
CG1 = FunctionSpace(mesh, "Lagrange", 1)
# Use ufl_element() to return the UFL element of the function spaces
V_CG2elem = V_CG2.ufl_element()
CG1elem = CG1.ufl_element()
# Define mixed function space specifying underying finite element
V = FunctionSpace(mesh, V_CG2elem * CG1elem)

# Define functions in mixed function space V
#------------------------------------------------------------------------------
du = TrialFunction(V)                       # Incremental trial function
v = TestFunction(V)                         # Test Function
w = Function(V)                             # Current solution for u and mu
w0 = Function(V)                            # Previous solution for u and mu

# Split test functions and unknowns (produces a shallow copy not a deep copy)
(v_u, v_mu) = split(v)
(u, mu) = split(w)                          # Split current
(u0, mu0) = split(w0)                       # Split previous

# Boundary Conditions (BC)
#------------------------------------------------------------------------------
# Displacement BC: pinned center to prevent translation
uFix = Expression(("0.0","0.0","0.0"), degree=0)
uDir = Expression(("0.0"), degree=0)

# The Dirichlet BCs are specified in respective subspaces
# Fixed center point to prevent translations
bc_0 = DirichletBC(V.sub(0), uFix, PinPoint, method='pointwise')
# Fixed right point and front point to prevent rotations
bc_pin_RY = DirichletBC(V.sub(0).sub(1), uDir, PinPointRight, method='pointwise')
bc_pin_RZ = DirichletBC(V.sub(0).sub(2), uDir, PinPointRight, method='pointwise')
bc_pin_FX = DirichletBC(V.sub(0).sub(0), uDir, PinPointTop, method='pointwise')
bc_pin_FZ = DirichletBC(V.sub(0).sub(2), uDir, PinPointTop, method='pointwise')
# Chemical Boundary conditions
bc_chem = DirichletBC(V.sub(1), ChemBC, OnBoundary())

# Combined boundary conditions
bc = [bc_0, bc_pin_RY, bc_pin_RZ, bc_pin_FX, bc_pin_FZ, bc_chem]

# Initial Conditions (IC)
#------------------------------------------------------------------------------
# Initial conditions are created by using the class defined and then
# interpolating into a finite element space
init = InitialConditions(degree=1)          # Expression requires degree def.
w.interpolate(init)                         # Interpolate current solution
w0.interpolate(init)                        # Interpolate previous solution

# Kinematics
#------------------------------------------------------------------------------
d = len(u)                      # Spatial dimension
I = Identity(d)                 # Identity tensor
F = I + grad(u)                 # Deformation gradient from current time step
F0 = I + grad(u0)               # Deformation gradient from previous time step
CG = F.T*F                      # Right Cauchy-Green (CG) tensor

# Invariants of deformation tensors
Ic = tr(CG)                     # First invariant
J = det(F)                      # Current time step for third invariant
J0 = det(F0)                    # Previous time step for third invariant

# Define terms for surface tension
N = FacetNormal(mesh)                    # Normal vector in the reference configuration
NansonOp = (cofac(F))                    # Element of area transformation operator
deformed_N = dot(NansonOp,N)             # Surface element vector in the deformed configuration
Jsurf = sqrt(dot(deformed_N,deformed_N)) # Norm of the surface element vector in the current configuration

# Definitions
#------------------------------------------------------------------------------
# Normalized nominal stress tensor: P = dU/dF
def P(u, mu):
    return F + (-1/J + (1/n)*(1/J + ln((J-1)/J) + chi/(J**2) - mu))*J*inv(F.T)

# Normalized flux
def Flux(u, mu):
    Part1 = dot(inv(F), grad(mu))
    return -(J-1)*dot(Part1,inv(F))

# Variational Problem where we have two equations for the weak form
F0 = inner(P(u, mu), grad(v_u))*dx - inner(T, v_u)*ds - dot(B, v_u)*dx
F1 = (1/n)*((J-1)*v_mu*dx - (J0-1)*v_mu*dx - DT*dot(Flux(u, mu), grad(v_mu))*dx)
# Surface Energy Density
SurfEnergyDen = Gamma*Jsurf
SurfEnergy = SurfEnergyDen*ds
F2 = derivative(SurfEnergy, u, v_u)
WF = F0 + F1 + F2       # Total weak form

# Compute directional derivative about w in the direction of du (Jacobian)
Jacobian = derivative(WF, w, du)

# SNES solver > Setup Non-linear variational problem
Problem = NonlinearVariationalProblem(WF, w, bc, J=Jacobian)
SolverProb = NonlinearVariationalSolver(Problem)
SolverProb.parameters.update(snes_solver_parameters)

# Save results to an .xdmf file since we have multiple fields (time-dependence)
File = XDMFFile(SaveDir + "/Result.xdmf")
File.parameters["flush_output"] = True
File.parameters["functions_share_mesh"] = True # Parameters will share the same mesh

# Postprocessing
Points = np.linspace(0, 1.0, 100)         # Points along the profile
ChemArray = np.zeros((1, len(Points)))    # Array for storing Chemical potential along profile
ProfChem = np.zeros((TotSteps, len(Points)))
PostProc = np.zeros((TotSteps, 6))

# Loop: solve for each value using the previous solution as a starting point
#------------------------------------------------------------------------------
t = 0   # Initialization of time
for (Step, Surf) in enumerate(SurfSteps):
    # Print outs to track code progress
    if MPI.rank(MPI.comm_world) == 0:
        print("\033[1;32m--- Step {0:2d} at time {1:.2e}: Surf = {2:.3f}, Chem = {3:.3f} ---\033[1;m".format(Step, t, Surf, ChemSteps[Step]))

    # Solve for each step
    SolverProb.solve()
    # Deep copy not a shallow copy like split(w) for writing the results to file
    (u, mu) = w.split()

    # Project to proper function spaces
    PTensor = project(P(u, mu), T_DG0)
    JScalar = project(J, DG0)
    CScalar = project((J-1), DG0)

    # Rename results for visualization in Paraview
    u.rename("Displacement", "u")
    mu.rename("Chemical Potential", "mu")
    PTensor.rename("Nominal Stress", "P")
    JScalar.rename("Jacobian", "JScalar")
    CScalar.rename("Concentration", "CScalar")

    # Write multiple time-dependent parameters to .xdmf results file
    File.write(u,t)
    File.write(mu,t)
    File.write(PTensor,t)
    File.write(JScalar,t)
    File.write(CScalar,t)

    # Postprocessing
    #---------------------------------------------------------------------------
    PostProc[Step, :] = np.array([Step, TimeSteps[Step], t, Surf, ChemSteps[Step], u(0,0,1)[0]])
    for (Ind, Val) in enumerate(Points):
        ChemArray[:,Ind] = mu(0,Val,0)
    ProfChem[Step,:] = ChemArray[:]
    # Save text files for post processing
    np.savetxt(SaveDir + '/PostProc.txt', PostProc)
    np.savetxt(SaveDir + '/PostChem.txt', ProfChem)

    # Update all expressions
    DT.dt = TimeSteps[Step]
    Gamma.gamma = Surf
    ChemBC.Step = Step
    ChemBC.ChemVal = ChemSteps[Step]
    t += TimeSteps[Step]                # Update total time for paraview file

    # Update fields containing u and mu and solve using the setup parameters
    w0.vector()[:] = w.vector()

end_time = time.time()
if MPI.rank(MPI.comm_world) == 0:
    print("Total RunTime: " + str(end_time-start_time))
