# Import FEnicSx/dolfinx
import dolfinx

# For numerical arrays
import numpy as np

# For MPI-based parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
 
# PETSc solvers
from petsc4py import PETSc

# specific functions from dolfinx modules
from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem import (Constant, dirichletbc, Function, functionspace, Expression )
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import VTXWriter, XDMFFile

# specific functions from ufl modules
import ufl
from ufl import (TestFunctions, TrialFunction, Identity, grad, det, exp, tr, dev, inv, sqrt, \
                 dx, inner, derivative, dot, split, outer, pi)


# basix finite elements (necessary for dolfinx v0.8.0)
import basix
from basix.ufl import element, mixed_element, quadrature_element

# Matplotlib for plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# For exporting timehistory results
import csv
import os

# For timing the code
from datetime import datetime


#
log.set_log_level(log.LogLevel.WARNING)

# file_type = "01_Freederickz_transition"
file_type = "02_Bending"
# file_type = "03_Buckling"

# =========================================================
# =====  CREATING A SAMPLE GEOMETRY and MESH   ============
# =========================================================
# A 2-D rectangle
Lx = 0.34e-3 # m
Ly = Lx/10# m
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [Lx, Ly]], [80, 8], mesh.CellType.triangle)

x = ufl.SpatialCoordinate(domain)

# Identify the planar boundaries of the  box mesh
def Left(x):
    return np.isclose(x[0], 0.0)
def LeftCenter(x):
    return np.logical_and(np.isclose(x[0], 0.0) , np.isclose(x[1], Ly/2.0))
def Left_btm(x):
    return np.isclose(x[0], 0) & (x[1] <= Ly / 5 )
def Right(x):
    return np.isclose(x[0], Lx)
def Bottom(x):
    return np.isclose(x[1], 0.0)
def Top(x):
    return np.isclose(x[1], Ly)
def Center(x):
    return np.isclose(x[1], Ly/2.0)

# Mark the sub-domains
boundaries = [(1,Left),(2,Bottom),(3,Right),(4,Top),(5,Left_btm),(6,Center),(7,LeftCenter)]
#
# build collections of facets on each subdomain and mark them appropriately.
facet_indices, facet_markers = [], [] # initalize empty collections of indices and markers.
fdim = domain.topology.dim - 1 # geometric dimension of the facet (mesh dimension - 1)
for (marker, locator) in boundaries:
    facets = mesh.locate_entities(domain, fdim, locator) # an array of all the facets in a 
                                                         # given subdomain ("locator")
    facet_indices.append(facets)                         # add these facets to the collection.
    facet_markers.append(np.full_like(facets, marker))   # mark them with the appropriate index.

# Format the facet indices and markers as required for use in dolfinx.
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
# Add these marked facets as "mesh tags" for later use in BCs.
facet_tags = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

# Define the boundary integration measure "ds" using the facet tags,
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags, metadata={'quadrature_degree':2})

# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
dx = ufl.Measure('dx', domain=domain, metadata={'quadrature_degree': 4})

# Create facet to cell connectivity required to determine boundary facets.
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)
domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)

#  Define facet normal
n = ufl.FacetNormal(domain)


fig_count = 1  # just a counter for figures, so we can obtain a separate figure for each analysed rate




mi_init = Constant(domain, 1.591e6)  # initial equilibrium shear modulus


analysis_type = "theta_fixed"
# analysis_type = "r_fixed"


if analysis_type == "theta_fixed":
    theta_list = [5]        # [deg]     # list of all the considered initial angles of director orientation
    r_list = [1.5, 1.7, 1.9, 2.1, 2.3]  # list of all the considered values of nematic anisotropy parameter
if analysis_type == "r_fixed":
    theta_list = [5, 30, 45, 60]        # list of all the considered initial angles of director orientation
    r_list = [1.5]                      # list of all the considered values of nematic anisotropy parameter


for rate_phi in [1.0]:      # list of all the considered rates of applying the load/voltage/stretch

    for theta_00_deg in theta_list:

        for rrr in r_list:
            
            # DLCE material prameters
            mi_0 = Constant(domain, 1.56e6)                      # elastic branch shear modulus  ----- Povećan za 7.5%
            mi_a = Constant(domain, PETSc.ScalarType(1/16*mi_0)) # anisotropic shear modulus     ----- 
            K_f = Constant(domain,1e-11)                        # Frank constant
            r = Constant(domain, PETSc.ScalarType(rrr))         # nematic anisotropy parameter  ----- Povećan za 12.7%
            a_0 = Constant(domain, 0.08)                        #         ----- Povećan za 21%
            a_p = Constant(domain,0.0)                          #
            m = Constant(domain,0.3)                            #         Smanjen za 18%
            delta = Constant(domain,0.0)                        #

            rho = Constant(domain, PETSc.ScalarType(1100))      # DLCE material mass density    1000 kg/m^3 = 1e-6 kg/mm^3


            scale = mi_init/mi_0                                # 
            uniform_boost = 1.2                                 #

            mi_neq_1 = Constant(domain, PETSc.ScalarType(2.641959*mi_0*scale*uniform_boost))  # noneqilibrium shear modulus of the 1st branch  # Keep fastest mode similar
            tau_1 = Constant(domain, PETSc.ScalarType(0.2))           # relaxation time for the 1st viscoelastic branch   # Fastest relaxation

            mi_neq_2 = Constant(domain, PETSc.ScalarType(1.526881*mi_0*scale*uniform_boost))  # noneqilibrium shear modulus of the 2nd branch  # Slight reduction
            tau_2 = Constant(domain, PETSc.ScalarType(1.0))          # relaxation time for the 1st viscoelastic branch   # Increased from 1.0

            mi_neq_3 = Constant(domain, PETSc.ScalarType(0.838750*mi_0*scale*uniform_boost))  # noneqilibrium shear modulus of the 3rd branch    # Reduced
            tau_3 = Constant(domain, PETSc.ScalarType(6.0))          # relaxation time for the 1st viscoelastic branch   # Increased significantly

            mi_neq_4 = Constant(domain, PETSc.ScalarType(0.383747*mi_0*scale*uniform_boost))  # noneqilibrium shear modulus of the 4th branch    # Reduced further
            tau_4 = Constant(domain, PETSc.ScalarType(42.0))         # relaxation time for the 1st viscoelastic branch   # Much longer timescale

            mi_neq_5 = Constant(domain, PETSc.ScalarType(0.237739*mi_0*scale*uniform_boost))  # noneqilibrium shear modulus of the 5th branch    # Smallest contribution
            tau_5 = Constant(domain, PETSc.ScalarType(336.0))        # relaxation time for the 1st viscoelastic branch   # Very slow relaxation

            # mi_neq_6 = Constant(domain, PETSc.ScalarType(0.15*mi_0*scale*uniform_boost))  # mi_neq₅ × 0.63 ≈ 0.15
            # tau_6 = Constant(domain, PETSc.ScalarType(2500.0))  # τ₅ × 7.44 ≈ 2500s (~42 min)

            Kbulk_0   = Constant(domain, PETSc.ScalarType(1e3*mi_0)) # Bulk modulus, kPa

            # Electro constants
            x0 = Constant(domain,8.85e-12)                           #                   
            epsilon_c = Constant(domain, PETSc.ScalarType(18.5*x0))  #   
            epsilon_a = Constant(domain, PETSc.ScalarType(7*x0))     #


            # Director initial orientation
            teta0 = Constant(domain, PETSc.ScalarType(theta_00_deg*pi/180)) # angle of mesogen orientation
            dd0_init_x = Constant(domain, PETSc.ScalarType(ufl.cos(teta0))) # 
            dd0_init_y = Constant(domain, PETSc.ScalarType(ufl.sin(teta0))) # 
            dd0 = ufl.as_vector([dd0_init_x,dd0_init_y, 0])                 # director - mesogen orientation vector
            
            # Initial Cauchy stress for a plane strain problem
            T11 = fem.Constant(domain, 0.0)  
            T12 = fem.Constant(domain, -1.0) 
            T13 = fem.Constant(domain, 0.0)  
            T21 = fem.Constant(domain,  1.0) 
            T22 = fem.Constant(domain, 0.0)  
            T23 = fem.Constant(domain, 0.0)  
            T31 = fem.Constant(domain,  0.0) 
            T32 = fem.Constant(domain, 0.0)  
            T33 = fem.Constant(domain, 1.0)  
            Tt = ufl.as_tensor([[T11, T12,T13],
                            [T21, T22,T23],
                            [T31, T32,T33]])


            # Numerical analysis parameters
            Vmax = 6000.00 #[V]                            # maximum considered voltage (electric potential)
            phi_norm = float(Vmax/Ly/sqrt(mi_0/epsilon_a))  # normalized electric potential

            trial_fact = 1.0                                # trial factor for preliminary calculations,  takes values between 0 and 1

            t        = 0.0 #[s]                             # start time for the computation
            Ttot     = trial_fact* phi_norm /rate_phi       # end time for the computation
            numSteps = trial_fact*200                       # number of steps in numerical calculation
            dt       = Ttot/numSteps                        # (fixed) step size
            # Create a constant for the time step
            dk = Constant(domain, PETSc.ScalarType(dt))
            
            # Create a function to ramp the voltage
            def phiRamp(t):
                phi = Vmax*(t/Ttot) 
                return phi

            # Generalized-alpha method parameters for calculating acceleration and velocity
            alpha   = Constant(domain, PETSc.ScalarType(0.0))
            gamma   = Constant(domain, PETSc.ScalarType(0.5+alpha))
            beta    = Constant(domain, PETSc.ScalarType(0.25*(gamma+0.5)**2))
            

            # ============================================================================
            # ====  DEFINING THE FINITE ELEMENT TYPES, FUNCTION SPACES AND FUNCTIONS  ====
            # ============================================================================

            # Define function space, both vectorial and scalar
            U2 = element("Lagrange", domain.basix_cell(), 2, shape=(2,))                # For displacement  (2D vector)
            P1 = element("Lagrange", domain.basix_cell(), 1)                            # For pressure  (scalar)
            P0 = quadrature_element(domain.basix_cell(), degree=2, scheme="default")    # For visualization
            T0 = element("DG", domain.basix_cell(), 1, shape=(3, 3))                    # For Cv    (2nd order tensor)
            D1 = element("Lagrange", domain.basix_cell(), 1, shape=(2,))                # For director
            M1 = element("Lagrange", domain.basix_cell(), 1)                            # For mi - inextensibility condition
            TH = mixed_element([U2, P1, P1, T0, T0, T0, T0, T0, P1])                # Taylor-Hood style mixed element
            ME = functionspace(domain,  TH)                                         # Total space for all DOFs
            
            V1 = functionspace(domain, P1) # Scalar function space
            V2 = functionspace(domain, U2) # Vector function space
            T3 = functionspace(domain, T0) # Tensor function space
            
            # Define actual functions with the required DOFs
            w    = Function(ME)
            u, p, teta, Fv_1, Fv_2, Fv_3, Fv_4, Fv_5, q = split(w)  # displacement u, pressure p, dd director vector, mm Lagrange multplier


            w.sub(2).interpolate(lambda x: np.full(x.shape[1], teta0))  

            # A copy of functions to store values in the previous step
            w_old         = Function(ME)
            u_old,  p_old, teta_old, Fv_1_old, Fv_2_old, Fv_3_old, Fv_4_old, Fv_5_old, q_old = split(w_old)  

            # Define test functions        
            u_test, p_test, teta_test, Fv_1_test, Fv_2_test, Fv_3_test, Fv_4_test, Fv_5_test, q_test= TestFunctions(ME)    

            # Define trial functions needed for automatic differentiation
            dw = TrialFunction(ME)    

            # Functions for storing the velocity and acceleration at prev. step
            v_old = Function(V2)
            a_old = Function(V2)
            
            # Initial conditions: 
            # A function for constructing the identity matrix.
            def identity(x):
                values = np.zeros((3*3,
                                x.shape[1]), dtype=np.float64)
                values[0] = 1
                values[4] = 1
                values[8] = 1
                return values
            
            # Interpolate the identity onto the tensor-valued Cv function for all the viscoelastic branches
            w.sub(3).interpolate(identity)  
            w.sub(4).interpolate(identity) 
            w.sub(5).interpolate(identity) 
            w.sub(6).interpolate(identity)
            w.sub(7).interpolate(identity)
            w_old.sub(3).interpolate(identity)  
            w_old.sub(4).interpolate(identity) 
            w_old.sub(5).interpolate(identity)
            w_old.sub(6).interpolate(identity)
            w_old.sub(7).interpolate(identity)


            def safe_sqrt(x):
                return sqrt(x + 1.0e-16)

            # Gradient of vector field u for plane strain
            def pe_grad_vector(u):
                grad_u = grad(u)
                pe_grad_u = ufl.as_tensor([ [grad_u[0,0], grad_u[0,1], 0.0],
                                            [grad_u[1,0], grad_u[1,1], 0.0],
                                            [        0.0,         0.0, 0.0] ]) 
                return pe_grad_u
            
            # (just need an extra zero for dimensions to work out)
            def pe_grad_scalar(y):
                grad_y = grad(y)
                pe_grad_y = ufl.as_vector([grad_y[0], grad_y[1], 0.])
                return pe_grad_y
            
            # Axisymmetric deformation gradient 
            def F_pe_calc(u):
                dim = len(u)                # dimension of problem (2)
                Id = Identity(dim)          # 2D Identity tensor
                F = Id + grad(u)            # 2D Deformation gradient
                F_pe =  ufl.as_tensor([ [F[0,0], F[0,1], 0.0],
                                        [F[1,0], F[1,1], 0.0],
                                        [   0.0,    0.0, 1.0]]) # Full plane strain F
                return F_pe

            # subroutine for the distortional part / unimodular part of a tensor A
            def dist_part(A):
                Abar = A / (det(A)**(1.0/3.0))
                return Abar


            #------------------------------------------------------------- 
            # Subroutines for computing the viscous flow update
            #-------------------------------------------------------------

            # Subroutine for computing the effective stretch
            def lambdaBar_calc(u):           
                F = F_pe_calc(u)
                J = det(F)
                Fbar = J**(-1/3)*F
                Cbar = Fbar.T*Fbar
                I1 = tr(Cbar)
                lambdaBar = safe_sqrt(I1/3.0)
                return lambdaBar

            def dd_calc(teta):
                dd  = ufl.as_vector([ufl.cos(teta), ufl.sin(teta), 0])
                return dd

            dd = dd_calc(teta)
 
            def I_4bar_calc(u):
                F = F_pe_calc(u)
                J = det(F)
                C = F.T * F
                Cbar= J**(-2/3)*C
                I_4bar = inner(Cbar, outer(dd0,dd0))
                return I_4bar

            def dd2_calc(teta):
                dd2  = ufl.as_vector([ufl.cos(teta), ufl.sin(teta)])
                return dd2

            dd2 = dd2_calc(teta)

            def l_0_calc(dd0):
                Id = Identity(3) 
                l_0  = (r-1)*outer(dd0,dd0) + Id
                return l_0

            def l_calc(teta):
                Id = Identity(3) 
                dd = dd_calc(teta)
                k = (r - 1)
                l  = k * outer(dd,dd) + Id
                return l

            def g_0_calc(u,teta):
                Id = Identity(3) 
                F = F_pe_calc(u)
                J = det(F)
                dd = dd_calc(teta)
                l = l_calc(teta)
                l_0 = l_0_calc(dd0)
                K_bar = J**(-2/3)*inv(l)*F*l_0*F.T
                g_0  = 1 + 4*(mi_a/mi_0) * (ufl.tr(K_bar)-3)
                return g_0


            def g_0_neq_calc(u, teta, Fv):
                Id = Identity(3) 
                F = F_pe_calc(u)
                J = det(F)
                dd = dd_calc(teta)
                l = l_calc(teta)
                l_0 = l_0_calc(dd0)
                K_bar = J**(-2/3)*inv(l)*(F*inv(Fv))*l_0*(inv(Fv.T)*F.T)
                g_0_neq  = 1 + 4*(mi_a/mi_0) * (ufl.tr(K_bar)-3)
                return g_0_neq

            def n_f_norm2bar_calc(u, teta):
                Id = Identity(3) 
                F   = F_pe_calc(u)
                J = det(F)
                dd = dd_calc(teta)
                n_f_norm2bar  = inner(J**(-2/3)*outer(dd,dd), F*(Id-outer(dd0,dd0))*F.T)
                return n_f_norm2bar

            def n_f_norm2bar_neq_calc(u, teta, Fv):
                Id = Identity(3) 
                F   = F_pe_calc(u)
                J = det(F)
                dd = dd_calc(teta)
                n_f_norm2bar_neq  = J**(-2/3)*tr(outer(dd,dd)*F*inv(Fv)*(Id-outer(dd0,dd0))*(inv(Fv.T)*F.T))
                return n_f_norm2bar_neq

            def a_tbar_calc(u):
                Id = Identity(3) 
                F = F_pe_calc(u)
                J = det(F)
                a_tbar  = a_0 / (m*(inner(J**(-2/3)*F,F)-3)+1)
                return a_tbar

            def a_tbar_neq_calc(u, Fv):
                Id = Identity(3) 
                F = F_pe_calc(u)
                J = det(F)
                a_tbar  = a_0 / (m*(inner(J**(-2/3)*F*inv(Fv),F*inv(Fv))-3)+1)
                return a_tbar

            def abar_calc(u):
                a_tbar = a_tbar_calc(u)
                abar  = a_tbar + a_p
                return abar

            def abar_neq_calc(u, Fv):
                a_tbar_neq = a_tbar_neq_calc(u, Fv)
                abar_neq  = a_tbar_neq + a_p
                return abar_neq

            def K0bar_calc(u, teta):
                Id = Identity(3) 
                F   = F_pe_calc(u)
                J = det(F)
                l = l_calc(teta)
                l_0 = l_0_calc(dd0)
                K_bar = J**(-2/3)*inv(l)*F*l_0*F.T
                K0bar  = K_bar - 1/3*tr(K_bar)*Id
                return K0bar

            def K0bar_neq_calc(u, teta, Fv):
                Id = Identity(3) 
                F   = F_pe_calc(u)
                J = det(F)
                l = l_calc(teta)
                l_0 = l_0_calc(dd0)
                K_bar = J**(-2/3)*inv(l)*F*inv(Fv)*l_0*(F*inv(Fv)).T
                K0bar_neq  = K_bar - 1/3*tr(K_bar)*Id
                return K0bar_neq

            def B0bar_calc(u):
                Id = Identity(3) 
                F   = F_pe_calc(u)
                J = det(F)
                B = F*F.T
                B0bar  = J**(-2/3)*(B-1/3*tr(B)*Id)
                return B0bar

            def B0bar_neq_calc(u, Fv):
                Id = Identity(3) 
                F   = F_pe_calc(u)
                J = det(F)
                Cv_inv = inv(Fv.T*Fv)
                B0bar_neq  = J**(-2/3)*(F*Cv_inv*F.T-1/3*tr(F*Cv_inv*F.T)*Id)
                return B0bar_neq

            def Z_bar_calc(u, teta):
                Id = Identity(3) 
                F   = F_pe_calc(u)
                dd = dd_calc(teta)
                J = det(F)
                Z_bar  = J**(-2/3)*(outer(dd,dd))*F*(Id-outer(dd0,dd0))*F.T
                return Z_bar

            def Z_bar_neq_calc(u, teta, Fv):
                Id = Identity(3) 
                F   = F_pe_calc(u)
                dd = dd_calc(teta)
                J = det(F)
                Z_bar_neq  = J**(-2/3)*(outer(dd,dd))*F*inv(Fv)*(Id-outer(dd0,dd0))*inv(Fv.T) * F.T
                return Z_bar_neq

            def Z0bar_calc(u, teta):
                Id = Identity(3) 
                F   = F_pe_calc(u)
                J = det(F)
                dd = dd_calc(teta)
                Z_bar = Z_bar_calc(u, teta)
                Z0bar  = Z_bar - 1/3*tr(Z_bar)*Id
                return Z0bar

            def Z0bar_neq_calc(u, teta, Fv):
                Id = Identity(3) 
                F   = F_pe_calc(u)
                J = det(F)
                dd = dd_calc(teta)
                Z_bar_neq = Z_bar_neq_calc(u, teta, Fv)
                Z0bar_neq  = Z_bar_neq - 1/3*tr(Z_bar_neq)*Id
                return Z0bar_neq


            def Fv_update(u, teta, Fv, Fv_old, tau_r):
                F = F_pe_calc(u)
                abar_neq = abar_neq_calc(u, Fv)
                a_tbar_neq = a_tbar_neq_calc(u, Fv)
                J = det(F)
                C = F.T * F
                g_0 = g_0_calc(u, teta)
                g_0_neq = g_0_neq_calc(u, teta, Fv)
                n_f_norm2bar_neq = n_f_norm2bar_neq_calc(u, teta, Fv)
                K0bar_neq = K0bar_neq_calc(u, teta, Fv)
                B0bar_neq = B0bar_neq_calc(u, Fv)
                Z0bar_neq = Z0bar_neq_calc(u, teta, Fv)
                Cv = Fv.T*Fv   # Isochoric right Cauchy-Green
                # Compute required terms
                # Add small regularization to avoid singular inverses
                inv_Fv = inv(Fv)
                # Stabilized residual formulation
                residual = (Fv - Fv_old)/dk - (1/(tau_r)) * inv(Fv.T)*F.T*(g_0_neq * K0bar_neq - \
                                    (m/a_0)*a_tbar_neq**2* n_f_norm2bar_neq*B0bar_neq + abar_neq*Z0bar_neq)*inv(F.T)*Cv
                return residual 

            def Dn_calc(teta):
                Id = Identity(3)  # Ensure Id is a 2x2 identity matrix
                dd = dd_calc(teta)
                Dn = (epsilon_c - epsilon_a) * outer(dd, dd) + epsilon_a * Id
                return Dn

            def D_eld_calc(u, teta, q):
                Id = Identity(3) 
                F =  F_pe_calc(u)
                dd = dd_calc(teta)
                J = det(F)
                Dn_tensor = Dn_calc(teta)
                D_eld  = -J*inv(F)*Dn_tensor*inv(F.T)*pe_grad_scalar(q)
                return D_eld

            def T_R_MW_calc(u, teta, q):
                Id = Identity(3) 
                F  =  F_pe_calc(u)
                dd = dd_calc(teta)
                J  = det(F)
                Dn = Dn_calc(teta)
                D_eld = D_eld_calc(u, teta, q)
                T_R_MW = inv(J)*(inv(Dn) * outer(F*D_eld, F*D_eld)-1/2*(inner(inv(Dn),outer(F*D_eld, F*D_eld)))*Id)*inv(F.T)
                return  T_R_MW


            # Subroutine for calculating the  equilibrium Cauchy stress
            def T_R_nceq_calc(u,p,teta):
                Id = Identity(3) 
                B0bar = B0bar_calc(u)
                F = F_pe_calc(u)
                J = det(F)
                Fbar = J**(-1/3)*F
                abar = abar_calc(u)
                n_f_norm2bar = n_f_norm2bar_calc(u, teta)
                K0bar = K0bar_calc(u, teta)
                Z0bar = Z0bar_calc(u, teta)
                l_0 = l_0_calc(dd0)
                l = l_calc(teta)
                g_0 = g_0_calc(u,teta)
                a_tbar = a_tbar_calc(u)
                dd = dd_calc(teta)
                I_4bar = I_4bar_calc(u)

                T_R_nceq = mi_0 * (g_0*K0bar + abar*Z0bar - \
                    n_f_norm2bar*a_tbar**2*(m/a_0)*B0bar)*inv(F.T) - p*J*inv(F.T)
                return T_R_nceq

            # Subroutine for the non-equilibrium Cauchy stress.
            def T_R_neq_calc(u, Fv, mi_neq):
                Id = Identity(3) 
                F = F_pe_calc(u)
                abar_neq = abar_neq_calc(u, Fv)
                a_tbar_neq = a_tbar_neq_calc(u, Fv)
                J = det(F)
                C = F.T * F
                g_0_neq = g_0_neq_calc(u, teta, Fv)
                n_f_norm2bar_neq = n_f_norm2bar_neq_calc(u, teta, Fv)
                K0bar_neq = K0bar_neq_calc(u, teta, Fv)
                B0bar_neq = B0bar_neq_calc(u, Fv)
                Z0bar_neq = Z0bar_neq_calc(u, teta, Fv)
                Cv = Fv.T*Fv   # Isochoric right Cauchy-Green
                # Compute required terms
                # Add small regularization to avoid singular inverses
                inv_Fv = inv(Fv)
                # Stabilized residual formulation
                F  = F_pe_calc(u)
                F_bar = J**(-1/3)*F
                T_R_neq = mi_neq*(g_0_neq * K0bar_neq -\
                                    (m/a_0)*a_tbar_neq**2* n_f_norm2bar_neq*B0bar_neq + abar_neq*Z0bar_neq)*inv(F.T)
                return T_R_neq

            def T_R_frank_calc(u, teta):
                    Id = Identity(3) 
                    F  =  F_pe_calc(u)
                    J  = det(F)
                    dd = dd_calc(teta)
                    H = pe_grad_vector(dd)
                    l = l_calc(teta)
                    T_R_frank = (J*K_f/2) * (inner(H*inv(F), H*inv(F))*Id - 2*inv(F.T)*H.T*H*inv(F)) * inv(F.T)
                    return T_R_frank
                
            def T_R_mod_calc(u, teta):
                    Id = Identity(3) 
                    F = F_pe_calc(u)
                    n_f_norm2bar = n_f_norm2bar_calc(u, teta)
                    Z0bar = Z0bar_calc(u, teta)
                    l = l_calc(teta)
                    T_R_mod = delta * mi_0 * n_f_norm2bar * Z0bar * inv(F.T) 
                    return T_R_mod

            def Piola_nceqneq_calc(u, p, teta, Fv_1, Fv_2, Fv_3, Fv_4, Fv_5, mi_neq_1, mi_neq_2, mi_neq_3,mi_neq_4,mi_neq_5, q):
                F = F_pe_calc(u)
                J = det(F)
                T_R_mod = T_R_mod_calc(u, teta)
                T_R_frank = T_R_frank_calc(u, teta)
                T_R_nceq = T_R_nceq_calc(u,p,teta)
                T_R_neq_1 =T_R_neq_calc(u, Fv_1, mi_neq_1)
                T_R_neq_2 = T_R_neq_calc(u, Fv_2, mi_neq_2)
                T_R_neq_3 = T_R_neq_calc(u, Fv_3, mi_neq_3)
                T_R_neq_4 = T_R_neq_calc(u, Fv_4, mi_neq_4)
                T_R_neq_5 = T_R_neq_calc(u, Fv_5, mi_neq_5)
                T_R_MW = T_R_MW_calc(u, teta, q)
                Piola_nceqneq = T_R_nceq + T_R_neq_1 + T_R_neq_2 +T_R_neq_3 + T_R_neq_4 + T_R_neq_5 + T_R_MW + T_R_mod + T_R_frank
                return Piola_nceqneq

            #---------------------------------------------------------------------
            # Subroutine for updating  acceleration using the Newmark beta method:
            # a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
            #---------------------------------------------------------------------
            def update_a(u, u_old, v_old, a_old):
                return (u-u_old-dk*v_old)/beta/dk**2 - (1-2*beta)/2/beta*a_old

            #---------------------------------------------------------------------
            # Subroutine for updating  velocity using the Newmark beta method
            # v = dt * ((1-gamma)*a0 + gamma*a) + v0
            #---------------------------------------------------------------------
            def update_v(a, u_old, v_old, a_old):
                return v_old + dk*((1-gamma)*a_old + gamma*a)

            #---------------------------------------------------------------------
            # alpha-method averaging function
            #---------------------------------------------------------------------
            def avg(x_old, x_new, alpha):
                return alpha*x_old + (1-alpha)*x_new

            def Sigma_Frank_calc(u, teta):
                Id = Identity(3) 
                F  =  F_pe_calc(u)
                J  = det(F)
                Sigma_Frank = J*K_f*inv(F)*inv(F.T)*pe_grad_scalar(teta)
                return  Sigma_Frank

            def m_nc_calc(u, teta): 
                Id = Identity(3) 
                F  =  F_pe_calc(u)
                J = det(F)
                dd = dd_calc(teta)
                abar = abar_calc(u)
                Fbar = J**(-1/3)*F
                g_0 = g_0_calc(u,teta)
                l_0 = l_0_calc(dd0)
                k = 1/r - 1
                m_nc = mi_0*(g_0*k*Fbar*l_0*Fbar.T*dd + abar * Fbar * (Id - outer(dd0,dd0))*Fbar.T*dd)
                return  m_nc

            def m_neq_calc(u, teta, Fv, mi_neq): 
                Id = Identity(3) 
                F = F_pe_calc(u)
                dd = dd_calc(teta)
                abar_neq = abar_neq_calc(u, Fv)
                a_tbar_neq = a_tbar_neq_calc(u, Fv)
                J = det(F)
                g_0_neq = g_0_neq_calc(u, teta, Fv)
                B0bar_neq = B0bar_neq_calc(u, Fv)
                Z0bar_neq = Z0bar_neq_calc(u, teta, Fv)
                Cv = Fv.T*Fv   # Isochoric right Cauchy-Green
                l = l_calc(teta)
                l_0 = l_0_calc(dd0)
                k = 1/r - 1
                K_bar_neq = J**(-2/3)*inv(l)*F*inv(Fv)*l_0*(F*inv(Fv)).T
                Z2_bar_neq  = J**(-2/3)*F*inv(Fv)*(Id-outer(dd0,dd0))*inv(Fv.T) * F.T
                m_neq = mi_neq*(g_0_neq * k * l * K_bar_neq * dd + abar_neq * Z2_bar_neq *dd )
                return  m_neq

            def m_frank_calc(u, teta): 
                F  =  F_pe_calc(u)
                J = det(F)
                m_frank = J*K_f*(inner(inv(F)*inv(F.T), outer(pe_grad_scalar(teta),pe_grad_scalar(teta)) ))*dd
                return  m_frank

            def m_elec_calc(u, teta, q): 
                F  =  F_pe_calc(u)
                J = det(F)
                D_eld = D_eld_calc(u, teta, q)
                b = - (epsilon_c - epsilon_a)/(epsilon_c * epsilon_a)
                m_elec = b/J*outer(F*D_eld, F*D_eld)*dd
                return  m_elec

            def m_mod_calc(u, teta): 
                Id = Identity(3) 
                n_f_norm2bar = n_f_norm2bar_calc(u, teta)   
                F  =  F_pe_calc(u)
                dd = dd_calc(teta)
                Fbar = J**(-1/3)*F
                m_mod = delta * mi_0 * n_f_norm2bar * Fbar * (Id - outer(dd0,dd0))*Fbar.T*dd
                return  m_mod

            def m_tot_calc(u, teta, Fv_1, Fv_2, Fv_3, Fv_4, Fv_5, mi_neq_1, mi_neq_2, mi_neq_3, mi_neq_4, mi_neq_5, q): 
                m_frank  =  m_frank_calc(u, teta)
                m_nc = m_nc_calc(u, teta)
                m_mod = m_mod_calc(u, teta)
                m_neq_1 = m_neq_calc(u, teta, Fv_1, mi_neq_1)
                m_neq_2 = m_neq_calc(u, teta, Fv_2, mi_neq_2)
                m_neq_3 = m_neq_calc(u, teta, Fv_3, mi_neq_3)
                m_neq_4 = m_neq_calc(u, teta, Fv_4, mi_neq_4)
                m_neq_5 = m_neq_calc(u, teta, Fv_5, mi_neq_5)
                m_elec = m_elec_calc(u, teta,q)
                pi  = - (m_frank + m_nc + m_neq_1 + m_neq_2 + m_neq_3 + m_neq_4 + m_neq_5 + m_elec + m_mod) 
                m_tot = inner(pi, Tt*dd)
                return  m_tot

            #Evaluate kinematics and constitutive relations
            # Get acceleration and velocity at end of step
            a_new = update_a(u, u_old, v_old, a_old)
            v_new = update_v(a_new, u_old, v_old, a_old)

            # get avg (u,p) fields for generalized-alpha method
            u_avg  = avg(u_old, u, alpha)
            p_avg  = avg(p_old, p, alpha)
            teta_avg  = avg(teta_old, teta, alpha)
            q_avg  = avg(q_old, q, alpha)


            # Kinematical quantities
            F  = F_pe_calc(u_avg)
            J  = det(F)
            lambdaBar = lambdaBar_calc(u_avg)

            residual_1 = Fv_update(u_avg, teta, Fv_1, Fv_1_old, tau_1)  
            residual_2 = Fv_update(u_avg, teta, Fv_2, Fv_2_old, tau_2) 
            residual_3 = Fv_update(u_avg, teta, Fv_3, Fv_3_old, tau_3)
            residual_4 = Fv_update(u_avg, teta, Fv_4, Fv_4_old, tau_4)
            residual_5 = Fv_update(u_avg, teta, Fv_5, Fv_5_old, tau_5)


            # Referential configuration quantities
            m_tot = m_tot_calc(u_avg, teta_avg, Fv_1, Fv_2, Fv_3, Fv_4, Fv_5, mi_neq_1, mi_neq_2, mi_neq_3, mi_neq_4, mi_neq_5, q_avg)
            Sigma_Frank = Sigma_Frank_calc(u_avg, teta_avg)
            Piola_tot = Piola_nceqneq_calc(u_avg, p_avg, teta_avg, Fv_1, Fv_2, Fv_3, Fv_4, Fv_5, mi_neq_1, mi_neq_2, mi_neq_3, mi_neq_4, mi_neq_5, q_avg)
            D_eld = D_eld_calc(u_avg, teta_avg, q_avg)


            # --------------------------------------------------
            # --------   WEAK FORM FORMULATION   ---------------
            # --------------------------------------------------

            # The weak form for the equilibrium equation 
            Res_0_LCE = inner(Piola_tot, pe_grad_vector(u_test))*dx + inner(rho*a_new, u_test)*dx 

            # The weak form for the angular mometum equation.  
            Res_1_LCE = inner(-Sigma_Frank, pe_grad_scalar(teta_test))*dx + m_tot*teta_test*dx

            # The weak form for the pressure 
            Res_2_LCE =  inner((J-1) + p_avg/Kbulk_0, p_test)*dx


            Res_3 =  inner(residual_1, Fv_1_test)*dx
            Res_4 =  inner(residual_2, Fv_2_test)*dx
            Res_5 =  inner(residual_3, Fv_3_test)*dx
            Res_6 =  inner(residual_4, Fv_4_test)*dx
            Res_7 =  inner(residual_5, Fv_5_test)*dx



            Res_8 = inner(pe_grad_scalar(q_test) , D_eld)*dx 



            Res_LCE = Res_0_LCE +  Res_1_LCE  +  Res_2_LCE + Res_3 + Res_4 + Res_5 + Res_6 + Res_7 + Res_8

            # Automatic differentiation tangent:
            a_LCE = derivative(Res_LCE, w, dw)


            # -------------------------------------------------------
            # ----  RESULTS OUTPUT AND VISUALIZATION PARAMETERS  ----
            # -------------------------------------------------------
            # results file name
            results_name = "2D_DLCE_bending__r={}__".format(rrr)

            # # Function space for projection of results
            P1 = element("Lagrange", domain.basix_cell(), 1)
            VV1 = fem.functionspace(domain, P1) # linear scalar function space
            
            U1 = element("Lagrange", domain.basix_cell(), 1, shape=(2,)) 
            VV2 = fem.functionspace(domain, U1) # linear Vector function space
            #
            T1 = element("Lagrange", domain.basix_cell(), 1, shape=(3,3)) 
            VV3 = fem.functionspace(domain, T1) # linear tensor function space
            # For visualization purposes, we need to re-project the stress tensor onto a linear function space before 
            # we write it (and its components and the von Mises stress, etc) to the VTX file. 
            #
            # This is because the stress is a complicated "mixed" function of the (quadratic Lagrangian) displacements
            # and the (quadrature representation) plastic strain tensor and scalar equivalent plastic strain. 
            #
            # First, define a function for setting up this kind of projection problem for visualization purposes:
            def setup_projection(u, V):

                trial = ufl.TrialFunction(V)
                test  = ufl.TestFunction(V)   

                a = ufl.inner(trial, test)*dx
                L = ufl.inner(u, test)*dx

                projection_problem = dolfinx.fem.petsc.LinearProblem(a, L, [], \
                    petsc_options={"ksp_type": "cg", "ksp_rtol": 1e-16, "ksp_atol": 1e-16, "ksp_max_it": 1000})
                
                return projection_problem
                # Create a linear problem for projecting the stress tensor onto the linear tensor function space VV3.
            #
            tensor_projection_problem = setup_projection(Piola_tot, T3)
            Piola_temp = tensor_projection_problem.solve()

            # fields to write to output file
            u_vis = Function(VV2)
            u_vis.name = "disp"

            teta_vis = Function(VV1)
            teta_vis.name = "teta"

            phi_vis = Function(VV1)
            phi_vis.name = "phi"

            # Mises stress
            T     = Piola_temp*F.T/J
            T0    = T - (1/3)*tr(T)*Identity(3)
            Mises = sqrt((3/2)*inner(T0, T0))
            Mises_vis= Function(VV1,name="Mises")
            Mises_expr = Expression(Mises,VV1.element.interpolation_points())

            dd_vis = Function(VV2, name="dd")
            dd_expr = Expression(dd2, VV2.element.interpolation_points())

            # set up the output VTX files.
            file_results = VTXWriter(
                MPI.COMM_WORLD,
                file_type+"/"+"results/rate={}/".format(rate_phi) + results_name + ".bp",
                [  # put the functions here you wish to write to output
                    u_vis, teta_vis, phi_vis, Mises_vis, dd_vis, # P11, P22, P33, J_vis, P11_LCE, P22_LCE, P33_LCE, lambdaBar_vis,
                    # Mises_vis,
                ],
                engine="BP4",
            )

            def writeResults(t):
                u_vis.interpolate(w.sub(0))
                teta_vis.interpolate(w.sub(2))
                phi_vis.interpolate(w.sub(8))
                Piola_temp = tensor_projection_problem.solve()
                Mises_vis.interpolate(Mises_expr)
                dd_vis.interpolate(dd_expr)
                # P11.interpolate(P11_expr)
                # P22.interpolate(P22_expr)
                # P33.interpolate(P33_expr)
                # Write output fields
                file_results.write(t) 
            

            # Infrastructure for pulling out time history data 
            pointForDisp = np.array([[Lx, Ly, 0.0]]) 

            n_3d = ufl.as_vector([n[0], n[1], 0])  # Extend to 3D

            # Computing the reaction force using the stress field
            traction = dot(Piola_temp, n_3d)
            Force    = dot(traction, n_3d)*ds(3)
            rxnForce = fem.form(Force)

            pointForEval = np.array([Lx/2, Ly/2, 0])
            bb_tree = dolfinx.geometry.bb_tree(domain,domain.topology.dim)
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, pointForEval)

            # fenicsx v0.8.0 syntax:
            colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, pointForEval).array

            # Give the step a descriptive name
            step = "Voltage"
            # Constant for applied load/voltage/displacement
            phi_cons = Constant(domain,PETSc.ScalarType(phiRamp(0)))

            # Find the specific DOFs which will be constrained. - DISPLACEMENT 
            Left_u1_dofs = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(1))
            Left_u2_dofs = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(1))
            Right_u1_dofs = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(3))
            Right_u2_dofs = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(3))
            Bottom_u1_dofs = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(2))
            Bottom_u2_dofs = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(2))
            Top_u1_dofs = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(4))
            Top_u2_dofs = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(4))

            Left_teta_dofs = fem.locate_dofs_topological(ME.sub(2), facet_tags.dim, facet_tags.find(1))
            Right_teta_dofs = fem.locate_dofs_topological(ME.sub(2), facet_tags.dim, facet_tags.find(3))
            Bottom_teta_dofs = fem.locate_dofs_topological(ME.sub(2), facet_tags.dim, facet_tags.find(2))

            # -----------  LCE  ----------------#
            Bottom_q_dofs = fem.locate_dofs_topological(ME.sub(8), facet_tags.dim, facet_tags.find(2))
            Top_q_dofs = fem.locate_dofs_topological(ME.sub(8), facet_tags.dim, facet_tags.find(4))
            Center_q_dofs = fem.locate_dofs_topological(ME.sub(8), facet_tags.dim, facet_tags.find(6))

        
        # --------------   BCs for Fredericksz transition  --------------------------------
            # bcs_1 = dirichletbc(0.0, Bottom_u2_dofs, ME.sub(0).sub(1)) 

            # bcs_2 = dirichletbc(phi_cons, Top_q_dofs, ME.sub(8))  # phi ramp - Top
            # bcs_3 = dirichletbc(0.0, Bottom_q_dofs, ME.sub(8))  # phi ground - Bottom

            # # bcs_4 = dirichletbc(teta0, Bottom_teta_dofs, ME.sub(2))  # Lambda - Left

            # bcs = [bcs_1, bcs_2, bcs_3]
        # --------------------------------------------------------------------------------

        # --------------   BCs for bending  ----------------------------------------------
            bcs_11 = dirichletbc(0.0, Left_u1_dofs , ME.sub(0).sub(0)) 
            bcs_12 = dirichletbc(0.0, Left_u2_dofs , ME.sub(0).sub(1)) 

            bcs_4 = dirichletbc(phi_cons, Top_q_dofs, ME.sub(8))  # phi ramp - Top
            bcs_5 = dirichletbc(0.0, Center_q_dofs, ME.sub(8))    # phi ground - Bottom
            bcs_6 = dirichletbc(0.0, Bottom_q_dofs, ME.sub(8))    # phi ground - Bottom

            bcs = [bcs_11, bcs_12, bcs_4, bcs_5, bcs_6]
        # --------------------------------------------------------------------------------

        # --------------   BCs for buckling 1---------------------------------------------
            # bcs_11 = dirichletbc(0.0, Left_u1_dofs , ME.sub(0).sub(0)) 
            # bcs_12 = dirichletbc(0.0, Left_u2_dofs , ME.sub(0).sub(1)) 

            # bcs_21 = dirichletbc(0.0, Right_u1_dofs , ME.sub(0).sub(0)) 
            # bcs_22 = dirichletbc(0.0, Right_u2_dofs , ME.sub(0).sub(1)) 

            # bcs_4 = dirichletbc(phi_cons, Top_q_dofs, ME.sub(8))  # phi ramp - Top
            # bcs_5 = dirichletbc(0.0, Center_q_dofs, ME.sub(8))    # phi ground - Bottom
            # bcs_6 = dirichletbc(0.0, Bottom_q_dofs, ME.sub(8))    # phi ground - Bottom

            # bcs = [bcs_11,bcs_12, bcs_21,bcs_22, bcs_4, bcs_5, bcs_6]
        # --------------------------------------------------------------------------------


            # Set up the nonlinear problem
            problem = NonlinearProblem(Res_LCE, w, bcs, a_LCE)

            # the global newton solver and params
            solver = NewtonSolver(MPI.COMM_WORLD, problem)
            solver.convergence_criterion = "residual"
            solver.rtol = 1e-8
            solver.atol = 1e-8
            solver.max_it = 50
            solver.report = True

            #  The Krylov solver parameters.
            ksp = solver.krylov_solver
            opts = PETSc.Options()
            option_prefix = ksp.getOptionsPrefix()
            opts[f"{option_prefix}ksp_type"] = "preonly" # "preonly" works equally well
            opts[f"{option_prefix}pc_type"] = "lu" # do not use 'gamg' pre-conditioner
            opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
            opts[f"{option_prefix}mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
            ksp.setFromOptions()


            # Variables for storing time history
            totSteps = 1000000
            timeHist0 = np.zeros(shape=[totSteps])
            timeHist1 = np.zeros(shape=[totSteps]) 
            timeHist2 = np.zeros(shape=[totSteps]) 
            timeHist3 = np.zeros(shape=[totSteps]) 

            #Iinitialize a counter for reporting data
            ii=0



            # Get director angle at a specific point (e.g., center)
            point_for_theta = np.array([Lx/2, Ly*0.75, 0])  # Center point
            bb_tree_for_theta = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
            cell_candidates_for_theta = dolfinx.geometry.compute_collisions_points(bb_tree_for_theta, point_for_theta)
            colliding_cells_for_theta = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates_for_theta, point_for_theta).array

            if len(colliding_cells_for_theta) > 0:
                teta_at_point = w.sub(2).eval(point_for_theta, colliding_cells_for_theta[:1])
                timeHist3[ii] = teta_at_point[0]  # Should be 0.1 (theta0)
            else:
                timeHist3[ii] = np.nan  # If point not found



            # Create function space for F11 component only (more efficient)
            V_F11 = functionspace(domain, ("DG", 0))  # Discontinuous Galerkin space
            F11_func = Function(V_F11)

            # Evaluation point (changed from corner to center)
            point_for_lambda = np.array([Lx/2, Ly/2, 0])  # Center point is more reliable
            bb_tree_for_lambda = dolfinx.geometry.bb_tree(domain, domain.topology.dim)

            # Compute and evaluate F11 at point
            F = F_pe_calc(u_avg)
            F11_expr = Expression(F[0,0], V_F11.element.interpolation_points())
            F11_func.interpolate(F11_expr)

            # Find cells containing the point
            cell_candidates_for_lambda = dolfinx.geometry.compute_collisions_points(bb_tree_for_lambda, point_for_lambda)
            colliding_cells_for_lambda = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates_for_lambda, point_for_lambda).array

            if len(colliding_cells_for_lambda) > 0:
                # Returns a 1D array - take first element
                lambda1 = F11_func.eval(point_for_lambda, colliding_cells_for_lambda[:1])[0]  
                timeHist2[ii] = lambda1
            else:
                timeHist2[ii] = np.nan

            # and also for the velocity and acceleration.
            v_temp = Function(V2)
            a_temp = Function(V2)
            #
            v_expr = Expression(v_new,V2.element.interpolation_points())
            a_expr = Expression(a_new,V2.element.interpolation_points())
            # Write initial state to file
            writeResults(t=0.0)   

            # Print out message for simulation start
            print("------------------------------------")
            print("Simulation Start")
            print("------------------------------------")
            # Store start time 
            startTime = datetime.now()
            
            print(phi_cons.value)
            print(rrr)
            # Time-stepping solution procedure loop
            while (round(t + dt, 9) <= Ttot):
                
                # increment time
                t += dt 
                # increment counter
                ii += 1
                # update time variables in time-dependent BCs 
                phi_cons.value = phiRamp(t)

                # Solve the problem
                try:
                    (iter, converged) = solver.solve(w)
                except: # Break the loop if solver fails
                    print("Ended Early")
                    break

                # Collect results from MPI ghost processes
                w.x.scatter_forward()

                # Print progress of calculation
                if ii%1 == 0:      
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print("Step: {} | Increment: {}, Iterations: {}".\
                        format(step, ii, iter))
                    print("      Simulation Time: {} s  of  {} s".\
                        format(round(t,4), Ttot))
                    print()  
                
                # Write output to file
                writeResults(t)
                timeHist0[ii] = t
                timeHist1[ii] = phiRamp(t)
                # Compute and evaluate F11 at point
                F = F_pe_calc(u_avg)
                F11_expr = Expression(F[0,0], V_F11.element.interpolation_points())
                F11_func.interpolate(F11_expr)
                
                # Find cells containing the point
                cell_candidates_for_lambda = dolfinx.geometry.compute_collisions_points(bb_tree_for_lambda, point_for_lambda)
                colliding_cells_for_lambda = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates_for_lambda, point_for_lambda).array
                
                if len(colliding_cells_for_lambda) > 0:
                    # Returns a 1D array - take first element
                    lambda1 = F11_func.eval(point_for_lambda, colliding_cells_for_lambda[:1])[0]  
                    timeHist2[ii] = lambda1
                else:
                    timeHist2[ii] = np.nan
                

                if len(colliding_cells_for_theta) > 0:
                    teta_at_point = w.sub(2).eval(point_for_theta, colliding_cells_for_theta[:1])
                    timeHist3[ii] = teta_at_point[0]
                else:
                    timeHist3[ii] = np.nan  # Point not found in this process
                

                v_temp.interpolate(v_expr)
                a_temp.interpolate(a_expr)

                # Update DOFs for next step
                w_old.x.array[:] = w.x.array

                v_old.x.array[:] = v_temp.x.array[:]
                a_old.x.array[:] = a_temp.x.array[:]
                
            # close the output file.
            file_results.close()
                    
            # End analysis
            print("-----------------------------------------")
            print("End computation")                 
            # Report elapsed real time for the analysis
            endTime = datetime.now()
            elapseTime = endTime - startTime
            print("------------------------------------------")
            print("Elapsed real time:  {}".format(elapseTime))
            print("------------------------------------------")  



# ====================================================================================
# ======   WRITE TIMEHISTORY RESULTS TO CSV FILES   ==================================
# ====================================================================================
            
            # Only plot as far as we have time history data
            ind = np.argmax(timeHist0) + 1
            
            csv_dir_path = file_type+"/"+"results/rate={}/csvfiles_".format(rate_phi)+analysis_type
            if not os.path.exists(csv_dir_path):
                os.makedirs(csv_dir_path)
            
            with open(csv_dir_path+"/Bending__r={}__theta0=5__timehistory_0___t.csv".format(rate_phi, rrr), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(timeHist0[0:ind])
            csvfile.close()
            
            with open(csv_dir_path+"/Bending__r={}__theta0=5__timehistory_1___phi.csv".format(rate_phi, rrr), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(timeHist1[0:ind])
            csvfile.close()

            with open(csv_dir_path+"/Bending__r={}__theta0=5__timehistory_2___lambda1.csv".format(rate_phi, rrr), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(timeHist2[0:ind])
            csvfile.close()

            with open(csv_dir_path+"/Bending__r={}__theta0=5__timehistory_3___theta.csv".format(rate_phi, rrr), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(timeHist3[0:ind])
            csvfile.close()

# ====================================================================================
# ====================================================================================
# ====================================================================================


        #  .........................................................  #


# ====================================================================================
# =============     PLOT THE RESULTS     =============================================
# ====================================================================================

            # set plot font to size 14
            font = {'size': 16}
            plt.rc('font', **font)

            # Get array of default plot colors
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']

            # Only plot as far as we have time history data
            ind = np.argmax(timeHist0) + 1

            # ********************************************************************
            plot_line_styles = ['solid', 'dotted', 'dashed', 'dashdot', (5, (10, 3))]#'long dash with offset']

            if analysis_type=="theta_fixed":
                line_ind = r_list.index(rrr)
                label_txt = r"$r = $" + format(rrr)
                figs_dir_path = file_type+"/"+"results/figs/theta_fixed"
            elif analysis_type=="r_fixed":
                line_ind = theta_list.index(theta_00_deg)
                label_txt = r"$\theta_0 = $" + format(theta_00_deg) + r"$^\circ$"
                figs_dir_path = file_type+"/"+"results/figs/r_fixed"
            # ********************************************************************

            
            if not os.path.exists(figs_dir_path):
                os.makedirs(figs_dir_path)


# ====================================================================================
# ======   FIGURE 1: LAMBDA-PHI    ===================================================
# ====================================================================================
            plt.figure(fig_count)

            fig = plt.gcf() 
            ax = fig.gca()  

            
            plt.plot(timeHist2[0:ind], timeHist1[0:ind]/1e3, 
                    linewidth=2.0, 
                    linestyle=plot_line_styles[line_ind],  
                    label=label_txt )

            plt.grid(linestyle="--", linewidth=0.5, color='b')
            ax.set_xlabel(r'$\lambda_1$', fontdict=font)
            ax.set_ylabel(r'$\phi~\left[\mathrm{kV}\right]$', fontdict=font)

            # Control number of major ticks (4-6)
            from matplotlib.ticker import MaxNLocator, AutoMinorLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # 5 ticks (including ends)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            # Add minor ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Set limits
            ax.set_ylim(0, 7.5)   
            plt.yticks([0.0, 2.5, 5.0, 7.5])
            ax.set_xlim(0.8, 2.0)            
            plt.xticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
            
            # Force immediate calculation of ticks
            plt.draw()  
            # Optional: Format tick labels (e.g., 2 decimal places)
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            plt.legend()
            fig.set_size_inches(7, 7)
            plt.tight_layout()
            if analysis_type=="theta_fixed":
                plt.savefig(figs_dir_path+"/Fig_Bending____rate={}____r__theta0={}___lambda-phi.png".format(rate_phi, theta_00_deg), dpi=600)
            if analysis_type=="r_fixed":
                plt.savefig(figs_dir_path+"/Fig_Bending____rate={}____theta0__r={}___lambda-phi.png".format(rate_phi, rrr), dpi=600)


# ====================================================================================
# ======   FIGURE 2: THETA-PHI    ===================================================
# ====================================================================================

            plt.figure(fig_count+1)

            fig_theta_phi = plt.gcf() 
            ax_theta_phi = fig_theta_phi.gca()  

            plt.plot(timeHist3[0:ind]*180/pi, timeHist1[0:ind]/1e3, 
                    linewidth=2.0, 
                    linestyle=plot_line_styles[line_ind],  
                    label=label_txt )

            plt.grid(linestyle="--", linewidth=0.5, color='b')
            ax_theta_phi .set_xlabel(r'$\theta~\left[\mathrm{deg}\right]$', fontdict=font)
            ax_theta_phi.set_ylabel(r'$\phi~\left[\mathrm{kV}\right]$', fontdict=font)

            # Control number of major ticks (4-6)
            from matplotlib.ticker import MaxNLocator, AutoMinorLocator
            ax_theta_phi.xaxis.set_major_locator(MaxNLocator(nbins=5))  # 5 ticks (including ends)
            ax_theta_phi.yaxis.set_major_locator(MaxNLocator(nbins=5))

            # Add minor ticks
            ax_theta_phi.xaxis.set_minor_locator(AutoMinorLocator())
            ax_theta_phi.yaxis.set_minor_locator(AutoMinorLocator())

            # Set limits
            # ax.set_ylim(0, 6.0)   
            # plt.yticks([0.0, 2.0, 4.0, 6.0])
            # ax.set_xlim(0, 90)             
            # plt.xticks([0, 30, 60, 90]) 
            
            # Force immediate calculation of ticks
            plt.draw()  
            
            # Optional: Format tick labels (e.g., 2 decimal places)
            ax_theta_phi.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            ax_theta_phi.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            plt.legend()
            fig_theta_phi.set_size_inches(7, 7)
            plt.tight_layout()

            if analysis_type=="theta_fixed":
                plt.savefig(figs_dir_path+"/Fig_Bending____rate={}____r__theta0={}___theta-phi.png".format(rate_phi, theta_00_deg), dpi=600)
            if analysis_type=="r_fixed":
                plt.savefig(figs_dir_path+"/Fig_Bending____rate={}____theta0__r={}___theta-phi.png".format(rate_phi, rrr), dpi=600)


    fig_count = fig_count + 2
