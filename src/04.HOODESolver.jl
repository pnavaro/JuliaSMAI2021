# # HOODESolver.jl
# 
# The objective of this Julia package is to valorize the recent developments carried out within [INRIA team MINGuS](https://team.inria.fr/mingus/) on Uniformly Accurate numerical methods (UA) for highly oscillating problems. We propose to solve the following equation 
# 
# $$\frac{d u(t)}{dt} = \frac{1}{\varepsilon} A u(t) + f(t, u(t)), \qquad u(t=t_0)=u_0, \qquad \varepsilon\in ]0, 1], \qquad (1)$$
# 
# with 
# -  $u : t \in [t_0, t_1] \mapsto u(t)\in \mathbb{R}^n, \quad t_0, t_1 \in \mathbb{R}$, 
# -  $u_0 \in \mathbb{R}^n$, 
# -  $A\in {\mathcal{M}}_{n,n}(\mathbb{R})$ is such that $\tau \mapsto \exp(\tau A)$ is $2 \pi$-periodic,  
# -  $f : (t, u) \in  \mathbb{R}\times \mathbb{R}^n \mapsto \mathbb{R}^n$.
# 
# https://pnavaro.github.io/HOODESolver.jl/stable/
#
# Philippe Chartier, Nicolas Crouseilles, Mohammed Lemou, Florian Mehats and Xiaofei Zhao.
#
# Package: Yves Mocquard and Pierre Navaro.
#
# ---
#
# ## Two-scale formulation 
# 
# First, rewrite equation (1) using the variable change $w(t)=\exp(-(t-t_0)A/\varepsilon) u(t)$ to obtain
# 
# $$\frac{d w(t)}{dt} = F\Big(\frac{t-t_0}{\varepsilon}, w(t) \Big), $$
#
# $$w(t_0) = u_0, \varepsilon \in ]0, 1], $$
# 
# 
# where the function $F$ is expressed from the data of the original problem (1)
# 
# $$F\Big( \frac{s}{\varepsilon}, w \Big) = \exp(-sA/\varepsilon) \; f( \exp(sA/\varepsilon), \; w).$$
# 
# We then introduce the function $U(t, \tau), \tau\in [0, 2 \pi]$ such that $U(t, \tau=(t-t_0)/\varepsilon) = w(t)$. The two-scale function is then the solution of the following equation.
# 
# $$\frac{\partial U}{\partial t} + \frac{1}{\varepsilon} \frac{\partial U}{\partial \tau} =  F( \tau, U), \;\;\; U(t=t_0, \tau)=\Phi(\tau), \;\; \varepsilon\in ]0, 1], \;\;\;\;\;\;\;\;\;\; (2)$$
# 
# where $\Phi$ is a function checking $\Phi(\tau=0)=u_{0}$ chosen so that the $U$ solution of (2) is smooth. 
# 
# 
# ---

# # Hénon-Heiles Example

# We consider the system of Hénon-Heiles satisfied by $u(t)=(u_1, u_2, u_3, u_4)(t)$.
# 
# $$\frac{d u }{dt} = \frac{1}{\varepsilon} Au + f(u),  $$
# $$ u(t_0)=u_0 \in \mathbb{R}^4,$$
# 
# where $A$ and $f$ are selected as follows
# 
# $$A=
# \\begin{pmatrix}
# 0 & 0 & 1 & 0  \\\\
# 0 & 0 & 0 & 0  \\\\
# -1 & 0 & 0 & 0  \\\\
# 0 & 0 & 0 & 0  
# \\end{pmatrix}, \qquad
# f(u) = \left(
# \begin{array}{cccc}
# 0 \\\\
# u_4\\\\
# -2 u_1 u_2\\\\
# -u_2-u_1^2+u_2^2
# \end{array}
# \right).$$
# 

# ---

# # SplitODEProblem
# 
# The `SplitODEProblem` type from package [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/types/split_ode_types/) offers a interface for this kind of problem.
# 

using Plots, DifferentialEquations

epsilon = 0.002
A = [ 0 0 1 0 ;
      0 0 0 0 ;
     -1 0 0 0 ;
      0 0 0 0 ]

f1 = DiffEqArrayOperator( A ./ epsilon)

function f2(du, u, p, t)
    du[1] = 0
    du[2] = u[4]
    du[3] = 2*u[1]*u[2]
    du[4] = -u[2] - u[1]^2 + u[2]^2 
end

tspan = (0.0, 0.1)

u0 = [0.55, 0.12, 0.03, 0.89]

prob1 = SplitODEProblem(f1, f2, u0, tspan);
sol1 = solve(prob1, ETDRK4(), dt=0.001);

# ---

using HOODESolver, Plots

A = [ 0 0 1 0 ; 
      0 0 0 0 ; 
     -1 0 0 0 ; 
      0 0 0 0 ]

f1 = LinearHOODEOperator( epsilon, A)

prob2 = SplitODEProblem(f1, f2, u0, tspan);

sol2 = solve(prob2, HOODEAB(), dt=0.01);

# ---

plot(sol1, vars=[3], label="EDTRK4")
plot!(sol2, vars=[3], label="HOODEAB")
plot!(sol2.t, getindex.(sol2.u, 3), m=:o, label="points")
savefig("sol1sol2.png"); nothing #hide

# ![](sol1sol2.png)

# ---

# # High precision

u0 = BigFloat.([90, -44, 83, 13]//100)
t_end = big"1.0"
epsilon = big"0.0017"

prob = HOODEProblem(f2, u0, (big"0.0",t_end), missing, A, epsilon)

# The float are coded on 512 bits.


# ---  

# ## Precision of the result with ε = 0.015

# ![](assets/error_order.png)

# JOSS paper https://joss.theoj.org/papers/10.21105/joss.03077
#
# Much of HOODESolver.jl was implemented by Y. Mocquard while he was supported by Inria through the AdT (Aide au développement technologique) J-Plaff of the center Rennes- Bretagne Atlantique.
# 
