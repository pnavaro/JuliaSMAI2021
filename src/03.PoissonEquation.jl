# # Poisson Equation
#
# $$
# \frac{\partial^2 u}{\partial x^2} = b  \qquad x \in [0,1]
# $$
#
# $$
# u(0) = u(1) = 0, \qquad b = \sin(2\pi x)
# $$

using Plots, SparseArrays


Δx = 0.05
x = Δx:Δx:1-Δx ## Solve only interior points: the endpoints are set to zero.
n = length(x)
B = sin.(2π*x) * Δx^2

P = spdiagm( -1 =>    ones(Float64,n-1),
              0 => -2*ones(Float64,n),
              1 =>    ones(Float64,n-1))

u1 = P \ B

# ---

plot([0;x;1],[0;u1;0], label="computed")
scatter!([0;x;1],-sin.(2π*[0;x;1])/(4π^2),label="exact")
savefig("poisson1.png"); nothing #hide

# ![](poisson1.png)

# ---

# # DiffEqOperators.jl

using DiffEqOperators

Δx = 0.05
x = Δx:Δx:1-Δx ## Solve only interior points: the endpoints are set to zero.
n = length(x)
b = sin.(2π*x) 

## Second order approximation to the second derivative
order = 2
deriv = 2

Δ = CenteredDifference{Float64}(deriv, order, Δx, n)
bc = Dirichlet0BC(Float64)

u2 = (Δ * bc) \ b

# ---

plot([0;x;1],[0;u2;0], label="computed")
scatter!([0;x;1],-sin.(2π*[0;x;1])/(4π^2),label="exact")
savefig("poisson2.png"); nothing #hide

# ![](poisson2.png)

# ---
