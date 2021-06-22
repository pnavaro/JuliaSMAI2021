# # Implement your own numerical methods to solve
#
# $$
# y'(t) = 1 - y(t),  t \in [0,5],  y(0) = 0.
# $$

# --

# ## Explicit Euler


euler(f, t, y, h) = t + h, y + h * f(t, y)

# ## Runge-Kutta 2nd order

rk2(f, t, y, h) = begin
    ỹ = y + h / 2 * f(t, y)
    t + h, y + h * f(t + h / 2, ỹ)
end

# ---

# ## Runge-Kutta 4th order

function rk4(f, t, y, dt)

    y₁ = dt * f(t, y)
    y₂ = dt * f(t + dt / 2, y + y₁ / 2)
    y₃ = dt * f(t + dt / 2, y + y₂ / 2)
    y₄ = dt * f(t + dt, y + y₃)

    t + dt, y + (y₁ + 2 * y₂ + 2 * y₃ + y₄) / 6

end

# ---

# ## Solve function

function dsolve(f, method, t₀, y₀, h, nsteps)

    t = zeros(Float64, nsteps)
    y = similar(t)

    t[1] = t₀
    y[1] = y₀

    for i = 2:nsteps
        t[i], y[i] = method(f, t[i-1], y[i-1], h)
    end

    t, y

end

# ---

# ## Plot solutions

using Plots

nsteps, tfinal = 7, 5.0
t₀, x₀ = 0.0, 0.0
dt = tfinal / (nsteps - 1)
f(t, x) = 1 - x

t, y_euler = dsolve(f, euler, t₀, x₀, dt, nsteps)

t, y_rk2 = dsolve(f, rk2, t₀, x₀, dt, nsteps)

t, y_rk4 = dsolve(f, rk4, t₀, x₀, dt, nsteps)

# ---

plot(t, y_euler; marker = :o, label = "Euler")
plot!(t, y_rk2; marker = :d, label = "RK2")
plot!(t, y_rk4; marker = :p, label = "RK4")
plot!(t -> 1 - exp(-t); line = 3, label = "true solution")
savefig("dsolve.png") #hide
# ![dsolve](dsolve.png)

# ---

# ## DifferentialEquations.jl

using DifferentialEquations

f(y, p, t) = 1.0 - y
y₀, t = 0.0, (0.0, 5.0)

prob = ODEProblem(f, y₀, t)

sol_euler = solve(prob, Euler(), dt = 1.0)
sol = solve(prob)

# ---

plot(sol_euler, label = "Euler")
plot!(sol, label = "default")
plot!(1:0.1:5, t -> 1.0 - exp(-t), lw = 3, ls = :dash, label = "True Solution!")

savefig("diffeq.png"); nothing #hide

# ![diffeq](diffeq.png)

# ---

# `sol.t` is the array of time points that the solution was saved at

sol.t

# `sol.u` is the array of solution values

sol.u

# ---

function lorenz(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz, u0, tspan)

sol = solve(prob)

# ---

plot(sol, vars = (1, 2, 3))

savefig("lorenz.png"); nothing #hide

# ![lorenz](lorenz.png)

# ---

using ParameterizedFunctions

lotka_volterra = @ode_def begin
  d🐁 = α*🐁  - β*🐁*🐈
  d🐈 = -γ*🐈 + δ*🐁*🐈
end α β γ δ

u0 = [1.0, 1.0] # Initial condition

tspan = (0.0, 10.0) # Simulation interval 
tsteps = 0.0:0.1:10.0 # intermediary points

p = [1.5, 1.0, 3.0, 1.0] # equation parameters: p = [α, β, δ, γ]

prob = ODEProblem(lotka_volterra, u0, tspan, p)
sol = solve(prob)

# ---

# # Type-Dispatch Programming
#
# - Centered around implementing the generic template of the algorithm not around building representations of data.
# - The data type choose how to efficiently implement the algorithm.
# - With this feature share and reuse code is very easy
#
# [JuliaCon 2019 | The Unreasonable Effectiveness of Multiple Dispatch | Stefan Karpinski](https://youtu.be/kc9HwsxE1OY)

# ---

# Simple gravity pendulum

using DifferentialEquations, Plots

g = 9.79 # Gravitational constants
L = 1.00 # Length of the pendulum

#Initial Conditions
u₀ = [0, π / 60] # Initial speed and initial angle
tspan = (0.0, 6.3) # time domain

#Define the problem
function simplependulum(du, u, p, t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(g/L)*θ
end

prob = ODEProblem(simplependulum, u₀, tspan)
sol = solve(prob, Tsit5(), reltol = 1e-6)


# ---

# Analytic and computed solution
u = u₀[2] .* cos.(sqrt(g / L) .* sol.t)

scatter(sol.t, getindex.(sol.u, 2), label = "Numerical")
plot!(sol.t, u, label = "Analytic")
savefig("pendulum1.svg"); nothing # hide

# ![](pendulum1.svg)

# ---

# [Numbers with Uncertainties](http://tutorials.juliadiffeq.org/html/type_handling/02-uncertainties.html)

using Measurements

g = 9.79 ± 0.02; # Gravitational constants
L = 1.00 ± 0.01; # Length of the pendulum

#Initial Conditions
u₀ = [0 ± 0, π / 60 ± 0.01] # Initial speed and initial angle

#Define the problem
function simplependulum(du, u, p, t)
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(g/L)*θ
end

#Pass to solvers
prob = ODEProblem(simplependulum, u₀, tspan)
sol = solve(prob, Tsit5(), reltol = 1e-6);
nothing # hide

#md # ---

# Analytic solution
u = u₀[2] .* cos.(sqrt(g / L) .* sol.t)

plot(sol.t, getindex.(sol.u, 2), label = "Numerical")
plot!(sol.t, u, label = "Analytic")
savefig("pendulum2.svg"); nothing # hide

# ![](pendulum2.svg)

# ---
