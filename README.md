# Talk at INRIA Rennes January 2020

To open the notebooks run them locally:

```bash
git clone https://gitlab.inria.fr/navarop/JuliaInriaTech
cd JuliaInriaTech
julia --project
```

```julia
julia> using Pkg
julia> Pkg.instantiate()
julia> include("generate_nb.jl")
julia> using IJulia
julia> notebook(dir=joinpath(pwd(),"notebooks"))
[ Info: running ...
```
