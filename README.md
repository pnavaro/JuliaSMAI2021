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
julia> using IJulia
julia> notebook(dir=joinpath(pwd(),"notebooks"))
[ Info: running ...
```

# References

- [Francois Fevotte - Talk at Julia Day in Nantes June 2019](https://github.com/triscale-innov/Nantes2019)
- [Michael Herbst - Lecture notes and material for the Julia day at Sorbonne Université 2019](https://michael-herbst.com/teaching/2019-julia-day-jussieu/)
