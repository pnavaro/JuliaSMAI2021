[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/juliavlasov/Numkin2019/master?filepath=notebooks)

# Talk at INRIA Rennes January 2020

Either use the link above to open the notebooks in
[mybinder.org](https://mybinder.org/v2/gh/pnvaro/JuliaInriaTech/master?filepath=notebooks) or
run them locally:

```bash
git clone https://github.com/pnavaro/JuliaInriaTech
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


## References

- [Francois Fevotte - Talk at Julia Day in Nantes June 2019](https://github.com/triscale-innov/Nantes2019)
- [Michael Herbst - Lecture notes and material for the Julia day at Sorbonne Université 2019](https://michael-herbst.com/teaching/2019-julia-day-jussieu/)
