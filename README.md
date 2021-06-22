# Talk at SMAI 2021 La Grande Motte [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pnavaro/JuliaSMAI2021/HEAD?filepath=index.ipynb)

Link to [slides](https://pnavaro.github.io/JuliaSMAI2021).

Link to [binder](https://mybinder.org/v2/gh/pnavaro/JuliaSMAI2021/HEAD?filepath=index.ipynb).

To open the notebooks run them locally:

```bash
git clone https://github.com/pnavaro/JuliaSMAI2021
cd JuliaSMAI2021
julia --project
```

```julia
julia> using Pkg
julia> Pkg.instantiate()
julia> using IJulia
julia> notebook(dir=joinpath(pwd(),"notebooks"))
[ Info: running ...
```
