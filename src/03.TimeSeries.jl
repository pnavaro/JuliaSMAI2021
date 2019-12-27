#ms ## Time Series

#md # ---

struct TimeSeries{T,N}

   nt  :: Integer
   nv  :: Integer
   t   :: Vector{T}
   u   :: Vector{Array{T, 1}}

   function TimeSeries{T,N}( nt :: Int) where {T,N}
 
       t  = zeros(T, nt)
       u  = [zeros(T, N) for i in 1:nt]
       nv = N

       new( nt, nv, t, u)

   end

end

import Base:length
length(ts :: TimeSeries) = ts.nt

using Random
using Test
   
nt, nv = 100, 3
ts = TimeSeries{Float64, nv}(nt)
    
@test length(ts) == nt

import Base: getindex

getindex( ts :: TimeSeries ) = getindex.(ts.u)
    
import Base:+

function +(ts :: TimeSeries, ϵ :: Vector{Float64}) 

    for n in 1:ts.nt, d in 1:ts.nv
       ts.u[n][d] += ϵ[d]
    end
    return ts

end

ts  += rand(nv)
