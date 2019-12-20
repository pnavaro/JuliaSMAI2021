#ms ## Time Series

#md # ---

struct TimeSeries{T,N}

   nt  :: Integer
   t   :: Vector{T}
   u   :: Vector{Array{T, 1}}

   function TimeSeries{T,N}( nt :: Int) where {T,N}
 
       time   = zeros(T, nt)
       values = [zeros(T, N) for i in 1:nt]

       new( nt, nv, time, values)

   end

   function TimeSeries( time   :: Array{Float64, 1}, 
                        values :: Array{Array{Float64, 1}})
 
       nt = length(time)
       nv = size(first(values))[1]

       new( nt, nv, time, values)

   end

end

