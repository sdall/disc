module Disc

include.(("SetUtils.jl", "MaxEnt.jl", "Lattice.jl", "Decomposer.jl"))

_convert_dataset(xs::Vector{SetType}) where {SetType<:Union{Set,BitSet}} = xs
_convert_dataset(xs) = ThreadsX.map(x -> findall(!=(0), x) |> BitSet, eachrow(xs))

export disc_greedy, disc_heuristic, patterns

end # module Disc
