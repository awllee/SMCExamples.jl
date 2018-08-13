module SMCExamples

using SequentialMonteCarlo
using RNGPool
using StaticArrays

include("particles.jl")
include("lgModel.jl")
include("mvlgModel.jl")
include("smcSampler.jl")
include("finiteFK.jl")
include("nettoModel.jl")
include("lorenz96Model.jl")

end # module
