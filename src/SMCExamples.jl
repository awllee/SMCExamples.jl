module SMCExamples

using SequentialMonteCarlo
using Random
using StaticArrays

include("particles.jl")
include("lgModel.jl")
include("mvlgModel.jl")
include("smcSampler.jl")
include("finiteFK.jl")
include("nettoModel.jl")
include("lorenz96Model.jl")
include("stochVolatility.jl")

end # module
