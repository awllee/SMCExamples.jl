using SequentialMonteCarlo
using SMCExamples.Lorenz96
using StaticArrays

include("test.jl")

setSMCRNGs(0)

model, theta, ys = Lorenz96.defaultLorenzModel(8, 100)

## just run the algorithm a few times

testSMC(model, 1024, 2, false)
testSMCParallel(model, 1024, 2, false)
