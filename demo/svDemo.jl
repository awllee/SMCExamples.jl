using SequentialMonteCarlo
using RNGPool
import SMCExamples.StochVolatility.defaultSVModel

include("test.jl")

setRNGs(0)

model, guidedmodel, theta, ys = defaultSVModel(10)

numParticles = 1024*1024
numTrials = 2

testSMCParallel(model, numParticles, numTrials, false)
testSMCParallel(model, numParticles, numTrials, false, 0.5)
testSMCParallel(guidedmodel, numParticles, numTrials, false)
testSMCParallel(guidedmodel, numParticles, numTrials, false, 0.5)
