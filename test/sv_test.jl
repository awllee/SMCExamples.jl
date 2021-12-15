using SequentialMonteCarlo
import SequentialMonteCarlo.V
using RNGPool
import SMCExamples.StochVolatility.defaultSVModel
using Test

setRNGs(0)

N = 32768
n = 10
nt = Threads.nthreads()
model, guidedmodel, theta, ys = defaultSVModel(n)

smcio = SMCIO{model.particle, model.pScratch}(N, n, nt, false)
smc!(model, smcio)

guidedsmcio = SMCIO{guidedmodel.particle, guidedmodel.pScratch}(N, n, nt, false)
smc!(guidedmodel, guidedsmcio)
