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

smcio = SMCIO{model.particle, model.pScratch}(N, n, nt, true)
smc!(model, smcio)

guidedsmcio = SMCIO{guidedmodel.particle, guidedmodel.pScratch}(N, n, nt, true)
smc!(guidedmodel, guidedsmcio)

f = p -> p.x
@test V(smcio, f, false, false, n) ≥ V(guidedsmcio, f, false, false, n)
@test V(smcio, f, false, true, n) ≥ V(guidedsmcio, f, false, true, n)
@test V(smcio, f, true, false, n) ≥ V(guidedsmcio, f, true, false, n)
@test V(smcio, f, true, true, n) ≥ V(guidedsmcio, f, true, true, n)
