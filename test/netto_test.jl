using SequentialMonteCarlo
using Random
import SMCExamples.Netto.defaultNettoModel
using Test

Random.seed!(0)

N = 8192
n = 10
nt = Threads.nthreads()

model, theta, ys = defaultNettoModel(n)

smcio = SMCIO{model.particle, model.pScratch}(N, n, nt, false)
smc!(model, smcio)
