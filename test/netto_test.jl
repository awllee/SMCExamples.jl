using SequentialMonteCarlo
import SMCExamples.Netto.defaultNettoModel

VERSION.minor == 6 && using Base.Test
VERSION.minor > 6 && using Test

setSMCRNGs(0)

N = 8192
n = 10
nt = Threads.nthreads()

model, theta, ys = defaultNettoModel(n)

smcio = SMCIO{model.particle, model.pScratch}(N, n, nt, false)
smc!(model, smcio)
