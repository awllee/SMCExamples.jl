using SequentialMonteCarlo
using SMCExamples.FiniteFeynmanKac

VERSION.minor == 6 && using Base.Test
VERSION.minor > 6 && using Test

setSMCRNGs(0)

d = 3
n = 10
ffk = FiniteFeynmanKac.randomFiniteFK(d, n)
ffkout = FiniteFeynmanKac.calculateEtasZs(ffk)
model = FiniteFeynmanKac.makeSMCModel(ffk)
smcio = SMCIO{model.particle, model.pScratch}(2^13, n, Threads.nthreads(),
  false, 0.5)
smc!(model, smcio)

@test smcio.logZhats ≈ ffkout.logZhats atol=0.1

## just test that the commands run; actual testing is part of the tests in
## SequentialMonteCarlo.jl

FiniteFeynmanKac.fullDensity(ffk, FiniteFeynmanKac.Int642Path(1, d, n))
FiniteFeynmanKac.allEtas(ffkout, p -> p.x, false)
FiniteFeynmanKac.allEtas(ffkout, p -> p.x, true)
FiniteFeynmanKac.allGammas(ffkout, p -> p.x, false)
FiniteFeynmanKac.allGammas(ffkout, p -> p.x, true)
FiniteFeynmanKac.vpns(ffk, ffkout, p -> p.x, true, true, smcio.resample)
FiniteFeynmanKac.avar(ffk, ffkout, p -> p.x, true, true, smcio.resample)
FiniteFeynmanKac.allavarhat1s(ffk, ffkout, smcio.resample)

@test FiniteFeynmanKac.Path2Int64(FiniteFeynmanKac.Int642Path(1, d, n), d) == 1
