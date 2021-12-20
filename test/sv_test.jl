using SequentialMonteCarlo
import SequentialMonteCarlo: V, eta
using RNGPool
import SMCExamples.StochVolatility.defaultSVModel
using Test
import Statistics: std, mean

setRNGs(0)

N = 32768
n = 10
numTrials = 5
nt = Threads.nthreads()
model, guidedmodel, theta, ys = defaultSVModel(n)

function estimate_mean(; guided=false, filtering=false)
    m = guided ? guidedmodel : model
    smcio = SMCIO{m.particle, m.pScratch}(N, n, nt, false)
    smc!(m, smcio)
    return eta(smcio, particle -> particle.x, filtering, n)
end

guided_predictive_means = [estimate_mean(guided=true, filtering=false) for _ in 1:numTrials]
guided_filtering_means  = [estimate_mean(guided=true, filtering=true) for _ in 1:numTrials]
predictive_means        = [estimate_mean(guided=false, filtering=false) for _ in 1:numTrials]
filtering_means         = [estimate_mean(guided=false, filtering=true) for _ in 1:numTrials]

@test mean(guided_filtering_means) ≈ mean(filtering_means) atol=0.05
@test std(guided_filtering_means) ≤ std(filtering_means)

@test mean(guided_predictive_means) ≈ mean(predictive_means) atol=0.05

