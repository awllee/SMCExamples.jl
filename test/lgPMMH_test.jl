using SequentialMonteCarlo
using SMCExamples.LinearGaussian: LGTheta, Float64Particle, kalmanlogZ,
  defaultLGModel, makeLGModel
import SMCExamples.MarkovChains: simulateChain!, makeAMKernel
using StaticArrays
using Compat.Test

setSMCRNGs(0)
lgModel, theta, ys, ko = defaultLGModel(100)

const truex0 = theta.x0
const truev0 = theta.v0
const trueC = theta.C

@inline function toLGTheta(v::SVector{3, Float64})
  return LGTheta(v[1], v[2], trueC, v[3], truex0, truev0)
end

t0 = SVector{3, Float64}(theta.A, theta.Q, theta.R)

VERSION.minor == 6 &&
  (const sigmaProp = SMatrix{3, 3, Float64}(eye(3)))
VERSION.minor > 6 &&
  (const sigmaProp = SMatrix{3, 3, Float64}(Matrix{Float64}(I, 3, 3)))

@inline function lglogprior(theta::LGTheta)
  if theta.A < 0 || theta.A > 10 return -Inf end
  if theta.Q < 0 || theta.Q > 10 return -Inf end
  if theta.R < 0 || theta.R > 10 return -Inf end
  return 0.0
end

function makelgsmcltd(ys::Vector{Float64}, N::Int64, nthreads::Int64)
  smcio = SMCIO{Float64Particle, Void}(N, length(ys), nthreads, false, 0.5)
  function ltd(in::SVector{3, Float64})
    theta::LGTheta = toLGTheta(in)
    lp::Float64 = lglogprior(theta)
    if lp == -Inf return -Inf end
    model::SMCModel = makeLGModel(theta, ys)
    smc!(model, smcio)
    return smcio.logZhats[length(ys)]
  end
end

function makelgkalmanltd(ys::Vector{Float64})
  function ltd(in::SVector{3, Float64})
    theta::LGTheta = toLGTheta(in)
    lp::Float64 = lglogprior(theta)
    if lp == -Inf return -Inf end
    return kalmanlogZ(theta, ys)
  end
end

logtargetSMC = makelgsmcltd(ys, 1024, Threads.nthreads())
logtargetKalman = makelgkalmanltd(ys)

PSMC = makeAMKernel(logtargetSMC, sigmaProp)
PKalman = makeAMKernel(logtargetKalman, sigmaProp)

srand(12345)

chainSMC = Vector{SVector{3, Float64}}(2048)
simulateChain!(chainSMC, PSMC, t0)

chainKalman = Vector{SVector{3, Float64}}(1024*1024)
simulateChain!(chainKalman, PKalman, t0)

@test mean(chainSMC) ≈ mean(chainKalman) atol=0.2
@test cov(chainSMC) ≈ cov(chainKalman) atol=0.2
