using SMCExamples.MarkovChains
using StaticArrays
using Compat.Test

## demo with a MV normal target

function makelogMVN{d}(μ::SVector{d, Float64}, Σ::SMatrix{d, d, Float64})
  invΣ = inv(Σ)
  lognc = - 0.5 * d * log(2 * π) - 0.5 * logdet(Σ)
  function lpi(x::SVector{d, Float64})
    v = x - μ
    return lognc - 0.5*dot(v, invΣ * v)
  end
  return lpi
end

zero2 = [0.0, 0.0]
μ1 = [-1.0, 0.0]
Σ1 = [1.0 0.25 ; 0.25 3.0]
propSigma = .01*eye(2)

Szero2 = SVector{2, Float64}(zero2)
SpropSigma = SMatrix{2, 2, Float64}(propSigma)

logtarget = makelogMVN(SVector{2, Float64}(μ1),
  SMatrix{2, 2, Float64}(Σ1))

niterations = 2^20
chain = Vector{SVector{2, Float64}}(niterations)

srand(12345)

P_AM = MarkovChains.makeAMKernel(logtarget, SpropSigma)
MarkovChains.simulateChain!(chain, P_AM, Szero2)

@test mean(chain) ≈ P_AM(:meanEstimate)
@test cov(chain) ≈ P_AM(:covEstimate)
@test mean(chain) ≈ μ1 atol = 0.01
@test cov(chain) ≈ Σ1 atol = 0.05

vs = Vector{Vector{Float64}}(2)
for i = 1:2
  vs[i] = (x->x[i]).(chain)
end

d1(x) = 1/sqrt(2*π*Σ1[1,1])*exp(-1/2/Σ1[1,1]*(x-μ1[1])^2)
d2(x) = 1/sqrt(2*π*Σ1[2,2])*exp(-1/2/Σ1[2,2]*(x-μ1[2])^2)

xs, ys = MarkovChains.kdeMarkovChain(vs[1], P_AM(:acceptanceRate))
@test ys ≈ d1.(xs) atol = 0.05
xs, ys = MarkovChains.kdeMarkovChain(vs[2], P_AM(:acceptanceRate))
@test ys ≈ d2.(xs) atol = 0.05
xs, ys, f1 = MarkovChains.kdeMarkovChain(vs[1], vs[2], P_AM(:acceptanceRate))

tmp = 0.0
for x in xs
  for y in ys
    tmp += (f1(x,y) - exp(logtarget(SVector{2,Float64}(x,y))))^2
  end
end
@test sqrt(tmp) < 0.05

### AR1 process for testing asymptotic variance estimation

function makeAR1Kernel(c::Float64, φ::Float64, σ::Float64)
  function P(x::Float64)
    return c + φ * x + σ * randn()
  end
  return P
end

const c = 2.32
const φ = 0.98
const σ = 2.3
P = makeAR1Kernel(c, φ, σ)

chain = Vector{Float64}(1024*1024*4)

MarkovChains.simulateChain!(chain, P, 0.0)

@test mean(chain) ≈ c/(1-φ) atol=0.1
@test var(chain) ≈ σ^2/(1-φ^2) atol=5.0
@test MarkovChains.estimateAvar(chain) ≈ σ^2/(1-φ^2) * (1+φ)/(1-φ) rtol=0.1
