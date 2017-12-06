## Multivariate linear Gaussian model

module MVLinearGaussian

using SequentialMonteCarlo
using StaticArrays
import SMCExamples.Particles.MVFloat64Particle

struct MVLGTheta{d}
  A::SMatrix{d, d, Float64}
  Q::SMatrix{d, d, Float64}
  C::SMatrix{d, d, Float64}
  R::SMatrix{d, d, Float64}
  x0::SVector{d, Float64}
  v0::SVector{d, Float64}
end

## scratch space for computations
struct MVLGPScratch{d}
  t1::MVector{d, Float64}
  t2::MVector{d, Float64}
end
MVLGPScratch{d}() where d = MVLGPScratch{d}(MVector{d, Float64}(),
  MVector{d, Float64}())

function makeMVLGModel(theta::MVLGTheta, ys::Vector{SVector{d, Float64}}) where
  d
  n = length(ys)
  cholQ = chol(theta.Q)'
  invRover2 = 0.5 * inv(theta.R)
  sqrtv0 = sqrt.(theta.v0)
  logncG = - 0.5 * d * log(2 * π) - 0.5 * logdet(theta.R)
  @inline function lG(p::Int64, particle::MVFloat64Particle{d},
    scratch::MVLGPScratch{d})
    # v = theta.C*particle.x - ys[p]
    # return logncG - dot(v,invRover2 * v)
    A_mul_B!(scratch.t1, theta.C, particle.x)
    @inbounds scratch.t2 .= scratch.t1 .- ys[p]
    A_mul_B!(scratch.t1, invRover2, scratch.t2)
    return logncG - dot(scratch.t1,scratch.t2)
  end
  @inline function M!(newParticle::MVFloat64Particle{d}, rng::SMCRNG, p::Int64,
    particle::MVFloat64Particle{d}, scratch::MVLGPScratch{d})
    if p == 1
      randn!(rng, scratch.t1)
      scratch.t2 .= sqrtv0 .* scratch.t1
      newParticle.x .= theta.x0 .+ scratch.t2
    else
      # newParticle.x .= theta.A*particle.x + cholQ*randn(rng,d)
      randn!(rng, scratch.t1)
      A_mul_B!(scratch.t2, cholQ, scratch.t1)
      A_mul_B!(scratch.t1, theta.A, particle.x)
      newParticle.x .= scratch.t1 .+ scratch.t2
    end
  end
  return SMCModel(M!, lG, length(ys), MVFloat64Particle{d}, MVLGPScratch{d})
end

function simulateMVLGModel(theta::MVLGTheta{d}, n::Int64) where d
  model = makeMVLGModel(theta, Vector{SVector{d, Float64}}(0))
  ys = Vector{SVector{d, Float64}}(n)
  xParticle = MVFloat64Particle{d}()
  xScratch = MVLGPScratch{d}()
  cholR = chol(theta.R)'
  rng = getSMCRNG()
  for p in 1:n
    model.M!(xParticle, rng, p, xParticle, xScratch)
    ys[p] = theta.C*(xParticle.x) + cholR * randn(rng,d)
  end
  return ys
end

function defaultMVLGModel(d::Int64, n::Int64)
  function toeplitz(d::Int64, a::Float64, C::Float64)
    M = Matrix{Float64}(d, d)
    for i = 1:d
      for j = 1:d
        M[i,j] = C * a^abs(i-j)
      end
    end
    return SMatrix{d, d, Float64}(M)
  end

  VERSION.minor == 6 && (tA = SMatrix{d, d, Float64}(0.9 * eye(d)))
  VERSION.minor > 6 &&
    (tA = SMatrix{d, d, Float64}(0.9 * Matrix{Float64}(I, d, d)))
  tC = toeplitz(d, 0.5, 1.2)
  tQ = toeplitz(d, 0.2, 0.6)
  tR = toeplitz(d, 0.3, 1.5)
  tx0 = SVector{d,Float64}(linspace(1, d, d))
  tv0 = SVector{d,Float64}(linspace(2, d+1, d))
  theta = MVLGTheta(tA, tQ, tC, tR, tx0, tv0)
  ys = simulateMVLGModel(theta, n)

  ko = kalmanMV(theta, ys)

  mvlgModel = makeMVLGModel(theta, ys)

  return mvlgModel, theta, ys, ko
end

include("mvlgKalman.jl")

end