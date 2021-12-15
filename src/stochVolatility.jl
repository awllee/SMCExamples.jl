# Univariate stochastic volatility model
# The model is presented for instance in §2.3.4 of the following book and the 
# proposal for the guided PF is introduced in Example 10.5
# N Chopin, O Papaspiliopoulos - An Introduction to Sequential Monte Carlo, 2020

module StochVolatility
using SequentialMonteCarlo
using RNGPool
import SMCExamples.Particles.Float64Particle

struct SVTheta
    σ::Float64
    μ::Float64
    ρ::Float64
end

function makeSVModel(theta::SVTheta, ys::Vector{Float64}; guide=false)
    n::Int64 = length(ys)
    log2pi = -0.5 * log(2 * π)
    @inline function lG(p::Int64, particle::Float64Particle, ::Nothing)
        @inbounds v::Float64 = particle.x + ys[p]^2 / exp(particle.x)
        return log2pi - 0.5 * v
    end
    @inline function naiveM!(newParticle::Float64Particle, rng::RNG, p::Int64,
        particle::Float64Particle, ::Nothing)
        if p == 1
            newParticle.x = theta.μ + theta.σ*randn(rng)
        else
            newParticle.x = theta.μ + theta.ρ*(particle.x - theta.μ) + theta.σ*randn(rng)
        end
    end
    @inline function guideM!(newParticle::Float64Particle, rng::RNG, p::Int64,
        particle::Float64Particle, ::Nothing)
        μstar::Float64 = if p == 1
            theta.μ 
        else
            theta.μ + theta.ρ*(particle.x - theta.μ)
        end
        @inbounds ξ::Float64 = μstar + 0.25*theta.σ^2*(ys[p]^2*exp(-μstar) - 2)
        newParticle.x = ξ + theta.σ*randn(rng)
    end
    M! = guide ? guideM! : naiveM!
    return SMCModel(M!, lG, n, Float64Particle, Nothing)
end

function simulateSVModel(theta::SVTheta, n::Int64)
  model = makeSVModel(theta, Vector{Float64}(undef, 0); guide=false)
  ys = Vector{Float64}(undef, n)
  xParticle = Float64Particle()
  rng = getRNG()
  for p in 1:n
    model.M!(xParticle, rng, p, xParticle, nothing)
    ys[p] = exp(xParticle.x)*randn(rng)
  end
  return ys
end

function defaultSVModel(n::Int64)
    theta = SVTheta(0.2, -1, 0.97)
    ys = simulateSVModel(theta, n)

    svModel = makeSVModel(theta, ys; guide=false)
    guidedSvModel = makeSVModel(theta, ys; guide=true)

    return svModel, guidedSvModel, theta, ys
end

end
