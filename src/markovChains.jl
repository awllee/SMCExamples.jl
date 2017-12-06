## Lightweight functions for simulating a Monte Carlo Markov chain

module MarkovChains

using StaticArrays
using SMCExamples.Visualize

function simulateChain!(chain::Vector{T}, P::F, x0::T) where {F<:Function, T}
  n::Int64 = length(chain)
  x::T = x0
  for i = 1:n
    x = P(x)
    @inbounds chain[i] = x
  end
  return chain
end

function Base.cov(xs::Vector{SVector{d, Float64}}) where d
  xbar = mean(xs)
  Q::SMatrix{d, d, Float64} = zeros(SMatrix{d, d, Float64})
  for i = 1:length(xs)
    @inbounds Q += (xs[i] - xbar)*(xs[i] - xbar)'
  end
  return Q / (length(xs)-1)
end

## Basic batch means estimation of the asymptotic variance
function estimateAvar(xs::Vector{T}, f::F = x -> x) where F<:Function where T
  n::Int64 = length(xs)
  sqrtn::Int64 = floor(Int64, sqrt(n))
  m::Int64 = sqrtn * sqrtn
  start::Int64 = n - m
  overallMean::Float64 = 0.0
  for i = start+1:n
    overallMean += f(xs[i])
  end
  overallMean /= m
  acc::Float64 = 0.0
  for i = 1:sqrtn
    batchAcc::Float64 = 0.0
    batchStart::Int64 = start + (i-1)*sqrtn
    for j = 1:sqrtn
      batchAcc += f(xs[batchStart + j])
    end
    tmp::Float64 = batchAcc/sqrtn
    tmp -= overallMean
    acc += tmp * tmp
  end
  return sqrtn/(sqrtn-1)*acc
end

## The adaptive Metropolis kernel proposed by:
## Haario, H., Saksman, E. and Tamminen, J., 2001. An adaptive Metropolis
## algorithm. Bernoulli, 7(2), pp.223-242.
## Essentially provides an adaptive mechanism to make use of the optimal
## scaling results for random walk Metropolis on "Gaussian-like" targets
## described by:
## Roberts, G.O. and Rosenthal, J.S., 2001. Optimal scaling for various
## Metropolis--Hastings algorithms. Statistical Science, 16(4), pp.351-367.
function makeAMKernel(logTargetDensity::F, Σ::SMatrix{d, d, Float64}) where
  {F<:Function, d}
  S::MMatrix{d, d, Float64} = Σ
  A::MMatrix{d, d, Float64} = chol(Symmetric(S))'

  scratchv::MVector{d, Float64} = MVector{d, Float64}()
  prevx::MVector{d, Float64} = MVector{d, Float64}()
  ldprevx = Ref(-Inf)

  accepts = Ref(0)
  calls = Ref(0)
  covEstimate::MMatrix{d, d, Float64} = Σ
  meanEstimate::MVector{d, Float64} = zeros(MVector{d, Float64})
  function retuneSigma()
    S .= 5.6644/d * covEstimate * calls.x /(calls.x-1)
    A .= chol(Symmetric(S))'
  end
  function P(x::SVector{d, Float64})
    calls.x += 1
    randn!(scratchv)
    scratchv .= A * scratchv
    z::SVector{d, Float64} = x + scratchv
    if x == prevx
      lpi_x = ldprevx.x
    else
      lpi_x = logTargetDensity(x)
      prevx .= x
    end
    lpi_z = logTargetDensity(z)
    if -randexp() < lpi_z - lpi_x
      prevx .= z
      ldprevx.x = lpi_z
      accepts.x += 1
      rval = z
    else
      rval = x
    end
    t::Int64 = calls.x
    covEstimate .= (t-1)/t * (covEstimate +
      (rval - meanEstimate) * (rval - meanEstimate)' / t)
    meanEstimate .= (calls.x-1)/calls.x.*meanEstimate + rval/calls.x
    mod(calls.x, 1024) == 0 && retuneSigma()
    return rval
  end
  function P(s::Symbol)
    s == :acceptanceRate && return accepts.x / calls.x
    s == :meanEstimate && return meanEstimate
    s == :covEstimate && return covEstimate * calls.x /(calls.x-1)
  end
  return P
end

## Compute adjusted kernel density estimates using an conservative approximation
## of the quantity proposed by:
## Sköld, M. and Roberts, G.O., 2003. Density estimation for the
## Metropolis–Hastings algorithm. Scandinavian Journal of Statistics, 30(4),
## pp.699-718.

function kdeMarkovChain(vs::Vector{Float64}, acceptanceRate::Float64 = 1.0)
  adjust = (2 / acceptanceRate - 1)^(1/5)
  return Visualize.kde(vs, adjust)
end

function kdeMarkovChain(xs::Vector{Float64}, ys::Vector{Float64},
  acceptanceRate::Float64 = 1.0)
  adjust = (2 / acceptanceRate - 1)^(1/6)
  return Visualize.kde(xs, ys, adjust)
end

end
