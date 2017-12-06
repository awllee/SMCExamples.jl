module Visualize

using KernelDensity

## uses Silverman’s rule of thumb
function _defaultBandwidth(vs::Vector{Float64})
  return std(vs)*0.9*length(vs)^(-1/5)
end

## uses Silverman’s rule of thumb
function _defaultBandwidth(xs::Vector{Float64}, ys::Vector{Float64})
  @assert length(xs) == length(ys)
  len = length(xs)
  h1 = std(xs)*0.9*len^(-1/6)
  h2 = std(ys)*0.9*len^(-1/6)
  return [h1, h2]
end

# univariate, return value can be passed to Plots.plot
function kde(vs::Vector{Float64}, adjust::Float64 = 1.0)
  hDefault = _defaultBandwidth(vs)
  hAdjusted = hDefault * adjust
  left = minimum(vs) - 3 * hAdjusted
  right = maximum(vs) + 3 * hAdjusted
  xs = linspace(left, right, 512)
  ys = pdf(InterpKDE(KernelDensity.kde(vs; bandwidth = hAdjusted)), xs)
  return xs, ys
end

# bivariate, return value can be passed to Plots.contour
function kde(xs::Vector{Float64}, ys::Vector{Float64}, adjust::Float64 = 1.0)
  hAdjusted = _defaultBandwidth(xs, ys) * adjust
  left = minimum(xs) - 3 * hAdjusted[1]
  right = maximum(xs) + 3 * hAdjusted[1]
  bottom = minimum(ys) - 3 * hAdjusted[2]
  top = maximum(ys) + 3 * hAdjusted[2]
  xOut = linspace(left, right, 128)
  yOut = linspace(bottom, top, 128)
  ikde = InterpKDE(KernelDensity.kde((xs,ys); bandwidth = Tuple(hAdjusted)))
  function de(x, y)
    return pdf(ikde, x, y)
  end
  return xOut, yOut, de
end

end
