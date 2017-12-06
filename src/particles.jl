## these are pre-defined particle types, only for convenience / as examples

module Particles

using StaticArrays

mutable struct Int64Particle
  x::Int64
  Int64Particle() = new()
end

mutable struct Float64Particle
  x::Float64
  Float64Particle() = new()
end

struct MVFloat64Particle{d}
  x::MVector{d, Float64}
  MVFloat64Particle{d}() where d = new(MVector{d, Float64}())
end

end
