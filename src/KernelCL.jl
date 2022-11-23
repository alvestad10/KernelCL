module KernelCL

using LinearAlgebra, Statistics
using StochasticDiffEq
using Parameters, Measurements
using Plots, LaTeXStrings 
using JLD2
using SparseArrays
using Plots
using ForwardDiff, Zygote, Flux
using ThreadsX

include("model.jl")
include("kernel.jl")
include("problem.jl")
include("learnKernel.jl")

include("Implementations/imp_AHO.jl")
#include("Implementations/Imp_LM_AHO.jl")

include("solutions.jl")
include("plotScripts.jl")

end # module
