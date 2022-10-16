export LM_AHO, AHO

#################################

abstract type Model end



#################################

struct LM_AHO <: Model
    σ::Vector{Float64}
    λ::Float64

    function LM_AHO(σ::Vector{Float64},λ::Float64)
        size(σ) != (2,) && @warn "σ is of the wrong array size, it should be (2,) for this model"
        new(σ,λ)
    end

    function LM_AHO(σ::ComplexF64,λ::Float64)
        new([real(σ),imag(σ)],λ)
    end

    function LM_AHO(σ::Float64,λ::Float64)
        new([σ,0.],λ)
    end
end


##################################
"""
Schwinger-Keldysh contour
"""
struct SKContour
    a::Vector{ComplexF64}
    x0::Vector{ComplexF64}
    tp::Vector{Float64}
    RT::Float64
    β::Float64
    
    t_steps::Integer #Total number of steps
    FWRTSteps::Integer #Number of forward steps
    BWRTSteps::Integer #Number of Backward steps
    EucledianSteps::Integer #Number of eucledian steps (Id backward til this is 0)

    "Amount of tilt in the forward direction. Default=-1"
    ΔE::Float64 #Only used for amount of tilt, -1 if no tilt in both direction
    
    "Splitting up forward and backward pass like in Alexandru2016. Default = 0."
    Δβ::Float64
    
    "Applying cut to the action"
    κ::ComplexF64 # Regulator


    function SKContour(RT::Float64,β::Float64,steps_pr_length::Integer; Δβ = 0., ΔE = -1., κ = 0. * im)
        
        if Δβ == 0. && ΔE == -1.

            RT_points = floor(Integer,steps_pr_length*RT) + 1
            β_points = floor(Integer,steps_pr_length*β) + 1
            t_steps = floor(Integer,steps_pr_length*(2*RT + β))

            x0 = zeros(ComplexF64,t_steps+1)
            
            x0[1:RT_points] = range(0.,RT;length=RT_points)
            x0[RT_points+1:2*RT_points-1] = range(RT,0.;length=RT_points)[2:end]
            x0[2*RT_points:end] = -im*range(0.,β;length=β_points)[2:end]
            
            a = diff(x0)
            
            tp = [i==1 ? 0 : sum(abs.(a[1:i-1])) for i in 1:length(a)+1]

            new(a,x0,tp,RT,β,t_steps,RT_points,RT_points,β_points,-1.,Δβ,κ)
        
        elseif Δβ > 0. && ΔE == -1.
            
            Δβ != 0.5 && @warn "To use the split of the FW and BW only the 0.5 split is suported at the moment"

            RT_points = floor(Integer,steps_pr_length*RT) + 1
            β2_points = floor(Integer,steps_pr_length*β/2) + 1
            
            t_steps = floor(Integer,steps_pr_length*(RT + β/2 + RT + β/2))

            x0 = zeros(ComplexF64,t_steps+1)
            
            x0[1:RT_points] = range(0.,RT;length=RT_points)
            x0[RT_points+1:RT_points+β2_points-1] = RT .- im*range(0.,β/2;length=β2_points)[2:end]
            x0[RT_points+β2_points:2RT_points+β2_points-2] = range(RT,0.;length=RT_points)[2:end] .- im*β/2
            x0[2*RT_points+β2_points-1:end] = -im*range(β/2,β;length=β2_points)[2:end]
            
            a = diff(x0)
            
            tp = [i==1 ? 0 : sum(abs.(a[1:i-1])) for i in 1:length(a)+1]

            new(a,x0,tp,RT,β,t_steps,RT_points,RT_points,0,-1.,Δβ,κ)
            
        elseif Δβ == 0. && ΔE >= 0.

            FW_points = floor(Integer,steps_pr_length*sqrt(RT^2 + ΔE^2)) + 1
            BW_points = floor(Integer,steps_pr_length*sqrt((β-ΔE)^2 + RT^2)) + 1
            t_steps = floor(Integer,FW_points + BW_points - 2)
            x0 = zeros(ComplexF64,t_steps+1)
            
            x0[1:FW_points] = collect(range(0.,RT;length=FW_points)) .- im*collect(range(0.,ΔE;length=FW_points))
            x0[FW_points+1:end] = collect(range(RT,0.;length=BW_points))[2:end] .- im*collect(range(ΔE,β;length=BW_points))[2:end]
            
            a = diff(x0)
            
            tp = [i==1 ? 0 : sum(abs.(a[1:i-1])) for i in 1:length(a)+1]

            new(a,x0,tp,RT,β,t_steps,FW_points,BW_points,0,ΔE,Δβ,κ)
        end
    end
end

"""
    Anhamronic Oscillator model with the potential (1/2)mx^2 + (1/24)λx^4

    Containes the integration contour
"""
struct AHO <: Model
    m::Float64
    λ::Float64
    contour::SKContour

    function AHO(m::Float64,λ::Float64,RT::Float64,β::Float64,steps_pr_length::Integer;kwargs...)
        contour = SKContour(RT,β,steps_pr_length;kwargs...)
        new(m,λ,contour)
    end
end