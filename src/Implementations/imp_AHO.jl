
"""
    Get the drift and noise term to be used in the simulation
"""
function get_ab(model::AHO,kernel::ConstantKernel{T}) where {T <: AHOConstantKernelParameters}

    @unpack m, λ, contour = model
    @unpack a, t_steps, κ = contour
    @unpack sqrtK,K = kernel.pK
    
    KC = K[1:div(end,2),1:div(end,2)] .+ im*K[div(end,2)+1:end,1:div(end,2)]

    gm1=vcat([t_steps],1:t_steps-1)
    gp1=vcat(2:t_steps,[1])

    pre_fac = (1 / abs(a[1]))

    im_pre_fac_KC = KC*im*pre_fac
    sqrt2pre_fac__sqrtK = sqrt(2 * pre_fac)*sqrtK


    function a_func!(du,u,p,t)

        _x = @. (@view u[1:t_steps]) + im * (@view u[t_steps+1:end])
        _A = im_pre_fac_KC * ( 
                    (_x .- _x[gm1]) ./ a[gm1] + (_x .- _x[gp1]) ./ a
                    .- (a .+ a[gm1])/2 .* (m .* _x .+ (λ/6) .* _x.^3)
            )
        
        du[1:t_steps] .= real(_A)
        du[t_steps+1:end] .= imag(_A)

    end
    
    function b_func!(du,u,p,t)
        du .= sqrt2pre_fac__sqrtK
    end 

    return a_func!, b_func!

end

"""
    Calculate the observables for each of the trajectories
"""
function calc_obs(KP::KernelProblem{AHO},sol)
    t_steps = KP.model.contour.t_steps

    T = eltype( getKernelParams(KP.kernel) )
    avgRe = zeros(T,length(sol),t_steps)
    avgIm = zeros(T,length(sol),t_steps)
    avg2Re = zeros(T,length(sol),t_steps)
    avg2Im = zeros(T,length(sol),t_steps)
    corr0tRe = zeros(T,length(sol),t_steps)
    corr0tIm = zeros(T,length(sol),t_steps)

    for i in eachindex(sol)
        _u = hcat(sol[i].u...)
        avgRe[i,:] .= mean(_u[1:t_steps,:],dims=2)[:,1]
        avgIm[i,:] .= mean(_u[t_steps+1:end,:],dims=2)[:,1]

        x2Re = _u[1:t_steps,:].^2 .- _u[t_steps+1:end,:].^2
        x2Im = 2 .* _u[1:t_steps,:] .* _u[t_steps+1:end,:]

        avg2Re[i,:] .= mean(x2Re,dims=2)[:,1]
        avg2Im[i,:] .= mean(x2Im,dims=2)[:,1]
        corr0tRe[i,:] .= mean(_u[1:1,:] .* _u[1:t_steps,:] .- _u[t_steps+1:t_steps+1,:].*_u[t_steps+1:end,:],dims=2)[:,1]
        corr0tIm[i,:] .= mean(_u[1:t_steps,:] .* _u[t_steps+1:t_steps+1,:] .+ _u[1:1,:] .* _u[t_steps+1:end,:],dims=2)[:,1]
    end


    return avgRe, avgIm, avg2Re, avg2Im, corr0tRe, corr0tIm
end

"""
    Calculate the observable means over the different trajectories with the corresponding error
"""
function calc_meanObs(::KernelProblem{AHO},obs,NTr)
    avgRe, avgIm, avg2Re, avg2Im, corr0tRe, corr0tIm = obs
    d = 1
    return mean(avgRe,dims=d)[1,:], (std(avgRe,dims=d)/sqrt(NTr))[1,:], 
           mean(avgIm,dims=d)[1,:], (std(avgIm,dims=d)/sqrt(NTr))[1,:],
           mean(avg2Re,dims=d)[1,:], (std(avg2Re,dims=d)/sqrt(NTr))[1,:], 
           mean(avg2Im,dims=d)[1,:], (std(avg2Im,dims=d)/sqrt(NTr))[1,:],
           mean(corr0tRe,dims=d)[1,:], (std(corr0tRe,dims=d)/sqrt(NTr))[1,:], 
           mean(corr0tIm,dims=d)[1,:], (std(corr0tRe,dims=d)/sqrt(NTr))[1,:]
end

"""
    Caluclate the true loss
"""
function calcTrueLoss(sol,KP::KernelProblem{AHO}; obs_calc = nothing)
    
    if isnothing(obs_calc)
        obs = calc_obs(KP,sol)
        avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = calc_meanObs(KP,obs,length(sol)) 
    else
        avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = obs_calc
    end

    normalized_err = maximum(err_avg2Re ./ real(KP.y["x2"]))

    return normalized_err * sum(abs2,[ (real(KP.y["x"]) .- avgRe) ./ err_avgRe; 
    (imag(KP.y["x"]) .- avgIm) ./  err_avgIm; 
    (real(KP.y["x2"]) .- avg2Re) ./  err_avg2Re; 
    (imag(KP.y["x2"]) .- avg2Im) ./ err_avg2Im;
    (real(KP.y["corr0t"]) .- corr0tRe) ./ err_corr0tRe; 
    (imag(KP.y["corr0t"]) .- corr0tIm) ./ err_corr0tIm])
end

"""
    Calculate the loss based on the symmetry prior knowledge
"""
function calcSymLoss(sol,KP::KernelProblem{AHO}; obs_calc = nothing)

    
    βsteps = KP.model.contour.EucledianSteps

    if isnothing(obs_calc)
        obs = calc_obs(KP,sol)
        avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = calc_meanObs(KP,obs,length(sol)) 
    else
        avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = obs_calc
    end


    normalized_err = maximum(err_avg2Re ./ real(KP.y["x2"]))

    return normalized_err * sum(abs2, [ (real(KP.y["x"]) .- avgRe) ./ err_avgRe;
                    (imag(KP.y["x"]) .- avgIm) ./  err_avgIm;
                    (real(KP.y["x2"]) .- avg2Re) ./  err_avg2Re; 
                    (imag(KP.y["x2"]) .- avg2Im) ./ err_avg2Im;
                    (real(KP.y["corr0t"][end-βsteps]) .- corr0tRe)[end-βsteps] ./ err_corr0tRe[end-βsteps]; 
                    (imag(KP.y["corr0t"][end-βsteps]) .- corr0tIm)[end-βsteps] ./ err_corr0tIm[end-βsteps]])
end


"""
    Calculate the drift loss used to update approximate the gradient
"""
function calcDriftLoss(sol,KP::KernelProblem{AHO,T};p=getKernelParams(KP.kernel)) where {T <: ConstantKernel}

    @unpack m, λ, contour = KP.model
    @unpack a, t_steps, κ = contour
    @unpack K, sqrtK = KP.kernel


    ξ = 1.
    


    gm1=vcat([t_steps],1:t_steps-1)
    gp1=vcat(2:t_steps,[1])
    
    pre_fac = (1 / abs(a[1]))
    
    KRe,KIm = K([],p)
    KC = KRe .+ im*KIm
    
    im_pre_fac_KC = KC*im*pre_fac    

    g(u) = begin
        _x = (@view u[1:t_steps]) + im * (@view u[t_steps+1:end])
        
        _A_tmp = @. (_x - _x[gm1]) / a[gm1] + (_x - _x[gp1]) / a - (a + a[gm1])/2 * (m * _x + (λ/6) * _x^3)
        _A = im_pre_fac_KC * _A_tmp
        #mul!(_A, im_pre_fac_KC, copy(_A_tmp))
        return abs(real(adjoint(_A) * (-_x)) - norm(_A) * norm(_x))^ξ

    end

    return sum(
                mean(g(u) for u in eachrow(sol'))
            )

end
