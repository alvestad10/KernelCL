
"""
    Get the drift and noise term to be used in the simulation
"""
function get_ab(model::AHO,kernel::ConstantKernel{T}) where {T <: AHOConstantKernelParameters}
    
    @unpack m, λ, contour = model
    @unpack a, t_steps, κ = contour
    @unpack sqrtK,K = kernel.pK
    
    KRe = K[1:div(end,2),1:div(end,2)]
    KIm = K[div(end,2)+1:end,1:div(end,2)]

    gm1=vcat([t_steps],1:t_steps-1)
    gp1=vcat(2:t_steps,[1])

    a_m1 = a[gm1]

    one_over_a = 1 ./ a
    one_over_a_Re = real(one_over_a)
    one_over_a_Im = imag(one_over_a)
    one_over_a_m1 = 1 ./ a_m1
    one_over_a_m1_Re = real(one_over_a_m1)
    one_over_a_m1_Im = imag(one_over_a_m1)
    
    V_pre_fac = (a + a_m1)/2
    V_pre_fac_Re = real(V_pre_fac)
    V_pre_fac_Im = imag(V_pre_fac)

    pre_fac = (1 / abs(a[1]))
    #im_pre_fac = im*pre_fac

    function a_func!(du,u,p,t)


        tmp = get_tmp(p, u)
        ARe = @view tmp[1:t_steps]
        AIm = @view tmp[t_steps .+ (1:t_steps)]
        VRe = @view tmp[2t_steps .+ (1:t_steps)]
        VIm = @view tmp[3t_steps .+ (1:t_steps)]
        B1 = @view tmp[4t_steps .+ (1:t_steps)]
        B2 = @view tmp[5t_steps .+ (1:t_steps)]
        B3 = @view tmp[6t_steps .+ (1:t_steps)]
        B4 = @view tmp[7t_steps .+ (1:t_steps)]

        uRe = @view u[1:t_steps]
        uRem1 = @view u[gm1]
        uRep1 = @view u[gp1]
        uIm = @view u[t_steps+1:end]
        uImm1 = @view u[t_steps .+ gm1]
        uImp1 = @view u[t_steps .+ gp1]
        
        @. VRe = m * uRe + (λ/6) * (uRe^3 - 3*uRe*uIm^2)
        @. VIm = m * uIm - (λ/6) * (uIm^3 - 3*uIm*uRe^2)

        @. ARe = -pre_fac * (
            (uRe - uRem1)*one_over_a_m1_Im 
           + (uIm - uImm1)*one_over_a_m1_Re  
           + (uRe - uRep1)*one_over_a_Im 
           + (uIm - uImp1)*one_over_a_Re
           - V_pre_fac_Im * VRe #(m * uRe + (λ/6) * u3Re)
           - V_pre_fac_Re * VIm #(m * uIm + (λ/6) * u3Im)
        )

        @. AIm =  pre_fac * (
            (uRe - uRem1)*one_over_a_m1_Re 
           - (uIm - uImm1)*one_over_a_m1_Im  
           + (uRe - uRep1)*one_over_a_Re 
           - (uIm - uImp1)*one_over_a_Im
           - V_pre_fac_Re * VRe #(m * uRe + (λ/6) * u3Re)
           + V_pre_fac_Im * VIm #(m * uIm + (λ/6) * u3Im)
        )

        
        mul!(B1,KRe,ARe)
        mul!(B2,KIm,AIm)
        mul!(B3,KIm,ARe)
        mul!(B4,KRe,AIm)
    
        @. du[1:t_steps] = B1 - B2
        @. du[t_steps+1:end] = B3 + B4
        nothing
    end
    
    sqrt2pre_fac = sqrt(2 * pre_fac) 
    function b_func!(du,u,p,t)
        
        @. du = sqrt2pre_fac * sqrtK
        nothing
    end 

    return a_func!, b_func!
end


function getJac(model::AHO,kernel::ConstantKernel{T}) where {T <: AHOConstantKernelParameters}


    @unpack m, λ, contour = model
    @unpack a, t_steps, κ = contour
    @unpack K = kernel.pK

    N = t_steps
    
    KRe = @view K[1:div(end,2),1:div(end,2)]
    KIm = @view K[div(end,2)+1:end,1:div(end,2)]

    gm1=vcat([t_steps],1:t_steps-1)

    a_m1 = @view a[gm1]

    pre_fac = (1 / abs(a[1]))

    # Jacobian
    d2Sdphilj_C = BandedMatrix(Zeros{ComplexF64}(N,N), (1,1))
    d2Sdphilj_C[band(0)] .= im*pre_fac .* 
                    ((1 ./ a_m1) .+ (1 ./ a) .- ((a_m1 .+ a) ./ 2).*m)
    d2Sdphilj_C[band(-1)] .= im*pre_fac .* (( -1 ./ a[gm1[2:end]]))
    d2Sdphilj_C[band(1)] .= im*pre_fac .* (( -1 ./ a[1:end-1]))
    d2Sdphilj_C = sparse(d2Sdphilj_C)
    d2Sdphilj_C[1,N] = im*pre_fac * (( -1 / a[gm1[1]]))
    d2Sdphilj_C[N,1] = im*pre_fac * (( -1 / a[end]))
    d2Sdphilj_Re = real(d2Sdphilj_C)
    d2Sdphilj_Im = imag(d2Sdphilj_C)

    im_prefac_a_λ2 = im*pre_fac*((a_m1 + a) / 2) * (λ/2)
    im_prefac_a_λ2_Re = real(im_prefac_a_λ2)
    im_prefac_a_λ2_Im = imag(im_prefac_a_λ2)

    #function getJacCache()
    #    DiffCache(hcat(copy(d2Sdphilj_Re),copy(d2Sdphilj_Im)))
    #end

    function a_jac!(J,u,p,t)

        tmp = get_tmp(p, u)
        _x2Re = @view tmp[1:t_steps]
        _x2Im = @view tmp[t_steps .+ (1:t_steps)]

        _pre_x2Re = @view tmp[2t_steps .+ (1:t_steps)]
        _pre_x2Im = @view tmp[3t_steps .+ (1:t_steps)]


        uRe = @view u[1:t_steps]
        uIm = @view u[t_steps+1:end]

        @. _x2Re = (uRe^2 - uIm^2)
        @. _x2Im = 2 * uRe * uIm

        @. _pre_x2Re = im_prefac_a_λ2_Re * _x2Re - im_prefac_a_λ2_Im * _x2Im
        @. _pre_x2Im = im_prefac_a_λ2_Re * _x2Im + im_prefac_a_λ2_Im * _x2Re

        _d2Sdphilj_Re = copy(d2Sdphilj_Re)
        _d2Sdphilj_Im = copy(d2Sdphilj_Im)

        @simd for i in 1:t_steps
            #@inbounds _d2Sdphilj_C[i,i] = d2Sdphilj_C[i,i] - _x2[i]
            @inbounds _d2Sdphilj_Re[i,i] = d2Sdphilj_Re[i,i] - _pre_x2Re[i]
            @inbounds _d2Sdphilj_Im[i,i] = d2Sdphilj_Im[i,i] - _pre_x2Im[i]
            #@inbounds _tmpRe[i] = d2Sdphilj_Re[i,i] - _pre_x2Re[i]
            #@inbounds _tmpIm[i] = d2Sdphilj_Im[i,i] - _pre_x2Im[i]
        end


        #mul!(B1,KRe,_d2Sdphilj_Re)
        #mul!(B2,KIm,_d2Sdphilj_Im)
        #mul!(B3,KRe,_d2Sdphilj_Im)
        #mul!(B4,KIm,_d2Sdphilj_Re)

        B1 = KRe*_d2Sdphilj_Re
        B2 = KIm*_d2Sdphilj_Im
        B3 = KRe*_d2Sdphilj_Im
        B4 = KIm*_d2Sdphilj_Re

        @. J[1:t_steps,1:t_steps] = B1 - B2
        J[t_steps+1:end,t_steps+1:end] .= @view J[1:t_steps,1:t_steps]

        @. J[1:t_steps,t_steps+1:end] = B3 + B4
        J[t_steps+1:end,1:t_steps] .= (@view J[1:t_steps,t_steps+1:end])
        J[t_steps+1:end,1:t_steps] .*= -1
        nothing
    end

    return a_jac!
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
