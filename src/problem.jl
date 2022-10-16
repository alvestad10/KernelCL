export KernelProblem, RunSetup
export run_sim

struct KernelProblem{MType<:Model, KType<:Kernel,AType,BType,yType}
    model::MType
    kernel::KType
    a::AType
    b::BType
    y::yType
end

function KernelProblem(model::Model;kernel=ConstantKernel(model))

    a_func!, b_func! = get_ab(model,kernel)

    y = getSolutions(model)

    return KernelProblem(model,kernel,a_func!,b_func!,y)
end

function updateProblem(KP::KernelProblem)
    @unpack model, kernel, a, b, y = KP

    updateKernel!(kernel.pK)
    a_func!, b_func! = get_ab(model,kernel)

    return KernelProblem(model, kernel, a_func!, b_func!, y) 
end


@with_kw mutable struct RunSetup
    tspan=20
    NTr = 10
    saveat=0.01
    scheme=ImplicitEM() 
    dt=1e-4
    abstol=1e-3
    reltol=1e-3
    dtmax=1e-3
    adaptive=true
    tspan_thermalization=5
    dt_thermalization = 1e-3
    scheme_thermalization=ImplicitEM()
    dtmax_thermalization=1e-1
    abstol_thermalization=1e-2
    reltol_thermalization=1e-2
end



function run_sim(KP::KernelProblem, runSetup::RunSetup)
    @unpack tspan, NTr, saveat,scheme, dt, abstol, reltol, dtmax, adaptive, tspan_thermalization, dt_thermalization,scheme_thermalization, dtmax_thermalization, abstol_thermalization, reltol_thermalization = runSetup
#end


#function run_sim(KP::KernelProblem; tspan=20, NTr = 10, saveat=0.01, scheme=ImplicitEM(), 
#                dt=1e-4, abstol=1e-3, reltol=1e-3, dtmax=1e-3,adaptive=true,tspan_thermalization=5)

    @unpack kernel, a, b, model = KP
    K = getKernelParamsSim(kernel)

    if model isa AHO 
        u0 = zeros(eltype(getKernelParams(kernel)),2*model.contour.t_steps)
        noise_rate_prototype = zeros(eltype(getKernelParams(kernel)),2*model.contour.t_steps,model.contour.t_steps)
    else
        u0 = zeros(2)
        noise_rate_prototype = zeros(2,1)
    end

    function prob_init_func(prob,i,repeat)
        if model isa AHO
            prob.u0 .= [rand(model.contour.t_steps); zeros(model.contour.t_steps)]
        else
            prob.u0 .= [randn();0.]
        end
        prob
    end

    if !all(kernel.pK.sqrtK .== kernel.pK.K[:,1:div(end,2)]) # If not identity kernel
        prob_thermalization = SDEProblem(a,b,u0,(0.0,tspan_thermalization),K, noise_rate_prototype=noise_rate_prototype)
    else
        prob_thermalization = SDEProblem(a,b,u0,(0.0,tspan_thermalization),K)
    end
    
    ensembleProb_init = EnsembleProblem(prob_init,prob_func=prob_init_func)
    sol_init = solve(ensembleProb_init, scheme_thermalization,
                    EnsembleThreads(); trajectories=NTr, 
                    dt=dt_thermalization, saveat=[tspan_init], #save_start=false,
                    dtmax=dtmax_thermalization,abstol=abstol_thermalization,reltol=reltol_thermalization)

    function prob_func(prob,i,repeat)
        prob.u0 .= sol_init[i].u[end]
        prob
    end

    if !all(kernel.pK.sqrtK .== kernel.pK.K[:,1:div(end,2)]) # If not identity kernel
        prob = SDEProblem(a,b,u0,(0.0,tspan),K,noise_rate_prototype=noise_rate_prototype)
    else
        prob = SDEProblem(a,b,u0,(0.0,tspan),K)
    end

ensembleProb = EnsembleProblem(prob,prob_func=prob_func)
return solve(ensembleProb,
    scheme,
    EnsembleThreads(); trajectories=NTr, 
    adaptive=adaptive,
    dt=dt, 
    saveat=saveat, save_start=false,
    dtmax=dtmax,
    abstol=abstol,reltol=reltol)
end



