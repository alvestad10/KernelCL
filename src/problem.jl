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

Base.copy(KP::KernelProblem) = KernelProblem(KP.model;kernel=KP.kernel)

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



function get_caches(T::Type,t_steps)
    tmp = zeros(T,8t_steps)
    return dualcache(tmp)
end

function run_sim(KP::KernelProblem, runSetup::RunSetup)
    @unpack tspan, NTr, saveat,scheme, dt, abstol, reltol, dtmax, adaptive, tspan_thermalization = runSetup

    @unpack kernel, a, b, model = KP
    K = getKernelParamsSim(kernel)

    if model isa AHO 
        u0 = zeros(eltype(getKernelParams(kernel)),2*model.contour.t_steps)
        noise_rate_prototype = zeros(eltype(getKernelParams(kernel)),2*model.contour.t_steps,model.contour.t_steps)
        caches = [get_caches(Float64,model.contour.t_steps) for _ in 1:NTr]
    else
        u0 = zeros(2)
        noise_rate_prototype = zeros(2,1)
    end

    function prob_func(prob,i,repeat)
        if model isa AHO
            prob.u0 .= [rand(model.contour.t_steps); zeros(model.contour.t_steps)]
            prob = remake(prob,seed = 100 + i,p=caches[i])
        else
            prob.u0 .= [randn();0.]
            prob = remake(prob,seed = 100 + i)
        end
        prob
    end

    prob = SDEProblem(a,b,u0,(0.0,tspan),K,noise_rate_prototype=noise_rate_prototype)

    ensembleProb = EnsembleProblem(prob,prob_func=prob_func)
    return solve(ensembleProb,
                scheme,
                EnsembleThreads(); trajectories=NTr, 
                adaptive=adaptive,
                dt=dt, 
                saveat=tspan_thermalization:saveat:tspan, save_start=false,
                dtmax=dtmax,
                abstol=abstol,reltol=reltol)
end



function trueSolution(KP::KernelProblem, runSetup::RunSetup)
    sol = run_sim(KP,runSetup)
    
end