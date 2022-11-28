export LearnKernel, learnKernel
export ADAM

mutable struct LearnKernel
    KP::KernelProblem
    
    # Optimization parameters
    opt::Flux.Optimise.AbstractOptimiser
    epochs::Integer
    runs_pr_epoch::Integer
    
    # Simulation parameters
    runSetup::RunSetup
end

function LearnKernel(KP::KernelProblem,epochs;runs_pr_epoch=1,runSetup=runSetup(),opt=ADAM(0.05))
    return LearnKernel(KP,opt,epochs,runs_pr_epoch,runSetup)
end

function updatelr!(LK::LearnKernel,lr)
    LK.opt.eta = lr
end


function StartingLKText(LK::LearnKernel)
    @unpack epochs, runs_pr_epoch, runSetup = LK
    @unpack NTr, tspan, saveat = runSetup

    println("*************************************")
    println("***** Starting learning kernel ******")
    println("  Total epochs is ", epochs, " with ", runs_pr_epoch, " runs pr. epoch")
    println("  NTr=", NTr)
    println("  tspan=", tspan)
    println("  saveat=", saveat)
    println("*************************************")
end

##### LearnKernel shorthand functions ####
function run_sim(LK::LearnKernel)
    return run_sim(LK.KP,LK.runSetup)
end

"""
   CHECK FOR ERRORS/WARNINGS DURING RUN

   can also remove warning trajectories by setting
                remove_warning_tr=true
"""
function check_warnings!(sol) #;remove_warning_tr=true)
    warnings_inxes = []
    for (i,s) in enumerate(sol) 
        if (s.retcode != :Success)
            push!(warnings_inxes,i) 

        end
    end
        
    deleteat!(sol.u,warnings_inxes)
    return isempty(sol)
end


########

function d_driftOpt(LK::LearnKernel;sol=nothing,p = getKernelParams(LK.KP.kernel))
    
    @unpack KP, runSetup = LK
    NTr = runSetup.NTr

    if isnothing(sol)
        sol = run_sim(LK)
    end

    g(_p,i) = calcDriftLoss(reduce(hcat,sol[i].u),KP;p=_p)


    if length(p) > 10
        return ThreadsX.sum((i) -> Zygote.gradient((p) -> g(p,i),p)[1],1:NTr) ./ NTr
    else
        return ForwardDiff.gradient((p) -> calcDriftLoss(sol,KP;p=p),p)
    end
    
    
end



function learnKernel(LK::LearnKernel; cb=(LK::LearnKernel; sol=nothing, addtohistory=false) -> ())

    @unpack KP, opt, epochs, runs_pr_epoch, runSetup = LK
    @unpack kernel = KP
    @unpack tspan, NTr, saveat = runSetup


    ### Text to the user
    StartingLKText(LK)

    if KP.kernel isa ConstantKernel
        LK.KP = updateProblem(KP)
    end

    ###### Getting initial configurations
    trun = @elapsed sol = run_sim(LK)
    if check_warnings!(sol)
        @warn "All trajectories diverged"
        return 0
    end
    l = cb(LK; sol = sol, addtohistory=true)
    
    # initialize the derivative observable
    dKs = similar(getKernelParams(kernel))
    bestKernel = copy(KP)
    bestLSym = l
    
    for i in 1:epochs
        
        unstable = false

        println("EPOCH ", i, "/", epochs, "\t (time_run=",trun,")")
        
        for j in 1:runs_pr_epoch
            
            tdL = @elapsed dKs = d_driftOpt(LK;sol=sol)

            # Updating the kernel parameters
            Flux.update!(opt, getKernelParams(kernel), dKs)

            if KP.kernel isa ConstantKernel
                LK.KP = updateProblem(KP)
            end
            
            LD = mean(calcDriftLoss(reduce(hcat,sol[tr].u),KP) for tr in eachindex(sol))
            println("LDrift=",round(LD,digits=5), "\t (time_grad: ", round(tdL,digits=2), ")")
            
        end

        trun = @elapsed sol = run_sim(LK)
        if check_warnings!(sol)
            @warn "All trajectories diverged"
            unstable = true
        end

        if unstable
            break
        end

        l = cb(LK; sol=sol, addtohistory=true)

        if l < bestLSym
            bestLSym = l
            bestKernel = copy(KP)
        end
    end
    return bestLsym, bestKernel
end
