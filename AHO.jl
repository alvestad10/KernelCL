using KernelCL


M = AHO(1.,24.,1.0,1.0,10)
KP = KernelProblem(M;kernel=KernelCL.ConstantKernel(M,kernelType=:expA));
RS = RunSetup(tspan=50,NTr=50,saveat=0.05)


function get_new_lhistory()
    return Dict(:L => Float64[], 
                :LTrue => Float64[], 
                :LSym => Float64[])
end

cb(LK::LearnKernel;sol=nothing,addtohistory=false) = begin
    KP = LK.KP
    
    if isnothing(sol)
        sol = run_sim(KP,tspan=LK.tspan_test,NTr=LK.NTr_test)
        if check_warnings!(sol)
            @warn "All trajectories diverged"
        end
    end

    LTrain = KernelCL.mean(KernelCL.calcDriftLoss(reduce(hcat,sol[tr].u),KP) for tr in eachindex(sol))
    TLoss = KernelCL.calcTrueLoss(sol,KP)
    LSym =  KernelCL.calcSymLoss(sol,KP)

    println("LTrain: ", round(LTrain,digits=5), ", TLoss: ", round(TLoss,digits=5), ", LSym: ", round(LSym,digits=5))

    display(KernelCL.plotSKContour(KP,sol))
    #display(KernelCL.plotFWSKContour(KP,KP,sol))

    if addtohistory
        append!(lhistory[:L],LTrain)
        append!(lhistory[:LTrue],TLoss)
        append!(lhistory[:LSym],LSym)
    end
    return LSym
end

lhistory = get_new_lhistory()


LK = LearnKernel(KP,10;runs_pr_epoch=5,
            runSetup=RS,
            opt=KernelCL.ADAM(0.002));

bestLSym, bestKP = learnKernel(LK, cb=cb)

println("Testing the optimal kernel")
RS_test = RunSetup(tspan=30,NTr=100)
sol = run_sim(bestKP,RS_test)
l = KernelCL.calcTrueLoss(sol,bestKP)
plotSKContour(bestKP,sol)
println("True loss: ", l,"\t Best LSym: ", bestLSym)


begin
fig = KernelCL.plot(lhistory[:LSym],label=KernelCL.L"L^{\textrm{Sym}}",yaxis=:log)
KernelCL.plot!(fig,lhistory[:LTrue],label=KernelCL.L"L^{\textrm{True}}")
KernelCL.plot!(fig,lhistory[:L],label=KernelCL.L"L_D")
end
