# KernelCL.jl

[![DOI](https://zenodo.org/badge/552342814.svg)](https://zenodo.org/badge/latestdoi/552342814)

This is the code generating the simulations done in: 

To run the code you need a version of Julia installed, then you can make separate scripts or follow the AHO.jl file which can be run in bash using `julia --project=. AHO.jl` or line by line using the Julia vscode extension. Before you run the code follow the Instantite section below to setup the necessary packages.

## Instantiate

To initialize the project run these comments inside the Julia REPL (From inside the project directory)
```julia
    import Pkg
    Pkg.activate(".")
    Pkg.instantiate()
```
For more information see: https://docs.julialang.org/en/v1/stdlib/Pkg/

Now all dependencies should be downloaded and the code is ready to be run.

## Example 1
First a simple example to run the Anharmonic oscillator on the canonical Schwinger-Keldysh contour without a kernel up to 1.0 in real-time, and then plotting the result

```julia
M = AHO(;m=1.0,λ=24.,RT=1.0,β=1.0,steps_pr_length=10)
KP = KernelProblem(M;kernel=KernelCL.ConstantKernel(M,kernelType=:expA));
RS = RunSetup(tspan=30,NTr=30,saveat=0.05)

sol = run_sim(KP,RS)
l = KernelCL.calcTrueLoss(sol,KP)
display(plotSKContour(KP,sol))
println("True loss= ",l)
```


## Example 2

Second a simple example to learn a kernel for the Anharmonic oscillator on the canonical Schwinger-Keldysh contour without a kernel up to 1.0 in real-time

```julia
using KernelCL


M = AHO(;m=1.0,λ=24.,RT=1.0,β=1.0,steps_pr_length=10)
KP = KernelProblem(M;kernel=KernelCL.ConstantKernel(M,kernelType=:expA));
RS = RunSetup(tspan=30,NTr=30,saveat=0.05)


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


LK = LearnKernel(KP,30;runs_pr_epoch=5,
            runSetup=RS,
            opt=KernelCL.ADAM(0.002));

bestLSym, bestKP = learnKernel(LK, cb=cb)
```

Now we can use the optimal kernel and run once more with a higher statistics
```julia
println("Testing the optimal kernel")
RS_test = RunSetup(tspan=30,NTr=100)
sol = run_sim(bestKP,RS_test)
l = KernelCL.calcTrueLoss(sol,bestKP)
display(plotSKContour(bestKP,sol))
println("True loss: ", l,"\t Best LSym: ", bestLSym)
```

Then plot the loss functions
```julia
fig = KernelCL.plot(lhistory[:LSym],label=KernelCL.L"L^{\textrm{Sym}}",yaxis=:log)
KernelCL.plot!(fig,lhistory[:LTrue],label=KernelCL.L"L^{\textrm{True}}")
KernelCL.plot!(fig,lhistory[:L],label=KernelCL.L"L_D")
```