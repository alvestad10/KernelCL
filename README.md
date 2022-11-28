# KernelCL.jl

This is the code generating the simulations done in: 

To run the code you need a version of Julia installed, then you can make separate scripts or follow the AHO.jl file which can be run line by line using the Julia vscode extension.

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
M = AHO(1.,24.,1.0,1.0,10)
KP = KernelProblem(M;kernel=KernelCL.ConstantKernel(M,kernelType=:expA));
RS = RunSetup(tspan=30,NTr=30)

sol = run_sim(KP,RS)
l = KernelCL.calcTrueLoss(sol,KP)
plotSKContour(KP,sol)
println("True loss= ",l)
```


## Example 2

Second a simple example to learn a kernel for the Anharmonic oscillator on the canonical Schwinger-Keldysh contour without a kernel up to 1.0 in real-time

```julia
using KernelCL

M = AHO(1.,24.,1.0,1.0,10)
KP = KernelProblem(M;kernel=KernelCL.ConstantKernel(M,kernelType=:expA));
RS = RunSetup(tspan=30,NTr=30)

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

l, bestKP = learnKernel(LK, cb=cb)
```

Now we can use the optimal kernel and run once more with a higher statistics
```julia
println("Testing the optimal kernel")
RS_test = RunSetup(tspan=30,NTr=100)
l = KernelCL.calcTrueLoss(sol,KP)
plotSKContour(KP,sol)
println("True loss: ", l)
```

Then plot the loss functions
```julia
fig = KernelCL.plot(lhistory[:LSym],label=KernelCL.L"L^{\textrm{Sym}}",yaxis=:log)
KernelCL.plot!(fig,lhistory[:LTrue],label=KernelCL.L"L^{\textrm{True}}")
KernelCL.plot!(fig,lhistory[:L],label=KernelCL.L"L_D")
```