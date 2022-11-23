using KernelCL


M = AHO(1.,24.,0.5,1.0,10)
KP = KernelProblem(M;kernel=KernelCL.ConstantKernel(M,kernelType=:expA));
RS = RunSetup(tspan=10,NTr=10)

sol = run_sim(KP,RS)
KernelCL.calcTrueLoss(sol,KP)
plotSKContour(KP,sol)

