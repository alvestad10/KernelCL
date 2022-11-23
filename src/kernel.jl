export setKernel!

########################

abstract type Kernel end

abstract type KernelParameters end

abstract type ConstantKernelParameters <: KernelParameters end
abstract type AHOConstantKernelParameters <: ConstantKernelParameters end


function getKernelParams(pK::T) where {T <: AHOConstantKernelParameters}
    return pK.p
end

function getKernelParamsSim(::T) where {T <: AHOConstantKernelParameters}
    return []
end

function dK(u,p,::T) where {T <: AHOConstantKernelParameters}
    return nothing
end

function K_dK(u,p,pK::T) where {T <: AHOConstantKernelParameters}
    return K(u,p,pK),dK(u,p,pK)
end





"""
Constant kernel with the entries of the complex matrix K as the parameters.
We get the ``H`` by built in square root ``H=sqrt(K)``
"""
mutable struct AHO_ConstKernel_K <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_K(p)

    _K = p[1:div(end,2),:] + im*p[div(end,2)+1:end,:]

    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    H = [real(_H);imag(_H)]

    #H = real(sqrt(K))[:,1:div(end,2)]

    return AHO_ConstKernel_K(p,H,K)
end


function AHO_ConstKernel_K(M::AHO)
    @unpack t_steps = M.contour 
    
    p = Matrix([Diagonal(ones(t_steps)) ; zeros(t_steps,t_steps)])

    return AHO_ConstKernel_K(p)
end

function updateKernel!(pK::AHO_ConstKernel_K)
    _K = pK.p[1:div(end,2),:] + im*pK.p[div(end,2)+1:end,:]    
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    pK.sqrtK = [real(_H);imag(_H)]

    #pK.sqrtK = real(sqrt(pK.K))[:,1:div(end,2)]
end

function K(u,p,::AHO_ConstKernel_K)
    _K = p[1:div(end,2),:] + im*p[div(end,2)+1:end,:]
    return real(_K),imag(_K)
end


function sqrtK(u,p,::AHO_ConstKernel_K)
    _K = p[1:div(end,2),:] + im*p[div(end,2)+1:end,:]
    _H = sqrt(_K)
    return [real(_H);imag(_H)]  
    
    #K = hcat([real(_K);imag(_K)],[-imag(_K);real(_K)])
    #return real(sqrt(K))[:,1:div(end,2)]
end


"""
Constant kernel with the entries of the complex matrix H as the parameters.
We get the kernel by squaring ``K=H^2``
"""
mutable struct AHO_ConstKernel_H <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_H(p)
    sqrtK = p
    KRe = (transpose(p[1:div(end,2),:])*p[1:div(end,2),:]
         - transpose(p[div(end,2)+1:end,:])*p[div(end,2)+1:end,:])
        
    KIm = (transpose(p[1:div(end,2),:])*p[div(end,2)+1:end,:] 
         + transpose(p[div(end,2)+1:end,:])*p[1:div(end,2),:])
    
    K = hcat([KRe;KIm],[-KIm;KRe])

    return AHO_ConstKernel_H(p,sqrtK,K)
end

function AHO_ConstKernel_H(M::AHO)
    @unpack t_steps = M.contour 
    kRe = Matrix( Diagonal(ones(t_steps))) 
    kIm = zeros(t_steps,t_steps)
    K = vcat(kRe,kIm)
    return AHO_ConstKernel(K)
end


function updateKernel!(pK::AHO_ConstKernel_H)
    @unpack p = pK
    KRe = (transpose(p[1:div(end,2),:])*p[1:div(end,2),:]
         - transpose(p[div(end,2)+1:end,:])*p[div(end,2)+1:end,:])
        
    KIm = (transpose(p[1:div(end,2),:])*p[div(end,2)+1:end,:] 
         + transpose(p[div(end,2)+1:end,:])*p[1:div(end,2),:])
    

    pK.sqrtK = p
    pK.K = hcat([KRe;KIm],[-KIm;KRe])
end

function K(u,p,::AHO_ConstKernel_H)

    KRe = (transpose(p[1:div(end,2),:])*p[1:div(end,2),:]
    - transpose(p[div(end,2)+1:end,:])*p[div(end,2)+1:end,:])
    
    KIm = (transpose(p[1:div(end,2),:])*p[div(end,2)+1:end,:] 
    + transpose(p[div(end,2)+1:end,:])*p[1:div(end,2),:])
    

    return KRe,KIm
end


function sqrtK(u,p,::AHO_ConstKernel_H)
    return p
end





"""
Constant kernel with the entries of the matrix defined by the entries
``
H_{ij} = s_{ij}(\\cos(θ_{ij}) + \\sin(θ_{ij})
``
such that the parameters is the real matrices ``θ`` and ``s``. The kernel is
obtained by squaring:``K=H^2``.
"""
mutable struct AHO_ConstKernel_sincos <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_sincos(θ,s)
    HRe = s .* cos.(θ)
    HIm = s .* sin.(θ)

    KRe = (transpose(HRe)*HRe
         - transpose(HIm)*HIm)
        
    KIm = (transpose(HRe)*HIm
         + transpose(HIm)*HRe)

    sqrtK = [HRe;HIm]
    K = hcat([KRe;KIm],[-KIm;KRe])
    
    return AHO_ConstKernel_sincos([θ;s],sqrtK,K)
end

function ConstantKernel_sincos(M::AHO)
    @unpack t_steps = M.contour 
    θ = zeros(t_steps,t_steps)
    s = Matrix( Diagonal(ones(t_steps))) 
    
    return AHO_ConstKernel_sincos(θ,s)
end

function updateKernel!(pK::AHO_ConstKernel_sincos)

    θ = pK.p[1:div(end,2),:]
    s = pK.p[div(end,2)+1:end,:]

    HRe = s .* cos.(θ)
    HIm = s .* sin.(θ)

    KRe = (transpose(HRe)*HRe
         - transpose(HIm)*HIm)
        
    KIm = (transpose(HRe)*HIm
         + transpose(HIm)*HRe)

    pK.sqrtK = [HRe;HIm]
    pK.K = hcat([KRe;KIm],[-KIm;KRe])
end

function K(u,p,::AHO_ConstKernel_sincos)

    θ = p[1:div(end,2),:]
    s = p[div(end,2)+1:end,:]

    HRe = s .* cos.(θ)
    HIm = s .* sin.(θ)

    KRe = (transpose(HRe)*HRe
         - transpose(HIm)*HIm)
        
    KIm = (transpose(HRe)*HIm
         + transpose(HIm)*HRe)
    return KRe,KIm
end

function sqrtK(u,p,::AHO_ConstKernel_sincos)

    θ = p[1:div(end,2),:]
    s = p[div(end,2)+1:end,:]

    HRe = s .* cos.(θ)
    HIm = s .* sin.(θ)
    
    return [HRe;HIm]
end





"""
Constant kernel with the entries of the real matrix P such that the kernel becomes
```math
K = e^{iP}
```
We get ``H`` by the built in square root ``sqrt(K)``.
"""
mutable struct AHO_ConstKernel_expiP <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_expiP(p)
    _K = exp(im*p)
    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    H = [real(_H);imag(_H)]
    
    return AHO_ConstKernel_expiP(p,H,K)
end

function AHO_ConstKernel_expiP(M::AHO)
    @unpack t_steps = M.contour 
    
    p = zeros(t_steps,t_steps)

    return AHO_ConstKernel_expiP(p)
end


function updateKernel!(pK::AHO_ConstKernel_expiP)

    _K = exp(im*pK.p)
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    pK.sqrtK = [real(_H);imag(_H)]
end

function K(u,p,::AHO_ConstKernel_expiP)
    _K = exp(im*p)
    return real(_K),imag(_K)
end


function sqrtK(u,p,::AHO_ConstKernel_expiP)
    _K = exp(im*p)
    _H = sqrt(_K)
    return [real(_H);imag(_H)]    
end

"""
Constant kernel with the entries of the real matrix P such that the kernel becomes
```math
K = e^{A}
```
We get ``H`` by the built in square root ``sqrt(K)``.
"""
mutable struct AHO_ConstKernel_expA <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_expA(p)
    
    _K = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    H = [real(_H);imag(_H)]
    
    return AHO_ConstKernel_expA(p,H,K)
end

function AHO_ConstKernel_expA(M::AHO)
    @unpack t_steps = M.contour 
    
    p = zeros(2t_steps,t_steps)

    return AHO_ConstKernel_expA(p)
end


function updateKernel!(pK::AHO_ConstKernel_expA)

    _K = exp(pK.p[1:div(end,2),:] + im*pK.p[div(end,2)+1:end,:])
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    pK.sqrtK = [real(_H);imag(_H)]
end

function K(u,p,::AHO_ConstKernel_expA)
    _K = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    return real(_K),imag(_K)
end

function sqrtK(u,p,::AHO_ConstKernel_expA)
    _K = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    _H = sqrt(_K)
    return [real(_H);imag(_H)]    
end

"""
Constant kernel with the entries of the real matrix P such that the kernel becomes
```math
H = e^{A}
```
We get ``K`` by the built in squaring ``H^2``.
"""
mutable struct AHO_ConstKernel_HexpA <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_HexpA(p)
    
    _H = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    H = [real(_H);imag(_H)]
    
    _K=_H^2
    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    return AHO_ConstKernel_HexpA(p,H,K)
end

function AHO_ConstKernel_HexpA(M::AHO)
    @unpack t_steps = M.contour 
    
    p = zeros(2t_steps,t_steps)

    return AHO_ConstKernel_HexpA(p)
end


function updateKernel!(pK::AHO_ConstKernel_HexpA)

    _H = exp(pK.p[1:div(end,2),:] + im*pK.p[div(end,2)+1:end,:])
    pK.sqrtK = [real(_H);imag(_H)]
    
    _K=_H^2
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])
end

function K(u,p,::AHO_ConstKernel_HexpA)
    _H = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    
    _K=_H^2
    return real(_K),imag(_K)
end

function sqrtK(u,p,::AHO_ConstKernel_HexpA)
    _H = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    
    return [real(_H);imag(_H)]    
end


"""
Constant kernel with the entries of the real matrix P such that the kernel becomes
```math
K = √(1/M) + e^{iP}
```
We get the kernel by the built in squaring ``K = H^2``.
"""
mutable struct AHO_ConstKernel_invM_H <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtinvM::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_invM_H(p,sqrtinvM)

    _H = sqrtinvM[1:div(end,2),:] + im*sqrtinvM[div(end,2)+1:end,:] + exp(im*parent(Symmetric(p)))
    H = [real(_H);imag(_H)]
    
    _K = _H^2
    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])
    
    return AHO_ConstKernel_invM_H(p,sqrtinvM,H,K)
end

function AHO_ConstKernel_invM_H(M::AHO; m=M.m, g=1)
    @unpack a,t_steps,κ = M.contour 
    A = zeros(Complex{typeof(m)},t_steps,t_steps)
    for j in 1:t_steps
        jm1 = mod1(j-1,t_steps)
        jp1 = mod1(j+1,t_steps)
        A[j,j] = g*((1/a[jm1]) + (1/a[j])) - 0.5*(a[jm1] + a[j])*m
        A[j,jp1] = -g*(1/a[j])
        A[j,jm1] = -g*(1/a[jm1])
    end
    A = -im*A
    sqrtK = sqrt(inv(A))
    p = zeros(Float64,t_steps,t_steps)
    
    return AHO_ConstKernel_invM_H(p,[real(sqrtK) ; imag(sqrtK)])
end

function updateKernel!(pK::AHO_ConstKernel_invM_H)
    _H = pK.sqrtinvM[1:div(end,2),:] + im*pK.sqrtinvM[div(end,2)+1:end,:] + exp(im*parent(Symmetric(pK.p)))
    pK.sqrtK = [real(_H);imag(_H)]
    
    _K = _H^2
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])
end

function K(u,p,pK::AHO_ConstKernel_invM_H)
    _ϵ = sparse([1,size(p)[1]], [size(p)[2],size(p)[2]], [1e-10,0.])
    _H = pK.sqrtinvM[1:div(end,2),:] + im*pK.sqrtinvM[div(end,2)+1:end,:] + exp(im*Symmetric(p) .+ _ϵ)
    _K = _H^2
    return real(_K),imag(_K)
end


function sqrtK(u,p,pK::AHO_ConstKernel_invM_H)
    _H = pK.sqrtinvM[1:div(end,2),:] + im*pK.sqrtinvM[div(end,2)+1:end,:] + exp(im*parent(Symmetric(p)))
    _H = exp(im*p)
    return [real(_H);imag(_H)]
end



"""
Constant kernel with the entries of the Hermitian matrix P such that the kernel becomes
```math
K = e^{iP}
```
This will in practise mean that we are exponentiating an anti-HErmitian matatrix ``iP``.
We get ``H`` by the built in square root ``sqrt(K)``.
"""
mutable struct AHO_ConstKernel_expiHerm <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_expiHerm(p)
    _K = exp(im*(Symmetric(p) + Hermitian(im*p,:L)))
    
    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    H = [real(_H);imag(_H)]
    
    return AHO_ConstKernel_expiHerm(p,H,K)
end

function AHO_ConstKernel_expiHerm(M::AHO)
    @unpack t_steps = M.contour 
    
    p = zeros(t_steps,t_steps)

    return AHO_ConstKernel_expiHerm(p)
end

function updateKernel!(pK::AHO_ConstKernel_expiHerm)
    _K = exp(im*(Symmetric(pK.p) + Hermitian(im*pK.p,:L)))    
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    pK.sqrtK = [real(_H);imag(_H)]
end

function K(u,p,::AHO_ConstKernel_expiHerm)
    _K = exp(im*(Symmetric(p) + Hermitian(im*p,:L)))   
    return real(_K),imag(_K)
end


function sqrtK(u,p,::AHO_ConstKernel_expiHerm)
    _K = exp(im*(Symmetric(p) + Hermitian(im*p,:L))) 
    _H = sqrt(_K)
    return [real(_H);imag(_H)]    
end


"""
Constant kernel with the entries of the symmetric matrix P such that the kernel becomes
```math
K = e^{iP}
```
We get ``H`` by the built in square root ``sqrt(K)``.
"""
mutable struct AHO_ConstKernel_expiSym <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_expiSym(p)
    _K = exp(im*parent(Symmetric(p)))
    
    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    H = [real(_H);imag(_H)]
    
    return AHO_ConstKernel_expiSym(p,H,K)
end

function AHO_ConstKernel_expiSym(M::AHO)
    @unpack t_steps = M.contour 
    
    p = zeros(t_steps,t_steps)

    return AHO_ConstKernel_expiSym(p)
end

function updateKernel!(pK::AHO_ConstKernel_expiSym)
    _K = exp(im*parent(Symmetric(pK.p)))    
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    pK.sqrtK = [real(_H);imag(_H)]
end

function K(u,p,pK::AHO_ConstKernel_expiSym)
    _ϵ = sparse([1,size(p)[1]], [size(p)[2],size(p)[2]], [1e-10,0.])
    _p = Symmetric(p) .+ _ϵ
    _K = exp(im*_p) 
    return real(_K),imag(_K)
end


function sqrtK(u,p,::AHO_ConstKernel_expiSym)
    _ϵ = sparse([1,size(p)[1]], [size(p)[2],size(p)[2]], [1e-10,0.])
    _p = Symmetric(p) .+ _ϵ
    _K = exp(im*_p)
    _H = sqrt(_K)
    return [real(_H);imag(_H)]    
end



##### LM_AHO Constant kernel
abstract type LM_AHOConstantKernelParameters <: ConstantKernelParameters end

function K(pK::T) where {T <: LM_AHOConstantKernelParameters}
    p = getKernelParams(pK)
    return K([],p,pK)
end

function dK(u,p,::T) where {T <: LM_AHOConstantKernelParameters}
    return nothing
end

function updateKernel!(pK::T) where {T <: LM_AHOConstantKernelParameters}
    nothing
end

function getKernelParamsSim(pK::T) where {T <: LM_AHOConstantKernelParameters}
    return getKernelParams(pK)
end


function K_dK(u,p,pK::T) where {T <: LM_AHOConstantKernelParameters}
    return K(u,p,pK),dK(u,p,pK)
end


"""
Kernel defined by a complex number ``H``, and normalized to length ``1``
To get the kernel we square ``K=H^2``
"""
struct LM_AHO_ConstKernel_H <: LM_AHOConstantKernelParameters
    model::LM_AHO
    H
    function LM_AHO_ConstKernel(model::LM_AHO; H=nothing)
        if isnothing(H)
            H = [pi,0.]
        end
        new(model,H)
    end
end


function K(u,p,::LM_AHO_ConstKernel_H)
    _p = p ./ sqrt(p[1]^2 + p[2]^2)
    return _p[1]^2 - _p[2]^2, 2*_p[1]*_p[2]
end


function sqrtK(u,p,::LM_AHO_ConstKernel_H)
    return p ./ sqrt(p[1]^2 + p[2]^2)
end

function getKernelParams(pK::T) where {T <: LM_AHOConstantKernelParameters}
    return pK.H
end




"""
Kernel defined by a real number ``θ``, such that the kernel is a phase rotation
```math
K = cos(θ) + isin(θ)
```
To get H we use the frommula for the square root
```math
H = cos(θ/2) + isin(θ/2)
```
"""
struct LM_AHO_ConstKernel_θ <: LM_AHOConstantKernelParameters
    model::LM_AHO
    θ
    function LM_AHO_ConstKernel_θ(model::LM_AHO; θs=[0.])
        new(model,θs)
    end
end

function getKernelParams(pK::LM_AHO_ConstKernel_θ)
    return pK.θ
end


function K(u,p,::LM_AHO_ConstKernel_θ)
    return [cos(p[1]),sin(p[1])]
end


function sqrtK(u,p,pK::LM_AHO_ConstKernel_θ)
    return [cos(p[1]*0.5),sin(p[1]*0.5)]
end



"""
Structure to store the kernel functions
"""
struct ConstantKernel{pType<:KernelParameters, KType <: Function, dKType <: Function,K_dKType <: Function, sqrtKType <: Function} <: Kernel
    pK::pType
    K::KType
    dK::dKType
    K_dK::K_dKType
    sqrtK::sqrtKType

    function ConstantKernel(pK::T) where {T<:ConstantKernelParameters}        
        _K(u,p) = K(u,p,pK) 
        _dK(u,p) = dK(u,p,pK) 
        _K_dK(u,p) = K_dK(u,p,pK)
        _sqrtK(u,p) = sqrtK(u,p,pK)
        new{T,typeof(_K),typeof(_dK),typeof(_K_dK),typeof(_sqrtK)}(pK,_K,_dK,_K_dK,_sqrtK) 
    end
end

function ConstantKernel(M::AHO; kernelType=:expiP)
    if kernelType == :K
        pK = AHO_ConstKernel_K(M)  
    elseif kernelType == :H
        pK = AHO_ConstKernel_expH(M)
    elseif kernelType == :sincos
        pK = AHO_ConstKernel_sincos(M)    
    elseif kernelType == :expiP
        pK = AHO_ConstKernel_expiP(M)
    elseif kernelType == :expA
        pK = AHO_ConstKernel_expA(M) 
    elseif kernelType == :HexpA
        pK = AHO_ConstKernel_HexpA(M)   
    elseif kernelType == :expiHerm
        pK = AHO_ConstKernel_expiHerm(M)   
    elseif kernelType == :expiSym
        pK = AHO_ConstKernel_expiSym(M)   
    elseif kernelType == :inM_expiP
        pK = AHO_ConstKernel_invM_expiP(M)  
    end

    return ConstantKernel(pK)
end



function ConstantFreeKernel(M::AHO; m=M.m, g=1, kernelType=:K)
    @unpack a,t_steps,κ = M.contour 
    A = zeros(Complex{typeof(m)},t_steps,t_steps)
    for j in 1:t_steps
        jm1 = mod1(j-1,t_steps)
        jp1 = mod1(j+1,t_steps)
        A[j,j] = g*((1/a[jm1]) + (1/a[j])) - 0.5*(a[jm1] + a[j])*m
        A[j,jp1] = -g*(1/a[j])
        A[j,jm1] = -g*(1/a[jm1])
    end
    A = -im*A
    K = inv(A)

    if kernelType == :K
        return ConstantKernel(AHO_ConstKernel_K([real(K) ; imag(K)]))
    elseif kernelType == :expA
        AiP = log(K)
        return ConstantKernel(AHO_ConstKernel_expA([real(AiP) ; imag(AiP)]))
    end

end






function ConstantKernel(M::LM_AHO)
    pK = LM_AHO_ConstKernel_θ(M)
    return ConstantKernel(pK)
end

function getKernelParams(kernel::ConstantKernel{pKType}) where {pKType <: ConstantKernelParameters}
    return getKernelParams(kernel.pK)
end

function getKernelParamsSim(kernel::ConstantKernel{pKType}) where {pKType <: ConstantKernelParameters}
    return getKernelParamsSim(kernel.pK)
end

function setKernel!(kernel::ConstantKernel{pKType},v) where {pKType <: ConstantKernelParameters}
    kernel.pK.p .= reshape(v,size(kernel.pK.p))
end

function setKernel!(kernel::ConstantKernel{LM_AHO_ConstKernel_θ},v)
    kernel.pK.θ .= v
end


###############

abstract type AHO_FieldKernelParameters <: KernelParameters end
abstract type LM_FieldKernelParameters <: KernelParameters end

struct LM_AHO_FieldKernel <: LM_FieldKernelParameters
    model::LM_AHO
    θ::Float64
    function LM_AHO_FieldKernel(model::LM_AHO,θ::Float64)
        new(model,θ)
    end
end

function getKernelParams(pK::LM_AHO_FieldKernel)
    @unpack θ = pK
    return [θ]
end

function getKernelParamsSim(pK::LM_AHO_FieldKernel)
    @unpack θ = pK
    return [θ]
end

function updateKernel!(pK::LM_AHO_FieldKernel)
    nothing
end

function f(u,pK::LM_AHO_FieldKernel)
    @unpack σ,λ = pK.model
    
    _σ = σ[1] + im*σ[2]
    _λ = λ[1] + im*λ[2]
    return exp.((u[1]^2 - u[2]^2) .* ((_λ)/_σ)) .* (cos(2*u[1]*u[2]*((_λ)/_σ)) + im*sin(2*u[1]*u[2]*((_λ)/_σ)))
end

function K(u,p,pK::LM_AHO_FieldKernel; _f = f(u,pK))
    @unpack σ,λ = pK.model
    θ, = p
    
    _σ = σ[1] + im*σ[2]
    _λ = λ[1] + im*λ[2]
    _K = (1/abs(_σ))*_f*(cos(θ) - im*sin(θ)) + (1/abs(_λ))*(1 .- _f)
    return real(_K),imag(_K)
end

function dK(u,p,pK::LM_AHO_FieldKernel;_f = f(u,pK))
    @unpack σ,λ = pK.model
    θ, = p

    _σ = σ[1] + im*σ[2]
    _λ = λ[1] + im*λ[2]
    _dK = 2*(u[1] + im*u[2])*((_λ)/_σ)*((1/abs.(_σ))*(cos(θ) - im*sin(θ)) - (1/abs(_λ)))*_f
    return real(_dK),imag(_dK)
end

function K_dK(u,p,pK::LM_AHO_FieldKernel)
    _f = f(u,pK)
    return K(u,p,pK;_f=_f),dK(u,p,pK;_f=_f)
end

function sqrtK(u,p,pK::LM_AHO_FieldKernel)
    _KRe, _KIm = K(u,p,pK) 
    sqrtK = sqrt(_KRe + im*_KIm)
    return real(sqrtK),imag(sqrtK)
end






abstract type FieldDependentKernel  <: Kernel end

struct FunctionalKernel{pType<:KernelParameters, KType <: Function, dKType <: Function,K_dKType <: Function, sqrtKType <: Function} <: FieldDependentKernel
    pK::pType
    K::KType
    dK::dKType
    K_dK::K_dKType
    sqrtK::sqrtKType

    function FunctionalKernel(pK::T) where {T<:KernelParameters}        
        _K(u,p) = K(u,p,pK) 
        _dK(u,p) = dK(u,p,pK) 
        _K_dK(u,p) = K_dK(u,p,pK)
        _sqrtK(u,p) = sqrtK(u,p,pK)
        new{T,typeof(_K),typeof(_dK),typeof(_K_dK),typeof(_sqrtK)}(pK,_K,_dK,_K_dK,_sqrtK) 
    end
end


function FieldDependentKernel(p::LM_AHO; θ=pi/2)
    pK = LM_AHO_FieldKernel(p,θ)
    return FunctionalKernel(pK)
end

function FieldDependentKernel(M::AHO)
    @unpack t_steps = M.contour 
    p = zeros(2*t_steps,t_steps)
    pK = AHO_FieldKernel(M,p)
    return FunctionalKernel(pK)
end


function getKernelParams(kernel::FieldDependentKernel)
    return getKernelParams(kernel.pK)
end

function getKernelParamsSim(kernel::FieldDependentKernel)
    return getKernelParamsSim(kernel.pK)
end

