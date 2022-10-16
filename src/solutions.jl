export y_LM_AHO_x2, y_LM_AHO_x, y_U1_expix, y_U1_expi2x

using QuadGK
using SpecialFunctions

include("scroSol.jl")


# The solution for 1/2x^2 + 1/4x^4
function y_x3(p::LM_AHO)
    @unpack σ, λ = p

    z(x) = exp(-(0.5*(σ[1] + im*σ[2])*x^2 + (1/24)*λ*x^4))
    f(x) = x^3 * z(x)
    Z = quadgk(z, -Inf, Inf, rtol=1e-5)[1]
    avg3 = quadgk(f, -Inf, Inf, rtol=1e-5)[1] / Z
    return [real(avg3),imag(avg3)]
end 

function y_x2(p::LM_AHO)
    @unpack σ, λ = p

    z(x) = exp(-(0.5*(σ[1] + im*σ[2])*x^2 + (1/24)*λ*x^4))
    f(x) = x^2 * z(x)
    Z = quadgk(z, -Inf, Inf, rtol=1e-5)[1]
    avg2 = quadgk(f, -Inf, Inf, rtol=1e-5)[1] / Z
    return [real(avg2),imag(avg2)]
end 

function y_x(p::LM_AHO)

    @unpack σ, λ = p

    z(x) = exp(-(0.5*(σ[1] + im*σ[2])*x^2 + (1/24)*λ*x^4))
    f(x) = x * z(x)
    Z = quadgk(z, -Inf, Inf, rtol=1e-5)[1]
    avg = quadgk(f, -Inf, Inf, rtol=1e-5)[1] / Z
    return [real(avg),imag(avg)]

end 


function getSolutions(p::LM_AHO)
    yx = y_x(p)
    yx2 = y_x2(p)
    yx3 = y_x3(p)

    return Dict("x" => yx, "x2" => yx2, "x3" => yx3)

end


function y_U1_expix(p::U1)
    @unpack β = p
    return - im * besselj1(β)/besselj0(β)
end 

function y_U1_expi2x(p::U1)
    @unpack β = p
    return - besselj(2,β)/besselj0(β)
end 

function getSolutions(p::U1)
    exp1x = y_U1_expix(p)
    exp2x = y_U1_expi2x(p)

    return Dict("exp1Re" => real(exp1x), "exp1Im" => imag(exp1x), "exp2Re" => real(exp2x), "exp2Im" => imag(exp2x))

end