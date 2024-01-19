using SumOfSquares
using MosekTools
using LinearAlgebra
using DynamicPolynomials
using CairoMakie

CairoMakie.activate!(type = "png")

function lorenzf(x,rho)
    M=16
    r = (rho-1)/M
    [30x[2]-30x[1], 3M*x[1]*r-3M*x[1]*x[3]-3x[2]+3x[1], 3M*x[1]*x[2]-8x[3]]
end

"""
Form the semi-definite program to show that the Lorenz system is gradient-like at rho,
with a polynomial Lypaunov function of degree d 
"""
function lorenz(rho, d)
    @polyvar x[1:3]

    f = lorenzf(x,rho)

    model = SOSModel(Mosek.Optimizer)

    V_basis = monomials(x,1:d)
    Vc = @variable(model, [1:length(V_basis)])
    V = dot(Vc,V_basis)

    dVdt = dot(f,differentiate(V,x))

    @constraint(model, dVdt >= (x[2]-x[1])^2)

    optimize!(model)

    return solution_summary(model).dual_objective_value
end

function testall(d)
    rhos = 0:0.1:14
    dobj = Vector{Float32}()
    for rho in rhos
        dobj = [dobj; lorenz(rho,d)]
    end
    return rhos,abs.(dobj)
end

function makefigure()
    with_theme(theme_latexfonts()) do
        fig = Figure(size=(700,400))

        ax2 = Axis(fig[1, 1], limits=((0, 14), nothing), ylabelsize=18, xlabelsize=18, yscale=log10, ylabel="Dual objective", xlabel=L"\rho")
        hidedecorations!(ax2, label=false, ticks=false, ticklabels=false)

        rhos,dobj = testall(4)
        lines!(ax2,rhos,dobj, label=L"d=4",linewidth=2)

        rhos,dobj = testall(6)
        lines!(ax2,rhos,dobj, label=L"d=6",linewidth=2)
        
        rhos,dobj = testall(8)
        lines!(ax2,rhos,dobj, label=L"d=8",linewidth=2)
        
        rhos,dobj = testall(10)
        lines!(ax2,rhos,dobj, label=L"d=10",linewidth=2)

        rhos,dobj = testall(12)
        lines!(ax2,rhos,dobj, label=L"d=12",linewidth=2)
        
        rhos,dobj = testall(14)
        lines!(ax2,rhos,dobj, label=L"d=14",linewidth=2)

        axislegend(position=:rb)
        return fig
    end   
end