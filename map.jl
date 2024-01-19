using SumOfSquares
using MosekTools
using LinearAlgebra
using DynamicPolynomials
using CairoMakie

CairoMakie.activate!(type = "png")

function henonf(x, a, b)
    [1-a*x[1]^2+x[2], b*x[1]]
end

function iterate(x0, a, b, d)
    x = copy(x0)
    for _ = 1:d
        x = henonf(x,a,b)
    end

    return x
end

function plotbifurcation(ax, b)
    a = 0:0.00001:1.5
    x = zeros(length(a),2)
    for i in eachindex(a)
        x[i,:] = iterate([1 0],a[i],b,rand(1000:2000))
    end

    scatter!(ax,a,x[:,1], alpha=0.2, markersize=1,color=:black)
end

"""
Attempts to find a Lyapunov function demonstrating periodic points of period greater than k
    do not exist in the Hénon map, using a Lyapunov function of degree d.
    Returns true if able to non-rigorously numerically do this, otherwise false.
"""
function henon(a, b, d, k = 1)
    @polyvar x[1:2]

    f = henonf(x,a,b)

    fk = iterate(x,a,b,k)

    g = (fk[1]-x[1])^2 + (fk[2]-x[2])^2

    model = SOSModel(Mosek.Optimizer)

    V_basis = monomials(x,1:d)
    Vc = @variable(model, [1:length(V_basis)])
    V = dot(Vc,V_basis)

    Vf = subs(V, x=>f)

    @constraint(model, Vf - V >= g)

    optimize!(model)

    return solution_summary(model).dual_objective_value
end

function testall(b, d, k)
    as = 0:0.015:1.5
    dobj = Vector{Float32}()
    for a in as
        dobj = [dobj; henon(a,b,d,k)]
    end
    return as,dobj
end

function makefigure(b)
    with_theme(theme_latexfonts()) do
        fig = Figure(size=(700,400))

        ax1 = Axis(fig[1, 1], limits=((0, 1.4), (-1.5,1.5)), xlabel=L"a", ylabel=L"x_1", ylabelsize=18, xlabelsize=18)
        ax2 = Axis(fig[1, 1], limits=((0, 1.4), nothing), yaxisposition = :right, yscale=log10, ylabelsize=18, ylabel="Dual objective")
        hidespines!(ax2)
        hidedecorations!(ax1, label=false, ticks=false, ticklabels=false)
        hidedecorations!(ax2, label=false, ticks=false, ticklabels=false)
        hidexdecorations!(ax2)

        plotbifurcation(ax1,b)

        as,dobj = testall(b,6,1)
        lines!(ax2,as,dobj, label=L"d=6,\,k=1",linewidth=2)

        as,dobj = testall(b,8,1)
        lines!(ax2,as,dobj, label=L"d=8,\,k=1",linewidth=2)

        as,dobj = testall(b,6,2)
        lines!(ax2,as,dobj, label=L"d=6,\,k=2",linewidth=2)

        as,dobj = testall(b,8,2)
        lines!(ax2,as,dobj, label=L"d=8,\,k=2",linewidth=2)

        as,dobj = testall(b,6,3)
        lines!(ax2,as,dobj, label=L"d=6,\,k=3",linewidth=2)

        as,dobj = testall(b,8,3)
        lines!(ax2,as,dobj, label=L"d=8,\,k=3",linewidth=2)

        as,dobj = testall(b,10,3)
        lines!(ax2,as,dobj, label=L"d=10,\,k=3",linewidth=2)

        axislegend(position=:rb)
        return fig
    end   
end