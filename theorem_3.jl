using SumOfSquares
using DynamicPolynomials
using LinearAlgebra
using MosekTools
using IntervalArithmetic
using IntervalMatrices

"""
This extends the SumOfSquares.jl package
for a priori block-diagonal Gram matrices
"""
struct SignedBasis{
    CT<:SumOfSquares.SOSLikeCone,
    BT<:FixedPolynomialBasis,
} <: Certificate.SimpleIdealCertificate{CT,BT}
    cone::CT
    basis::Vector{BT}
end
function Certificate.gram_basis(certificate::SignedBasis, poly)
    return certificate.basis
end
function Certificate.gram_basis_type(::Type{SignedBasis{CT,BT}}) where {CT,BT}
    return Vector{BT}
end

"""
Returns hard-coded bases for the Gram decomposition
found through trial and error
"""
function make_gram_bases(x, r, M, degreex, degreer)
    monoms1even = [1; x[3]]
    monoms1odd = [x[1]; x[2]]

    monoms2even = [monoms1even; x[1]*x[1]; x[1]*x[2]; x[2]*x[2]; x[3]*x[3]]
    monoms2odd = [monoms1odd; x[1]*x[3]; x[2]*x[3]]

    monoms3even = [monoms2even; x[1]*x[1]*x[3]; x[1]*x[2]*x[3]; x[2]*x[2]*x[3]; x[3]*x[3]*x[3]]
    monoms3odd = [monoms2odd; x[1]*x[1]*x[1]; x[1]*x[1]*x[2]; x[1]*x[2]*x[2]; x[2]*x[2]*x[2]; x[1]*x[3]*x[3]; x[2]*x[3]*x[3]]

    maxr = (degreer+1)÷2

    if degreex == 4
        b_e = [(x[2] - x[1])*kron(monoms1odd, monomials(r,0:maxr)); 
                     (r-x[3])*kron(monoms1even[2:end], monomials(r,0:maxr-1)); 
                     (3*M*r*x[2]*x[2]-8*x[3]*x[3])*monomials(r,0:maxr-1)]
        b_o = [(x[2] - x[1])*kron(monoms1even, monomials(r,0:maxr))[1:end-1]; 
                    (r-x[3])*kron(monoms1odd, monomials(r,0:maxr-1))]    

        a_e = [(x[2] - x[1])*monoms1odd; (3*M*x[1]*x[2]-8*x[3])]
        a_o = (x[2] - x[1])*monoms1even[1:end-1]   
    elseif degreex == 6
        b_e = [(x[2] - x[1])*kron(monoms2odd, monomials(r,0:maxr)); 
                     (r-x[3])*kron(monoms2even[2:end], monomials(r,0:maxr-1)); 
                     (3*M*r*x[2]*x[2]-8*x[3]*x[3])*kron(monoms1even, monomials(r,0:maxr-1))]
        b_o = [(x[2] - x[1])*kron(monoms2even, monomials(r,0:maxr))[1:end-1]; 
                    (r-x[3])*kron(monoms2odd, monomials(r,0:maxr-1)); 
                    (3*M*r*x[2]*x[2]-8*x[3]*x[3])*kron(monoms1odd, monomials(r,0:maxr-1))]    

        a_e = [(x[2] - x[1])*monoms2odd; (3*M*x[1]*x[2]-8*x[3])*monoms1even]
        a_o = [(x[2] - x[1])*monoms2even[1:end-1]; (3*M*x[1]*x[2]-8*x[3])*monoms1odd]    
    elseif degreex == 8
        b_e = [(x[2] - x[1])*kron(monoms3odd, monomials(r,0:maxr)); 
                     (r-x[3])*kron(monoms3even[2:end], monomials(r,0:maxr-1)); 
                     (3*M*r*x[2]*x[2]-8*x[3]*x[3])*kron(monoms2even, monomials(r,0:maxr-1))]
        b_o = [(x[2] - x[1])*kron(monoms3even, monomials(r,0:maxr))[1:end-1]; 
                    (r-x[3])*kron(monoms3odd, monomials(r,0:maxr-1)); 
                    (3*M*r*x[2]*x[2]-8*x[3]*x[3])*kron(monoms2odd, monomials(r,0:maxr-1))]    

        a_e = [(x[2] - x[1])*monoms3odd; (3*M*x[1]*x[2]-8*x[3])*monoms2even]
        a_o = [(x[2] - x[1])*monoms3even[1:end-1]; (3*M*x[1]*x[2]-8*x[3])*monoms2odd]    
    else
        throw("only degree 4, 6 or 8 currently supported")
    end


    return (b_e, b_o, a_e, a_o)
end

"""
Builds and solves the SDP
"""
function lorenz_model(x, r, V_basis, s_basis, rhomin, rhomax, M, rhs_bases)       
    rmaxM = (rhomax-1)
    rminM = (rhomin-1)

    # The Lorenz system rescaled:
    f = [30x[2] - 30x[1]; 
         3M*r*x[1] - 3M*x[3]*x[1] - 3x[2] + 3x[1]; 
         3M*x[1]*x[2] - 8x[3]];
      
    model = SOSModel(MosekTools.Optimizer)
    

    V_coeffs = @variable(model, [1:length(V_basis)])
    
    V = dot(V_coeffs,V_basis)
    dVdx = differentiate(V,x)

    s_coeffs = @variable(model, [1:length(s_basis)])
    s = dot(s_coeffs, s_basis)
    
    P = dot(f, dVdx) - (x[2] - x[1])^2

    @assert(length(rhs_bases) == 4)
    b_e = FixedPolynomialBasis(rhs_bases[1])
    b_o = FixedPolynomialBasis(rhs_bases[2])
    a_e = FixedPolynomialBasis(rhs_bases[3])
    a_o = FixedPolynomialBasis(rhs_bases[4])

    cert1 = SignedBasis(SOSCone(), [a_e, a_o])
    cert2 = SignedBasis(SOSCone(), [b_e, b_o])

    range = (M*r-rminM)*(rmaxM-M*r)

    aQa = @constraint(model, s >= 0, sparsity = Sparsity.SignSymmetry(), ideal_certificate = cert1)
    bRb = @constraint(model, P >= s*range, sparsity = Sparsity.SignSymmetry(), ideal_certificate = cert2)

    optimize!(model)   

    return model, P, V_coeffs, aQa, bRb, range
end


function assemble_linear_system(x, r, P, range, V_coeffs, constraints, rhs_bases)
    monomial_basis = monomials(P)

    num_RHSunknowns = sum(sum(length(block.Q.Q) for block in gram_matrix(constraint).blocks) for constraint in constraints)
    @polyvar RHSunknowns[1:num_RHSunknowns]
    RHScoeffs = zeros(typeof(P), num_RHSunknowns)

    num_LHSunknowns = length(V_coeffs)

    num_unknowns = num_LHSunknowns + num_RHSunknowns
    y0 = zeros(num_LHSunknowns + num_RHSunknowns) # This is the initial guess for the vector of unknowns
    y0[1:num_LHSunknowns] = value.(V_coeffs)

    connum = 0
    blocknum = 0
    varnum = 1
    for constraint in constraints
        connum += 1

        for block in gram_matrix(constraint).blocks
            blocknum += 1
            basis = rhs_bases[blocknum]

            # The unknowns for each block are the elements of the symmetric matrix
            numvars = length(block.Q.Q)
            y0[num_LHSunknowns+varnum : num_LHSunknowns+varnum+numvars-1] = block.Q.Q

            # This creates a vector with the coefficient for each unknown (which is a polynomial)
            terms = kron(transpose(basis), basis)
            RHScoeffs[varnum:varnum+numvars-1] = [(1+abs(sign(i-j)))*terms[i,j] for j in 1:length(basis) for i in 1:j]

            if connum == 2
                RHScoeffs[varnum:varnum+numvars-1] *= range
            end

            expanded = dot(basis,block.Q*basis) # also expand with actual values
            # This ensures we capture all monomials, even if they are not on LHS
            monomial_basis = merge_monomial_vectors([monomial_basis, monomials(expanded)]) 

            varnum += numvars
        end
    end

    # replace basis elements with x[1] at non-trivial fixed points
    basis_reduced = subs.(monomial_basis, x[2]=>x[1], x[3]=>x[1]*x[1], r=>x[1]*x[1])
    # Find the last element of each power of x[1]
    todelete = unique(i->basis_reduced[i], length(basis_reduced):-1:1)
    # These equations are implied by the others so we don't need to solve them
    deleteat!(monomial_basis, reverse(todelete))
    
    num_equations = length(monomial_basis)

    # Allocate the matrices
    A = zeros(Float64, num_equations, num_unknowns)
    c = zeros(Float64, num_equations)

    # Now for each monomial in the full equation, i.e. each row of the linear system
    Threads.@threads for i = 1:num_equations
        monomial = monomial_basis[i]
       
        # Find the term on the LHS of the full equation for this monomial
        LHSterm = DynamicPolynomials.coefficient(P, monomial)

        # The constant becomes the RHS of the linear system
        c[i] = -LHSterm.constant # move from LHS to RHS so include a minus sign

        # And each unknown may have a component in the linear system
        for n in eachindex(V_coeffs)
            if V_coeffs[n] ∈ LHSterm.terms.keys
                A[i,n] = LHSterm.terms[V_coeffs[n]]
            end
        end

        # Also the terms on the RHS of the full equation, the Gram matrices, contribute
        for n1=1:num_RHSunknowns
            # These terms were originally on the RHS so include a minus sign in the matrix A
            A[i,num_LHSunknowns+n1] = -DynamicPolynomials.coefficient(RHScoeffs[n1], monomial).constant
        end
    end

    return A, c, y0
end

"""
Builds gram matrices from the end of the vector of unknowns
"""
function reassemble_gram_matrices(y, ab)
    result = []

    # work backwards from the end
    endindex = length(y)
    for blocknum = length(ab):-1:1
        basissize = length(ab[blocknum])
        blocksize = basissize*(basissize+1)÷2

        result = [result; [SymMatrix(y[endindex-blocksize+1:endindex], basissize)]]

        endindex -= blocksize
    end
 
    return result
end


"""
Returns true if an interval symmetric matrix is verifiably positive semidefinite
"""
function is_positive_semidefinite(A::SymMatrix{Interval{T}}) where T
    mat = Matrix(A)

    # Check all the principal minors have positive determinant
    success = true
    for n=1:size(mat,1)
        if det(mat[1:n,1:n]).lo < 0
            success = false
            break
        end
    end

    return success
end

"""
Returns an interval matrix which contains the Moore-Penrose right inverse of a matrix A,
if it exists
"""
function rightinverse(A::IntervalMatrix{T}) where T
    return transpose((A*transpose(A))\A)
end

"""
Guassian elimination for interval matrices

Following Jaulin, Kieffer, Didrit & Walter (2001) Table 4.3
but for matrix instead of vector RHS
"""
function \(A::IntervalMatrix{T}, B::IntervalMatrix{T}) where T
    @assert(size(A,1)==size(A,2)==size(B,1))
    N = size(A,1)
    M = size(B,2)

    A = copy(A)
    B = copy(B)

    X = zero(B)

    for i = 1:N-1
        if 0 ∈ A[i,i]
            throw(LinearAlgebra.SingularException(i))
        end

        for j = i+1:N
            αj = A[j,i]/A[i,i]
            B[j,:] -= αj*B[i,:]

            for k = i+1:N
                A[j,k] -= αj*A[i,k]
            end
        end
    end

    for i = N:-1:1
        for j = 1:M
            X[i,j] = (B[i,j] - sum(A[i,i+1:end].*X[i+1:end,j])) / A[i,i]
        end
    end

    return X
end

"""
Generates a basis of monomials for the polynomials V and s
"""
function make_polynomial_basis(x, degx, mindeg, scale)
    # The basis should be invariant under change of sign of x1 and x2 but not x3
    symmetries = [1 1 0] 

    basisraw = monomials(x,mindeg:degx)
    basis = Vector{eltype(basisraw)}()

    for ind in eachindex(basisraw)
        powers = exponents(basisraw[ind])
        if !all(rem.(symmetries*powers,2).==0)
            continue
        end

        basis = [basis; scale^sum(powers) * basisraw[ind]]
    end

    return basis
end

"""
Call this to run the full validation for Theorem 3
    e.g. run(0,4,6,1)
"""
function run(rhomin, rhomax, degreex, degreer)
    @assert(degreex%2 == 0)

    setprecision(Interval, 256) # 128 is insufficient for higher rho
    setprecision(BigFloat, 256)

    @polyvar x[1:3]
    @polyvar r

    # rescale factor, for nicer numerics
    M = 8

    V_basis = kron(make_polynomial_basis(x, degreex, 1, 1), monomials(r,0:degreer))
    s_basis = kron(make_polynomial_basis(x, degreex, 1, 1), monomials(r,0:degreer-1))

    ab = make_gram_bases(x, r, M, degreex, degreer) # Bases for the Gram matrices

    # First run the optimisation once
    _, _, V_coeffs, _, _, _ = lorenz_model(x, r, V_basis, s_basis, rhomin, rhomax, M, ab)

    # Remove elements with zero coefficient from V_basis
    # Otherwise validation can fail
    basis_V_reduced = Vector{eltype(V_basis)}()
    for i in eachindex(V_coeffs)
        if abs(value(V_coeffs[i])) > 1e-20
            push!(basis_V_reduced, V_basis[i])
        end
    end

    # Run again, get the coefficients
    model, P, V_coeffs, aQa, bRb, range = lorenz_model(x, r, basis_V_reduced, s_basis, rhomin, rhomax, M, ab)

    display(solution_summary(model))

    println("Constructing linear system...")
    A,c,y0 = assemble_linear_system(x, r, P, range, V_coeffs, [bRb; aQa], ab)

    A = IntervalMatrix(interval.(BigFloat.(A)))
    y0 = interval.(BigFloat.(y0))
    c = interval.(BigFloat.(c))

    println("Finding right inverse...")
    A⁺ = rightinverse(A)

    println("Enclosing solution...")
    # Find a soluion of Ay=c near y0
    # y = A⁺*c + (I-A⁺*A)*y0 
    # This should be faster:
    y = A⁺*c + y0 - A⁺*(A*y0)

    println("Assembling Gram matrices...")
    gram_matrices = reassemble_gram_matrices(y, ab)

    println("Verifying Gram matrices...")
    success = true
    for Q in gram_matrices
        if !is_positive_semidefinite(Q)
            success = false
        end
    end

    if success
        println("All blocks verified positive definite")
    else
        println("Failed to validate")
    end

    return gram_matrices
end
