using cgb,JuMP,Mosek,MosekTools
include("../npa_utils.jl")
include("./three_meas_corr.jl")

"""
    pl_guess_prob_four(G, level, fname; pure=true, Alice=true)

Computes Eve's guessing probability P_g^K for the 4-state QKD protocol as a function of detector efficiency η.

# Arguments
- `G`: Upper bound on Eve's state guessing probability P_g^A ≤ G
- `level`: NPA hierarchy level (higher = tighter but slower)
- `fname`: Output filename to save results
- `pure`: Whether Alice's states are pure (default: true)
- `Alice`: Whether to extract key from Alice's inputs (true) or Bob's outcomes (false)

# Returns
- `(plx, ply)`: Tuple of (η values, guessing probabilities P_g^K)

# Protocol Context
This function implements the NPA hierarchy optimization to find Eve's optimal guessing probability
for the raw key in the 4-state protocol. The optimization problem is:

maximize: P_g^K(A/B) = (1/4) Σ_{x,b} Tr(ρ_x B_{b|1} E_{key_bit})

subject to:
- σ ≥ (1/4)ρ_x ∀x (guessing probability constraint)
- Tr(σ) ≤ G (bound on P_g^A)
- Tr(ρ_x B_{b|y}) = p(b|x,y,η) (observed statistics with efficiency η)
- ρ_x ≥ 0, Tr(ρ_x) = 1 (valid quantum states)
- B_{b|y} ≥ 0, Σ_b B_{b|y} = I (valid POVM)

The function varies η from 1 to 0 and computes the maximum achievable P_g^K for each value.
"""
function pl_guess_prob_four(G,level,fname;pure=true,Alice=true)

    # Set up the NPA hierarchy with monoids for Bob's measurements (B), Eve's measurements (E),
    # and combined Alice-Bob-Eve operators (BE)
    @pcmonoid M B[3,0] E[1,0] BE[5,0] 
    @comms B E  # B and E operators commute (spacelike separation)
    Projector.([B;E])  # B and E are projection operators (POVM elements)
    if pure
        Projector.(BE[1:4])  # Alice's 4 pure states are also projectors
    end
    build(M)

    # Define POVM elements: PB[outcome, measurement] 
    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:4]  # Alice's 4 quantum states ρ₁, ρ₂, ρ₃, ρ₄
    σ = BE[5]    # Auxiliary operator σ for guessing probability constraint
    PE=[E[1];1-E[1]]  # Eve's binary measurement operators E₁, I-E₁
    
    # Constraint: σ ≥ (1/4)ρₓ for all x (ensures P_g^A ≤ Tr(σ))
    op_ge = [σ-0.25*ρ[x] for x in 1:4]
    if ! pure
        # For mixed states, add constraint ρₓ - ρₓ² ≥ 0 (ensures ρₓ is positive semidefinite)
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:4])
    end
    # Normalization constraints: Tr(ρₓ) = 1
    tr_eq = vcat([ [ ρ[x], 1] for x in 1:4])
    # Upper bound constraint: Tr(σ) ≤ G
    tr_ge = [ [-σ, -G]]
    
    # Build the NPA relaxation
    model, unique_vars, unique_mons = npa_model(M.vertices, level;
        obj=0,
        min=false,
        op_ge = op_ge,
        tr_eq = tr_eq,
        tr_ge = tr_ge,
        lvl_principal=0,
        cyclic=true,
        normalize=false)
    Γ=Dict(zip(unique_mons, unique_vars))
    # Parameter sweep: detector efficiency η from 1.0 to 0.0
    plx = [1-i/100 for i in 0:100]
    # Constraints linking NPA variables to observed probabilities p(b=1|x,y,η)
    @constraint(model, c[x=1:4,y=1:3], Γ[cyclic_reduce(ρ[x] * PB[1, y])] == prob_four(1, x, y, plx[1]))

    ply=ones(length(plx))
    for i in 1:length(plx)
        # Update probability constraints for current η value
        for x in 1:4, y in 1:3
            set_normalized_rhs(c[x,y], prob_four(1, x, y, plx[i]))
        end
        
        # Define objective functions for key extraction
        # Bob's key: P_g^K(B) = (1/2) Σ_{x∈{1,2}} [Tr(ρₓ B₁E₁) + Tr(ρₓ B₂E₂)]
        # Uses correlation: if Bob gets outcome b, Eve guesses b for the key bit
        objB = cyclic_reduce((ρ[1]*PB[1,1]*PE[1] + ρ[2]*PB[2,1]*PE[2] + ρ[1]*PB[2,1]*PE[2] + ρ[2]*PB[1,1]*PE[1])/2)
        
        # Alice's key: P_g^K(A) = (1/2) Σ_{x∈{1,2}} [Tr(ρₓ B₁E₁) + Tr(ρₓ B₂E₁)] + similar for x∈{2,1}
        # Uses correlation: Eve guesses Alice's state preparation (x-1 mod 2) as key bit
        objA = cyclic_reduce((ρ[1]*PB[1,1]*PE[1] + ρ[1]*PB[2,1]*PE[1] + ρ[2]*PB[1,1]*PE[2] + ρ[2]*PB[2,1]*PE[2])/2)
        
        obj= Alice ? objA : objB
        @objective(model, Max, sum(c*Γ[m] for (m,c) in obj))
        optimize!(model)
        
        ply[i] = objective_value(model)
        println(ply[i])
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        if ply[i]>0.9999
            break  # Stop if guessing probability approaches 1 (no security)
        end
        for i in 1:3
            GC.gc()  # Garbage collection to manage memory
        end

    end
    return plx,ply
end



"""
    pl_guess_prob_six(G, level, fname; Alice=true, pure=true, adapt=false)

Computes Eve's guessing probability P_g^K for the 6-state QKD protocol as a function of detector efficiency η.

# Arguments
- `G`: Upper bound on Eve's state guessing probability P_g^A ≤ G
- `level`: NPA hierarchy level
- `fname`: Output filename to save results
- `Alice`: Whether to extract key from Alice's inputs (true) or Bob's outcomes (false)
- `pure`: Whether Alice's states are pure (default: true)
- `adapt`: Whether Eve can adapt her measurement to each round (default: false)

# Returns
- `(plx, ply)`: Tuple of (η values, guessing probabilities P_g^K)

# Protocol Context
Implements the enhanced 6-state protocol where:
- Alice prepares 6 states: |ψ₀±⟩, |ψ_{π/4}±⟩, |ψ_{π/8}±⟩
- Bob measures 3 observables: Z, X, (Z+X)/√2
- Key generation uses compatible state-measurement pairs

The guessing probability is:
P_g^K = (1/6) Σ_{compatible pairs} Tr(ρₓ B_{b|y} E_{key_bit})

where compatible pairs are:
- States 1,4 (Z-basis) with measurement 1 (Z)
- States 2,5 (X-basis) with measurement 2 (X)  
- States 3,6 ((Z+X)/√2-basis) with measurement 3 ((Z+X)/√2)
"""
function pl_guess_prob_six(G,level,fname;Alice=true,pure=true,adapt=false,start_eta=1)
    if !adapt
        # Non-adaptive Eve: single measurement strategy
        @pcmonoid M B[3,0] E[1,0] BE[7,0]
    else
        # Adaptive Eve: can choose measurement based on protocol round
        @pcmonoid M B[3,0] E[3,0] BE[7,0]
    end
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:6])  # Alice's 6 pure states
    end
    build(M)

    ρ = BE[1:6]  # Alice's 6 quantum states
    σ = BE[7]    # Auxiliary operator for guessing constraint
    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]  # Bob's 3 measurements
    if !adapt
        PE=[E[1];1-E[1]]  # Eve's single measurement
    else
        PE=[E[1] E[2] E[3]; 1-E[1] 1-E[2] 1-E[3]]  # Eve's 3 adaptive measurements
    end

    # Guessing probability constraint: σ ≥ (1/6)ρₓ ∀x
    op_ge = [σ-1/6*ρ[x] for x in 1:6]
    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:6])
    end
    tr_eq = vcat([ [ ρ[x], 1] for x in 1:6])
    tr_ge = [ [-σ, -G]]
    
    model, unique_vars, unique_mons = npa_model(M.vertices, level;
        obj=0,
        min=false,
        op_ge = op_ge,
        tr_eq = tr_eq,
        tr_ge = tr_ge,
        lvl_principal=0,
        cyclic=true,
        normalize=false)
    Γ=Dict(zip(unique_mons, unique_vars))
    plx = [start_eta-i/100 for i in 0:100]
    @constraint(model, c[x=1:6,y=1:3], Γ[cyclic_reduce(ρ[x] * PB[1, y])] == prob_six(1, x, y, plx[1]))

    ply=ones(length(plx))
    for i in 1:length(plx)
        for x in 1:6, y in 1:3
            set_normalized_rhs(c[x,y], prob_six(1, x, y, plx[i]))
        end
        # # key from 1 observable from Bob side
        # obj1B= cyclic_reduce((ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[2]+ρ[4]*PB[1,1]*PE[1]+ρ[4]*PB[2,1]*PE[2])/2)
        # # key from 1 observables from Alice side
        # obj1A=cyclic_reduce((ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[1]+ρ[4]*PB[1,1]*PE[2]+ρ[4]*PB[2,1]*PE[2])/2)
        # # key from 2 observables from Bob side
        # obj2B = cyclic_reduce((ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[2]+ρ[4]*PB[1,1]*PE[1]+ρ[4]*PB[2,1]*PE[2]+
        #              ρ[2]*PB[1,2]*PE[1]+ρ[2]*PB[2,2]*PE[2]+ρ[5]*PB[1,2]*PE[1]+ρ[5]*PB[2,2]*PE[2])/4)
        # # key from 2 observables from Alice's side
        # obj2A = cyclic_reduce((ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[1]+ρ[4]*PB[1,1]*PE[2]+ρ[4]*PB[2,1]*PE[2]+
        #              ρ[2]*PB[1,2]*PE[1]+ρ[2]*PB[2,2]*PE[1]+ρ[5]*PB[1,2]*PE[2]+ρ[5]*PB[2,2]*PE[2])/4)
        # # key from 3 observables from Bob side
        if !adapt
            # Non-adaptive Eve case
            obj3B = cyclic_reduce((ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[2]+ρ[4]*PB[1,1]*PE[1]+ρ[4]*PB[2,1]*PE[2]+
                        ρ[2]*PB[1,2]*PE[1]+ρ[2]*PB[2,2]*PE[2]+ρ[5]*PB[1,2]*PE[1]+ρ[5]*PB[2,2]*PE[2]+
                        ρ[3]*PB[1,3]*PE[1]+ρ[3]*PB[2,3]*PE[2]+ρ[6]*PB[1,3]*PE[1]+ρ[6]*PB[2,3]*PE[2])/6)
            # Alice's key extraction
            obj3A = cyclic_reduce((ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[1]+ρ[4]*PB[1,1]*PE[2]+ρ[4]*PB[2,1]*PE[2]+
                        ρ[5]*PB[1,2]*PE[1]+ρ[5]*PB[2,2]*PE[1]+ρ[2]*PB[1,2]*PE[2]+ρ[2]*PB[2,2]*PE[2]+
                        ρ[3]*PB[1,3]*PE[1]+ρ[3]*PB[2,3]*PE[1]+ρ[6]*PB[1,3]*PE[2]+ρ[6]*PB[2,3]*PE[2])/6)
        else
            # Adaptive Eve case: can tailor measurement to each state type
            obj3B = cyclic_reduce((ρ[1]*PB[1,1]*PE[1,1]+ρ[1]*PB[2,1]*PE[2,1]+ρ[4]*PB[1,1]*PE[1,1]+ρ[4]*PB[2,1]*PE[2,1]+
                        ρ[2]*PB[1,2]*PE[1,2]+ρ[2]*PB[2,2]*PE[2,2]+ρ[5]*PB[1,2]*PE[1,2]+ρ[5]*PB[2,2]*PE[2,2]+
                        ρ[3]*PB[1,3]*PE[1,3]+ρ[3]*PB[2,3]*PE[2,3]+ρ[6]*PB[1,3]*PE[1,3]+ρ[6]*PB[2,3]*PE[2,3])/6)
            # key from 3 observables from Alice's side
            obj3A = cyclic_reduce((ρ[1]*PB[1,1]*PE[1,1]+ρ[1]*PB[2,1]*PE[1,1]+ρ[4]*PB[1,1]*PE[2,1]+ρ[4]*PB[2,1]*PE[2,1]+
                        ρ[5]*PB[1,2]*PE[1,2]+ρ[5]*PB[2,2]*PE[1,2]+ρ[2]*PB[1,2]*PE[2,2]+ρ[2]*PB[2,2]*PE[2,2]+
                        ρ[3]*PB[1,3]*PE[1,3]+ρ[3]*PB[2,3]*PE[1,3]+ρ[6]*PB[1,3]*PE[2,3]+ρ[6]*PB[2,3]*PE[2,3])/6)
        end


        obj= Alice ? obj3A : obj3B
        @objective(model, Max, sum(c*Γ[m] for (m,c) in obj))
        optimize!(model)
        ply[i] = objective_value(model)
        println(ply[i])
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        for i in 1:3
            GC.gc()
        end

        if ply[i]>0.9999
            break
        end
    end
    return plx,ply
end


"""
    pl_guess_prob_four(level, fname; start_G=0.5, num_points=50, pure=true, Alice=true)

Computes Eve's guessing probability P_g^K for the 4-state protocol as a function of the bound G on P_g^A.

# Arguments
- `level`: NPA hierarchy level
- `fname`: Output filename to save results
- `start_G`: Starting value for the state guessing bound (default: 0.5)
- `num_points`: Number of points to compute (default: 50)
- `pure`: Whether Alice's states are pure (default: true)
- `Alice`: Whether to extract key from Alice's inputs (true) or Bob's outcomes (false)

# Returns
- `(plx, ply)`: Tuple of (G values, guessing probabilities P_g^K)

# Protocol Context
This version fixes the detector efficiency η=1 (perfect detectors) and varies the information
constraint G on Eve's state guessing probability P_g^A. As G increases (Eve has more information
about Alice's states), her key guessing probability P_g^K should also increase.

The trade-off shows how the bound on state distinguishability affects key security.
"""
function pl_guess_prob_four(level,fname;start_G=0.5,num_points=50,pure=true,Alice=true)

    @pcmonoid M B[3,0] E[1,0] BE[5,0] 
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:4])
    end
    build(M)

    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    ρ = BE[1:4]
    σ = BE[5]
    PE=[E[1];1-E[1]]
    op_ge = [σ-0.25*ρ[x] for x in 1:4]
    tr_eq = vcat([ [ ρ[x], 1] for x in 1:4])

    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:4])
    end

    model, unique_vars, unique_mons = npa_model(M.vertices, level;
        obj=0,
        min=false,
        op_ge = op_ge,
        tr_eq = tr_eq,
        lvl_principal=0,
        cyclic=true,
        normalize=false)

    Γ=Dict(zip(unique_mons, unique_vars))
    println("mosek starts")
    # Fixed detector efficiency η=1 (perfect detectors)
    @constraint(model, c[x=1:4,y=1:3], Γ[cyclic_reduce(ρ[x] * PB[1, y])] == prob_four(1, x, y, 1))
    # Variable constraint on state guessing: Tr(σ) ≤ G
    @constraint(model, d,Γ[cyclic_reduce(cgb.monomial(σ))] <= start_G)

    objB = cyclic_reduce((ρ[1]*PB[1,1]*PE[1] + ρ[2]*PB[2,1]*PE[2] + ρ[1]*PB[2,1]*PE[2] + ρ[2]*PB[1,1]*PE[1])/2)
    objA = cyclic_reduce((ρ[1]*PB[1,1]*PE[1] + ρ[1]*PB[2,1]*PE[1] + ρ[2]*PB[1,1]*PE[2] + ρ[2]*PB[2,1]*PE[2])/2)
    obj= Alice ? objA : objB

    plx = [start_G+i/100 for i in 1:num_points]
    ply=ones(num_points)
    for i in 1:num_points
        set_normalized_rhs(d,plx[i])

        @objective(model, Max, sum(c*Γ[m] for (m,c) in obj))
        optimize!(model)
        ply[i] = objective_value(model)
        println(ply[i])
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end

        for i in 1:3
            GC.gc()
        end

        if ply[i]>0.9999
            break
        end
    end
    return plx,ply
end



"""
    pl_guess_prob_six(level, fname; Alice=true, pure=true, adapt=false, num_points=67, start_G=1/3)

Computes Eve's guessing probability P_g^K for the 6-state protocol as a function of the bound G on P_g^A.

# Arguments
- `level`: NPA hierarchy level
- `fname`: Output filename to save results  
- `Alice`: Whether to extract key from Alice's inputs (true) or Bob's outcomes (false)
- `pure`: Whether Alice's states are pure (default: true)
- `adapt`: Whether Eve can adapt her measurement (default: false)
- `num_points`: Number of G values to compute (default: 67)
- `start_G`: Starting value for G (default: 1/3, the theoretical minimum for 6 equiprobable states)

# Returns
- `(plx, ply)`: Tuple of (G values, guessing probabilities P_g^K)

# Protocol Context
This version analyzes the 6-state protocol with perfect detectors (η=1) while varying the
information constraint. The default start_G=1/3 corresponds to the quantum bound for 
distinguishing 6 equiprobable symmetric states.

As G increases beyond 1/3, Eve gains more information about Alice's state preparation,
allowing higher key guessing probabilities and reducing the secure key rate.
"""
function pl_guess_prob_six(level,fname;Alice=true,pure=true,adapt=false,num_points=67,start_G=1/3)
    if !adapt
        @pcmonoid M B[3,0] E[1,0] BE[7,0]
    else
        @pcmonoid M B[3,0] E[3,0] BE[7,0]
    end
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:6])
    end
    build(M)

    ρ = BE[1:6]
    σ = BE[7]
    PB=[B[1] B[2] B[3];1-B[1] 1-B[2] 1-B[3]]
    if !adapt
        PE=[E[1];1-E[1]]
    else
        PE=[E[1] E[2] E[3]; 1-E[1] 1-E[2] 1-E[3]]
    end

    op_ge = [σ-1/6*ρ[x] for x in 1:6]
    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:6])
    end
    tr_eq = vcat([ [ ρ[x], 1] for x in 1:6])
    model, unique_vars, unique_mons = npa_model(M.vertices, level;
        obj=0,
        min=false,
        op_ge = op_ge,
        tr_eq = tr_eq,
        lvl_principal=0,
        cyclic=true,
        normalize=false)
    Γ=Dict(zip(unique_mons, unique_vars))
    @constraint(model, d,Γ[cyclic_reduce(cgb.monomial(σ))] <= start_G)

    @constraint(model, c[x=1:6,y=1:3], Γ[cyclic_reduce(ρ[x] * PB[1, y])] == prob_six(1, x, y, 1))


    # # key from 1 observable from Bob side
    # obj1B= cyclic_reduce((ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[2]+ρ[4]*PB[1,1]*PE[1]+ρ[4]*PB[2,1]*PE[2])/2)
    # # key from 1 observables from Alice side
    # obj1A=cyclic_reduce((ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[1]+ρ[4]*PB[1,1]*PE[2]+ρ[4]*PB[2,1]*PE[2])/2)
    # # key from 2 observables from Bob side
    # obj2B = cyclic_reduce((ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[2]+ρ[4]*PB[1,1]*PE[1]+ρ[4]*PB[2,1]*PE[2]+
    #              ρ[2]*PB[1,2]*PE[1]+ρ[2]*PB[2,2]*PE[2]+ρ[5]*PB[1,2]*PE[1]+ρ[5]*PB[2,2]*PE[2])/4)
    # # key from 2 observables from Alice's side
    # obj2A = cyclic_reduce((ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[1]+ρ[4]*PB[1,1]*PE[2]+ρ[4]*PB[2,1]*PE[2]+
    #              ρ[2]*PB[1,2]*PE[1]+ρ[2]*PB[2,2]*PE[1]+ρ[5]*PB[1,2]*PE[2]+ρ[5]*PB[2,2]*PE[2])/4)
    # # key from 3 observables from Bob side
    if !adapt
        obj3B = cyclic_reduce((ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[2]+ρ[4]*PB[1,1]*PE[1]+ρ[4]*PB[2,1]*PE[2]+
                    ρ[2]*PB[1,2]*PE[1]+ρ[2]*PB[2,2]*PE[2]+ρ[5]*PB[1,2]*PE[1]+ρ[5]*PB[2,2]*PE[2]+
                    ρ[3]*PB[1,3]*PE[1]+ρ[3]*PB[2,3]*PE[2]+ρ[6]*PB[1,3]*PE[1]+ρ[6]*PB[2,3]*PE[2])/6)
        # key from 3 observables from Alice's side
        obj3A = cyclic_reduce((ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[1]+ρ[4]*PB[1,1]*PE[2]+ρ[4]*PB[2,1]*PE[2]+
                    ρ[5]*PB[1,2]*PE[1]+ρ[5]*PB[2,2]*PE[1]+ρ[2]*PB[1,2]*PE[2]+ρ[2]*PB[2,2]*PE[2]+
                    ρ[3]*PB[1,3]*PE[1]+ρ[3]*PB[2,3]*PE[1]+ρ[6]*PB[1,3]*PE[2]+ρ[6]*PB[2,3]*PE[2])/6)
    else
        obj3B = cyclic_reduce((ρ[1]*PB[1,1]*PE[1,1]+ρ[1]*PB[2,1]*PE[2,1]+ρ[4]*PB[1,1]*PE[1,1]+ρ[4]*PB[2,1]*PE[2,1]+
                    ρ[2]*PB[1,2]*PE[1,2]+ρ[2]*PB[2,2]*PE[2,2]+ρ[5]*PB[1,2]*PE[1,2]+ρ[5]*PB[2,2]*PE[2,2]+
                    ρ[3]*PB[1,3]*PE[1,3]+ρ[3]*PB[2,3]*PE[2,3]+ρ[6]*PB[1,3]*PE[1,3]+ρ[6]*PB[2,3]*PE[2,3])/6)
        # key from 3 observables from Alice's side
        obj3A = cyclic_reduce((ρ[1]*PB[1,1]*PE[1,1]+ρ[1]*PB[2,1]*PE[1,1]+ρ[4]*PB[1,1]*PE[2,1]+ρ[4]*PB[2,1]*PE[2,1]+
                    ρ[5]*PB[1,2]*PE[1,2]+ρ[5]*PB[2,2]*PE[1,2]+ρ[2]*PB[1,2]*PE[2,2]+ρ[2]*PB[2,2]*PE[2,2]+
                    ρ[3]*PB[1,3]*PE[1,3]+ρ[3]*PB[2,3]*PE[1,3]+ρ[6]*PB[1,3]*PE[2,3]+ρ[6]*PB[2,3]*PE[2,3])/6)
    end
    obj= Alice ? obj3A : obj3B

    println("mosek starts")
    plx = [start_G+i/100 for i in 1:num_points]
    ply=ones(num_points)

    for i in 1:num_points
        set_normalized_rhs(d,plx[i])
        
        @objective(model, Max, sum(c*Γ[m] for (m,c) in obj))
        optimize!(model)
        ply[i] = objective_value(model)
        println(ply[i])
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        for i in 1:3
            GC.gc()
        end

        if ply[i]>0.9999
            break
        end
    end
    return plx,ply
end

"""
    pl_guess_prob_ten(G, level, fname; Alice=true, pure=true, adapt=false)

Computes Eve's guessing probability P_g^K for the 10-state QKD protocol as a function of detector efficiency η.

# Arguments
- `G`: Upper bound on Eve's state guessing probability P_g^A ≤ G
- `level`: NPA hierarchy level
- `fname`: Output filename to save results
- `Alice`: Whether to extract key from Alice's inputs (true) or Bob's outcomes (false)
- `pure`: Whether Alice's states are pure (default: true)
- `adapt`: Whether Eve can adapt her measurement strategy (default: false)

# Returns
- `(plx, ply)`: Tuple of (η values, guessing probabilities P_g^K)

# Protocol Context
Implements the extended 10-state protocol using 5 different angles (0, π/16, π/8, 3π/16, π/4)
with ± variants for each angle. This provides:
- Finer angular resolution for potentially better security bounds
- More measurement settings for enhanced parameter estimation
- Extended key generation from 5 compatible state-measurement pairs

The guessing probability accounts for key extraction from all 5 measurement bases,
with each pair contributing equally to the final key rate.
"""
function pl_guess_prob_ten(G,level,fname;Alice=true,pure=true,adapt=false)
    if !adapt
        @pcmonoid M B[5,0] E[1,0] BE[11,0]  # 5 measurements for Bob, 10 states + 1 constraint
    else
        @pcmonoid M B[5,0] E[5,0] BE[11,0]  # 5 measurements for both if adaptive
    end
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:10])
    end
    build(M)

    ρ = BE[1:10]  # Alice's 10 quantum states
    σ = BE[11]    # Auxiliary constraint variable
    PB=[B[1] B[2] B[3] B[4] B[5];1-B[1] 1-B[2] 1-B[3] 1-B[4] 1-B[5]]  # Bob's 5 measurements
    if !adapt
        PE=[E[1];1-E[1]]  # Eve's single measurement
    else
        PE=[E[1] E[2] E[3] E[4] E[5]; 1-E[1] 1-E[2] 1-E[3] 1-E[4] 1-E[5]]  # Eve's 5 adaptive measurements
    end

    # Guessing probability constraint: σ ≥ (1/10)ρₓ ∀x (uniform prior over 10 states)
    op_ge = [σ-0.1*ρ[x] for x in 1:10]
    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:10])
    end
    tr_eq = vcat([ [ ρ[x], 1] for x in 1:10])
    tr_ge = [ [-σ, -G]]
    
    model, unique_vars, unique_mons = npa_model(M.vertices, level;
        obj=0,
        min=false,
        op_ge = op_ge,
        tr_eq = tr_eq,
        tr_ge = tr_ge,
        lvl_principal=0,
        cyclic=true,
        normalize=false)
    Γ=Dict(zip(unique_mons, unique_vars))
    plx = [1-i/100 for i in 0:100]
    @constraint(model, c[x=1:10,y=1:5], Γ[cyclic_reduce(ρ[x] * PB[1, y])] == prob_ten(1, x, y, plx[1]))

    ply=ones(length(plx))
    for i in 1:length(plx)
        for x in 1:10, y in 1:5
            set_normalized_rhs(c[x,y], prob_ten(1, x, y, plx[i]))
        end
        
        # Key extraction from all 5 compatible measurement pairs
        if !adapt
            # Non-adaptive case: Eve has single measurement
            obj5B = cyclic_reduce((
                        # States 1,2 (angle 0) with measurement 1 (angle 0)
                        ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[2]+ρ[2]*PB[1,1]*PE[1]+ρ[2]*PB[2,1]*PE[2]+
                        # States 3,4 (angle π/16) with measurement 2 (angle π/16)  
                        ρ[3]*PB[1,2]*PE[1]+ρ[3]*PB[2,2]*PE[2]+ρ[4]*PB[1,2]*PE[1]+ρ[4]*PB[2,2]*PE[2]+
                        # States 5,6 (angle π/8) with measurement 3 (angle π/8)
                        ρ[5]*PB[1,3]*PE[1]+ρ[5]*PB[2,3]*PE[2]+ρ[6]*PB[1,3]*PE[1]+ρ[6]*PB[2,3]*PE[2]+
                        # States 7,8 (angle 3π/16) with measurement 4 (angle 3π/16)
                        ρ[7]*PB[1,4]*PE[1]+ρ[7]*PB[2,4]*PE[2]+ρ[8]*PB[1,4]*PE[1]+ρ[8]*PB[2,4]*PE[2]+
                        # States 9,10 (angle π/4) with measurement 5 (angle π/4)
                        ρ[9]*PB[1,5]*PE[1]+ρ[9]*PB[2,5]*PE[2]+ρ[10]*PB[1,5]*PE[1]+ρ[10]*PB[2,5]*PE[2])/10)
            
            # Alice's key extraction (correlates with her state preparation)
            obj5A = cyclic_reduce((
                        ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[1]+ρ[2]*PB[1,1]*PE[2]+ρ[2]*PB[2,1]*PE[2]+
                        ρ[3]*PB[1,2]*PE[1]+ρ[3]*PB[2,2]*PE[1]+ρ[4]*PB[1,2]*PE[2]+ρ[4]*PB[2,2]*PE[2]+
                        ρ[5]*PB[1,3]*PE[1]+ρ[5]*PB[2,3]*PE[1]+ρ[6]*PB[1,3]*PE[2]+ρ[6]*PB[2,3]*PE[2]+
                        ρ[7]*PB[1,4]*PE[1]+ρ[7]*PB[2,4]*PE[1]+ρ[8]*PB[1,4]*PE[2]+ρ[8]*PB[2,4]*PE[2]+
                        ρ[9]*PB[1,5]*PE[1]+ρ[9]*PB[2,5]*PE[1]+ρ[10]*PB[1,5]*PE[2]+ρ[10]*PB[2,5]*PE[2])/10)
        else
            # Adaptive case: Eve has 5 measurements matching each state type
            obj5B = cyclic_reduce((
                        ρ[1]*PB[1,1]*PE[1,1]+ρ[1]*PB[2,1]*PE[2,1]+ρ[2]*PB[1,1]*PE[1,1]+ρ[2]*PB[2,1]*PE[2,1]+
                        ρ[3]*PB[1,2]*PE[1,2]+ρ[3]*PB[2,2]*PE[2,2]+ρ[4]*PB[1,2]*PE[1,2]+ρ[4]*PB[2,2]*PE[2,2]+
                        ρ[5]*PB[1,3]*PE[1,3]+ρ[5]*PB[2,3]*PE[2,3]+ρ[6]*PB[1,3]*PE[1,3]+ρ[6]*PB[2,3]*PE[2,3]+
                        ρ[7]*PB[1,4]*PE[1,4]+ρ[7]*PB[2,4]*PE[2,4]+ρ[8]*PB[1,4]*PE[1,4]+ρ[8]*PB[2,4]*PE[2,4]+
                        ρ[9]*PB[1,5]*PE[1,5]+ρ[9]*PB[2,5]*PE[2,5]+ρ[10]*PB[1,5]*PE[1,5]+ρ[10]*PB[2,5]*PE[2,5])/10)
            
            obj5A = cyclic_reduce((
                        ρ[1]*PB[1,1]*PE[1,1]+ρ[1]*PB[2,1]*PE[1,1]+ρ[2]*PB[1,1]*PE[2,1]+ρ[2]*PB[2,1]*PE[2,1]+
                        ρ[3]*PB[1,2]*PE[1,2]+ρ[3]*PB[2,2]*PE[1,2]+ρ[4]*PB[1,2]*PE[2,2]+ρ[4]*PB[2,2]*PE[2,2]+
                        ρ[5]*PB[1,3]*PE[1,3]+ρ[5]*PB[2,3]*PE[1,3]+ρ[6]*PB[1,3]*PE[2,3]+ρ[6]*PB[2,3]*PE[2,3]+
                        ρ[7]*PB[1,4]*PE[1,4]+ρ[7]*PB[2,4]*PE[1,4]+ρ[8]*PB[1,4]*PE[2,4]+ρ[8]*PB[2,4]*PE[2,4]+
                        ρ[9]*PB[1,5]*PE[1,5]+ρ[9]*PB[2,5]*PE[1,5]+ρ[10]*PB[1,5]*PE[2,5]+ρ[10]*PB[2,5]*PE[2,5])/10)
        end

        obj = Alice ? obj5A : obj5B
        @objective(model, Max, sum(c*Γ[m] for (m,c) in obj))
        optimize!(model)
        ply[i] = objective_value(model)
        println(ply[i])
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        for i in 1:3
            GC.gc()
        end

        if ply[i]>0.9999
            break
        end
    end
    return plx,ply
end


"""
    pl_guess_prob_ten(level, fname; Alice=true, pure=true, adapt=false, num_points=67, start_G=0.2)

Computes Eve's guessing probability P_g^K for the 10-state protocol as a function of the bound G on P_g^A.

# Arguments
- `level`: NPA hierarchy level
- `fname`: Output filename to save results
- `Alice`: Whether to extract key from Alice's inputs (true) or Bob's outcomes (false)
- `pure`: Whether Alice's states are pure (default: true)
- `adapt`: Whether Eve can adapt her measurement strategy (default: false)
- `num_points`: Number of G values to compute (default: 67)
- `start_G`: Starting value for G (default: 0.2, accounting for 10 states)

# Returns
- `(plx, ply)`: Tuple of (G values, guessing probabilities P_g^K)

# Protocol Context
This version analyzes the 10-state protocol with perfect detectors (η=1) while varying
the information constraint G. The extended state set with 5 different angles may provide
better security bounds compared to the 4-state and 6-state protocols.

The default start_G=0.2 is above the theoretical minimum 1/10 for 10 equiprobable states,
allowing exploration of the trade-off between state distinguishability and key security.
"""
function pl_guess_prob_ten(level,fname;Alice=true,pure=true,adapt=false,num_points=67,start_G=0.2)
    if !adapt
        @pcmonoid M B[5,0] E[1,0] BE[11,0]
    else
        @pcmonoid M B[5,0] E[5,0] BE[11,0] 
    end
    @comms B E
    Projector.([B;E])
    if pure
        Projector.(BE[1:10])
    end
    build(M)

    ρ = BE[1:10]
    σ = BE[11] 
    PB=[B[1] B[2] B[3] B[4] B[5];1-B[1] 1-B[2] 1-B[3] 1-B[4] 1-B[5]]
    if !adapt
        PE=[E[1];1-E[1]]
    else
        PE=[E[1] E[2] E[3] E[4] E[5]; 1-E[1] 1-E[2] 1-E[3] 1-E[4] 1-E[5]]
    end

    op_ge = [σ-0.1*ρ[x] for x in 1:10]  # 1/10 factor for 10 states
    if ! pure
        op_ge = vcat(op_ge, [ρ[x]-ρ[x]*ρ[x] for x in 1:10])
    end
    tr_eq = vcat([ [ ρ[x], 1] for x in 1:10])
    model, unique_vars, unique_mons = npa_model(M.vertices, level;
        obj=0,
        min=false,
        op_ge = op_ge,
        tr_eq = tr_eq,
        lvl_principal=0,
        cyclic=true,
        normalize=false)
    Γ=Dict(zip(unique_mons, unique_vars))
    @constraint(model, d,Γ[cyclic_reduce(cgb.monomial(σ))] <= start_G)

    @constraint(model, c[x=1:10,y=1:5], Γ[cyclic_reduce(ρ[x] * PB[1, y])] == prob_ten(1, x, y, 1))

    # Key extraction objectives (same as previous function)
    if !adapt
        obj5B = cyclic_reduce((
                    ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[2]+ρ[2]*PB[1,1]*PE[1]+ρ[2]*PB[2,1]*PE[2]+
                    ρ[3]*PB[1,2]*PE[1]+ρ[3]*PB[2,2]*PE[2]+ρ[4]*PB[1,2]*PE[1]+ρ[4]*PB[2,2]*PE[2]+
                    ρ[5]*PB[1,3]*PE[1]+ρ[5]*PB[2,3]*PE[2]+ρ[6]*PB[1,3]*PE[1]+ρ[6]*PB[2,3]*PE[2]+
                    ρ[7]*PB[1,4]*PE[1]+ρ[7]*PB[2,4]*PE[2]+ρ[8]*PB[1,4]*PE[1]+ρ[8]*PB[2,4]*PE[2]+
                    ρ[9]*PB[1,5]*PE[1]+ρ[9]*PB[2,5]*PE[2]+ρ[10]*PB[1,5]*PE[1]+ρ[10]*PB[2,5]*PE[2])/10)
        
        obj5A = cyclic_reduce((
                    ρ[1]*PB[1,1]*PE[1]+ρ[1]*PB[2,1]*PE[1]+ρ[2]*PB[1,1]*PE[2]+ρ[2]*PB[2,1]*PE[2]+
                    ρ[3]*PB[1,2]*PE[1]+ρ[3]*PB[2,2]*PE[1]+ρ[4]*PB[1,2]*PE[2]+ρ[4]*PB[2,2]*PE[2]+
                    ρ[5]*PB[1,3]*PE[1]+ρ[5]*PB[2,3]*PE[1]+ρ[6]*PB[1,3]*PE[2]+ρ[6]*PB[2,3]*PE[2]+
                    ρ[7]*PB[1,4]*PE[1]+ρ[7]*PB[2,4]*PE[1]+ρ[8]*PB[1,4]*PE[2]+ρ[8]*PB[2,4]*PE[2]+
                    ρ[9]*PB[1,5]*PE[1]+ρ[9]*PB[2,5]*PE[1]+ρ[10]*PB[1,5]*PE[2]+ρ[10]*PB[2,5]*PE[2])/10)
    else
        obj5B = cyclic_reduce((
                    ρ[1]*PB[1,1]*PE[1,1]+ρ[1]*PB[2,1]*PE[2,1]+ρ[2]*PB[1,1]*PE[1,1]+ρ[2]*PB[2,1]*PE[2,1]+
                    ρ[3]*PB[1,2]*PE[1,2]+ρ[3]*PB[2,2]*PE[2,2]+ρ[4]*PB[1,2]*PE[1,2]+ρ[4]*PB[2,2]*PE[2,2]+
                    ρ[5]*PB[1,3]*PE[1,3]+ρ[5]*PB[2,3]*PE[2,3]+ρ[6]*PB[1,3]*PE[1,3]+ρ[6]*PB[2,3]*PE[2,3]+
                    ρ[7]*PB[1,4]*PE[1,4]+ρ[7]*PB[2,4]*PE[2,4]+ρ[8]*PB[1,4]*PE[1,4]+ρ[8]*PB[2,4]*PE[2,4]+
                    ρ[9]*PB[1,5]*PE[1,5]+ρ[9]*PB[2,5]*PE[2,5]+ρ[10]*PB[1,5]*PE[1,5]+ρ[10]*PB[2,5]*PE[2,5])/10)
        
        obj5A = cyclic_reduce((
                    ρ[1]*PB[1,1]*PE[1,1]+ρ[1]*PB[2,1]*PE[1,1]+ρ[2]*PB[1,1]*PE[2,1]+ρ[2]*PB[2,1]*PE[2,1]+
                    ρ[3]*PB[1,2]*PE[1,2]+ρ[3]*PB[2,2]*PE[1,2]+ρ[4]*PB[1,2]*PE[2,2]+ρ[4]*PB[2,2]*PE[2,2]+
                    ρ[5]*PB[1,3]*PE[1,3]+ρ[5]*PB[2,3]*PE[1,3]+ρ[6]*PB[1,3]*PE[2,3]+ρ[6]*PB[2,3]*PE[2,3]+
                    ρ[7]*PB[1,4]*PE[1,4]+ρ[7]*PB[2,4]*PE[1,4]+ρ[8]*PB[1,4]*PE[2,4]+ρ[8]*PB[2,4]*PE[2,4]+
                    ρ[9]*PB[1,5]*PE[1,5]+ρ[9]*PB[2,5]*PE[1,5]+ρ[10]*PB[1,5]*PE[2,5]+ρ[10]*PB[2,5]*PE[2,5])/10)
    end
    obj = Alice ? obj5A : obj5B

    println("mosek starts")
    plx = [start_G+i/100 for i in 1:num_points]
    ply=ones(num_points)

    for i in 1:num_points
        set_normalized_rhs(d,plx[i])
        
        @objective(model, Max, sum(c*Γ[m] for (m,c) in obj))
        optimize!(model)
        ply[i] = objective_value(model)
        println(ply[i])
        open(fname,"a") do file
            write(file,"$(plx[i]) $(ply[i])\n")
        end
        for i in 1:3
            GC.gc()
        end

        if ply[i]>0.9999
            break
        end
    end
    return plx,ply
end