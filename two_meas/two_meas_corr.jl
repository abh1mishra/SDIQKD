using LinearAlgebra
include("../utils.jl")
"""
    state(θ, x)

Creates a quantum state for the prepare-and-measure QKD protocol.

# Arguments
- `θ`: Angle parameter (in radians) that determines the state orientation on the Bloch sphere
- `x`: Binary input (1 or 2) determining which eigenstate to prepare

# Returns
- `ρ`: A 2×2 density matrix representing the quantum state
  - If x=1: Returns the density matrix |ψ⟩⟨ψ| where |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
  - If x=2: Returns the orthogonal state density matrix (I - |ψ⟩⟨ψ|)

# Protocol Context
This function implements the fundamental state preparation in the QKD protocol.
For different angles θ, it generates the required quantum states:
- θ=0: |0⟩ and |1⟩ states (Z-basis)
- θ=π/4: |+⟩ and |-⟩ states (X-basis)  
- θ=π/8: Eigenstates of (Z+X)/√2 observable
"""
function state(θ,x;v=1)
    ket=cos(θ).*[1 0]+sin(θ).*[0 1]
    Id=[1 0; 0 1]
    ρ=v*ket'*ket+(1-v)*Id/2
    return x==1 ? ρ : Id-ρ
end

"""
    povm(θ, b)

Creates a POVM (Positive Operator-Valued Measure) element for Bob's measurements.

# Arguments
- `θ`: Angle parameter determining the measurement basis
- `b`: Binary measurement outcome (1 or 2)

# Returns
- POVM element corresponding to measurement outcome b in the θ-basis

# Protocol Context
This implements Bob's measurement operators in the QKD protocol.
Each measurement basis corresponds to an observable like Z, X, or (Z+X)/√2.
"""
function povm(θ,b)
    return state(θ,b)
end

"""
    povm(θ, b, η)

Creates a noisy POVM element accounting for detector inefficiency.

# Arguments
- `θ`: Angle parameter determining the measurement basis
- `b`: Binary measurement outcome (1 or 2)
- `η`: Detection efficiency parameter (0 ≤ η ≤ 1)

# Returns
- Noisy POVM element accounting for detector losses and inefficiencies

# Protocol Context
Models realistic detector behavior in QKD where:
- η=1: Perfect detection (ideal case)
- η<1: Lossy detectors that may fail to click
For outcome b=1, adds a contribution proportional to (1-η)I representing no-click events.
"""
function povm(θ,b,η;bin=true)
    if bin
        if b==1
            η.*povm(θ,b)+(1-η).*I(2)
        elseif b==2
            η.*povm(θ,b)
        else
            error("Invalid outcome b for binary POVM. Must be 1 or 2.")
        end
    else
        if b==1 || b==2
            η.*povm(θ,b)
        else
            Matrix{Float64}((1-η).*I(2))
        end
    end
end


"""
    state_four(x)

Generates one of four quantum states for the 4-state CHSH-like QKD protocol.

# Arguments
- `x`: State index (1 to 4)

# Returns
- Density matrix of the corresponding quantum state

# Protocol Context
Implements the simplified 4-state protocol:
- States 1,2: |0⟩, |1⟩ (Z-basis)
- States 3,4: |+⟩, |-⟩ (X-basis)

This is the minimal CHSH-like protocol mentioned in the paper that provides
security against attacks exploiting shared randomness between Alice and Bob.
"""
function state_four(x;v=1)
    if x==1
        return state(0,1;v=v)
    elseif x==2
        return state(0,2;v=v)
    elseif x==3
        return state(pi/4,1;v=v)
    else
        return state(pi/4,2;v=v)
    end
end



"""
    povm_four(y, b; η=1)

Creates POVM elements for Bob's measurements in the 4-state protocol.

# Arguments
- `y`: Measurement choice index (1 to 4)
- `b`: Measurement outcome (1 or 2)
- `η`: Detection efficiency (default: 1 for perfect detectors)

# Returns
- POVM element for measurement y, outcome b

# Protocol Context
Implements Bob's three measurement choices for the 4-state protocol:
- y=1: Z measurement
- y=2: (Z+X)/√2 measurement (rotated by +π/8)

The Z measurement on Alice's Z-basis states and X measurement on Alice's X-basis states are used for key generation.
"""
function povm_four(y,b;η=1,bin=true)
    if y==1
        return povm(0,b,η;bin=bin)
    elseif y==2
        return povm(pi/8,b,η;bin=bin)
    end

end

"""
    prob_four(b, x, y)

Calculates the probability p(b|x,y) for the 4-state QKD protocol.

# Arguments
- `b`: Bob's measurement outcome (1 or 2)
- `x`: Alice's state choice (1 to 4)
- `y`: Bob's measurement choice (1 to 2)

# Returns
- Probability of outcome b given Alice prepared state x and Bob measured with setting y

# Protocol Context
Computes the conditional probability distributions that are observed in the
4-state protocol. These probabilities are used for:
- Parameter estimation (to bound Eve's information)
- Key generation (for compatible state-measurement pairs)
- Security analysis via the NPA hierarchy
"""
function prob_four(b,x,y;bin=true)
    tr(state_four(x)*povm_four(y,b;bin=bin))
end


"""
    prob_four(b, x, y, η)

Calculates noisy probabilities for the 4-state protocol with detector inefficiency.

# Arguments
- `b`: Bob's measurement outcome (1 or 2)
- `x`: Alice's state choice (1 to 4)
- `y`: Bob's measurement choice (1 to 2)
- `η`: Detection efficiency parameter

# Returns
- Probability including detector losses and noise effects

# Protocol Context
Models realistic implementation with imperfect detectors, accounting for
the detection efficiency η in the probability calculations used for
security analysis under device imperfections.
"""
function prob_four(b,x,y,η;v=1,bin=true)
    tr(state_four(x;v=v)*povm_four(y,b;η=η,bin=bin))
end

function jp_four(eta;v=1,bin=true)
    x1, x2, y = (1,2,1)
    nb = bin ? 2 : 3
    return [abs(0.5*prob_four(b, x, y, eta; v=v, bin=bin)) for x in (x1,x2), b in 1:nb]
end


function conditional_entropy_four_A(eta;v=1,bin=true)
    return conditional_entropy(jp_four(eta;v=v,bin=bin))
end

function conditional_entropy_four_B(eta;v=1,bin=true)
    return conditional_entropy(Matrix(jp_four(eta;v=v,bin=bin)')) 
end