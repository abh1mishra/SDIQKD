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
    state_six(x)

Generates one of six quantum states for the 6-state QKD protocol.

# Arguments
- `x`: State index (1 to 6)

# Returns
- Density matrix of the corresponding quantum state

# Protocol Context
Implements the 6-state protocol described in the paper:
- States 1,4: |ψ₀±⟩ = |0⟩, |1⟩ (Z-basis eigenstates)
- States 2,5: |ψ_{π/4}±⟩ = |+⟩, |-⟩ (X-basis eigenstates)  
- States 3,6: |ψ_{π/8}±⟩ = eigenstates of (Z+X)/√2

Key generation uses:
- Z measurement on states 1,4
- X measurement on states 2,5
- (Z+X)/√2 measurement on states 3,6
"""
function state_six(x;v=1)
    if x%3==1
        return x<4 ? state(0,1;v=v) : state(0,2;v=v)
    elseif x%3==2
        return x<4 ? state(pi/4,1;v=v) : state(pi/4,2;v=v)
    else
        return x<4 ? state(pi/8,1;v=v) : state(pi/8,2;v=v)
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
- States 3,4: |-⟩, |+⟩ (X-basis)

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
    povm_six(y, b; η=1)

Creates POVM elements for Bob's measurements in the 6-state protocol.

# Arguments
- `y`: Measurement choice index (1 to 3)
- `b`: Measurement outcome (1 or 2)  
- `η`: Detection efficiency (default: 1 for perfect detectors)

# Returns
- POVM element for measurement y, outcome b

# Protocol Context
Implements Bob's three measurement choices:
- y=1: Z measurement (computational basis)
- y=2: X measurement (Hadamard basis)
- y=3: (Z+X)/√2 measurement (intermediate basis)

These measurements are used both for key generation (when paired with
appropriate Alice states) and parameter estimation.
"""
function povm_six(y,b;η=1,bin=true)
    if y==1
        return povm(0,b,η;bin=bin)
    elseif y==2
        return povm(pi/4,b,η;bin=bin)
    else
        return povm(pi/8,b,η;bin=bin)
    end
end

"""
    povm_four(y, b; η=1)

Creates POVM elements for Bob's measurements in the 4-state protocol.

# Arguments
- `y`: Measurement choice index (1 to 3)
- `b`: Measurement outcome (1 or 2)
- `η`: Detection efficiency (default: 1 for perfect detectors)

# Returns
- POVM element for measurement y, outcome b

# Protocol Context
Implements Bob's three measurement choices for the 4-state protocol:
- y=1: Z measurement
- y=2: (Z+X)/√2 measurement (rotated by +π/8)
- y=3: (Z-X)/√2 measurement (rotated by -π/8)

The Z measurement on Alice's Z-basis states is used for key generation.
"""
function povm_four(y,b;η=1,bin=true)
    if y==1
        return povm(0,b,η;bin=bin)
    elseif y==2
        return povm(pi/8,b,η;bin=bin)
    else
        return povm(-pi/8,b,η;bin=bin)
    end
end

"""
    prob_four(b, x, y)

Calculates the probability p(b|x,y) for the 4-state QKD protocol.

# Arguments
- `b`: Bob's measurement outcome (1 or 2)
- `x`: Alice's state choice (1 to 4)
- `y`: Bob's measurement choice (1 to 3)

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
    prob_six(b, x, y)

Calculates the probability p(b|x,y) for the 6-state QKD protocol.

# Arguments
- `b`: Bob's measurement outcome (1 or 2)
- `x`: Alice's state choice (1 to 6)
- `y`: Bob's measurement choice (1 to 3)

# Returns
- Probability of outcome b given Alice prepared state x and Bob measured with setting y

# Protocol Context
Computes conditional probabilities for the enhanced 6-state protocol.
These probabilities determine both the key generation rate and security bounds
through the guessing probabilities P_g^K and constraints on Eve's information.
"""
function prob_six(b,x,y;bin=true)
    tr(state_six(x)*povm_six(y,b;bin=bin))
end

"""
    prob_four(b, x, y, η)

Calculates noisy probabilities for the 4-state protocol with detector inefficiency.

# Arguments
- `b`: Bob's measurement outcome (1 or 2)
- `x`: Alice's state choice (1 to 4)
- `y`: Bob's measurement choice (1 to 3)
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

"""
    prob_six(b, x, y, η)

Calculates noisy probabilities for the 6-state protocol with detector inefficiency.

# Arguments
- `b`: Bob's measurement outcome (1 or 2)
- `x`: Alice's state choice (1 to 6)
- `y`: Bob's measurement choice (1 to 3)
- `η`: Detection efficiency parameter

# Returns
- Probability including detector losses and noise effects

# Protocol Context
Enables security analysis of the 6-state protocol under realistic conditions
with lossy detectors, providing the p(b|x,y) distributions needed for
the NPA hierarchy optimization under device imperfections.
"""
function prob_six(b,x,y,η;v=1,bin=true)
    tr(state_six(x;v=v)*povm_six(y,b;η=η,bin=bin))
end

"""
    state_ten(x)

Generates one of ten quantum states for the 10-state QKD protocol.

# Arguments
- `x`: State index (1 to 10)

# Returns
- Density matrix of the corresponding quantum state

# Protocol Context
Implements an extended protocol using 5 different angles (0, π/16, π/8, 3π/16, π/4)
with ± variants for each angle:
- States 1,2: θ=0 (|0⟩, |1⟩)
- States 3,4: θ=π/16  
- States 5,6: θ=π/8
- States 7,8: θ=3π/16
- States 9,10: θ=π/4 (|+⟩, |-⟩)

This extended state set provides finer angular resolution for potentially
improved security bounds and key generation rates.
"""
function state_ten(x)
    # 10 states using 5 angles: 0, π/16, π/8, 3π/16, π/4
    # States 1,2: θ=0; States 3,4: θ=π/16; States 5,6: θ=π/8; 
    # States 7,8: θ=3π/16; States 9,10: θ=π/4
    if x <= 2
        return x == 1 ? state(0, 1) : state(0, 2)
    elseif x <= 4
        return x == 3 ? state(π/16, 1) : state(π/16, 2)
    elseif x <= 6
        return x == 5 ? state(π/8, 1) : state(π/8, 2)
    elseif x <= 8
        return x == 7 ? state(3π/16, 1) : state(3π/16, 2)
    else
        return x == 9 ? state(π/4, 1) : state(π/4, 2)
    end
end

"""
    povm_ten(y, b; η=1)

Creates POVM elements for Bob's measurements in the 10-state protocol.

# Arguments
- `y`: Measurement choice index (1 to 5)
- `b`: Measurement outcome (1 or 2)
- `η`: Detection efficiency (default: 1)

# Returns
- POVM element for measurement y, outcome b

# Protocol Context
Implements five measurement bases corresponding to the five angles
used in the 10-state protocol. This provides a richer measurement
set for enhanced parameter estimation and potentially better security bounds.
"""
function povm_ten(y, b; η=1)
    # 5 measurements corresponding to the 5 angles
    if y == 1
        return povm(0, b, η)
    elseif y == 2
        return povm(π/16, b, η)
    elseif y == 3
        return povm(π/8, b, η)
    elseif y == 4
        return povm(3π/16, b, η)
    else
        return povm(π/4, b, η)
    end
end

"""
    prob_ten(b, x, y)

Calculates the probability p(b|x,y) for the 10-state QKD protocol.

# Arguments
- `b`: Bob's measurement outcome (1 or 2)
- `x`: Alice's state choice (1 to 10)  
- `y`: Bob's measurement choice (1 to 5)

# Returns
- Probability of outcome b given Alice prepared state x and Bob measured with setting y

# Protocol Context
Computes conditional probabilities for the extended 10-state protocol,
providing the statistical data needed for security analysis and key
generation in this more sophisticated variant.
"""
function prob_ten(b, x, y;bin=true)
    tr(state_ten(x) * povm_ten(y, b;bin=bin))
end

"""
    prob_ten(b, x, y, η)

Calculates noisy probabilities for the 10-state protocol with detector inefficiency.

# Arguments
- `b`: Bob's measurement outcome (1 or 2)
- `x`: Alice's state choice (1 to 10)
- `y`: Bob's measurement choice (1 to 5)  
- `η`: Detection efficiency parameter

# Returns
- Probability including detector losses and noise effects

# Protocol Context
Models the 10-state protocol under realistic conditions with imperfect
detectors, enabling security analysis that accounts for implementation
imperfections in this extended protocol variant.
"""
function prob_ten(b, x, y, η;bin=true)
    tr(state_ten(x) * povm_ten(y, b; η=η,bin=bin))
end

function jp_four(eta;v=1,bin=true)
    if bin
        p11=abs(0.5*prob_four(1,1,1,eta;v=v,bin=true))
        p12=abs(0.5*prob_four(2,1,1,eta;v=v,bin=true))
        p21=abs(0.5*prob_four(1,2,1,eta;v=v,bin=true))
        p22=abs(0.5*prob_four(2,2,1,eta;v=v,bin=true))
        return [p11 p12; p21 p22]
    else
        p11=abs(0.5*prob_four(1,1,1,eta;v=v,bin=false))
        p12=abs(0.5*prob_four(2,1,1,eta;v=v,bin=false))
        p13=abs(0.5*prob_four(3,1,1,eta;v=v,bin=false))
        p21=abs(0.5*prob_four(1,2,1,eta;v=v,bin=false))
        p22=abs(0.5*prob_four(2,2,1,eta;v=v,bin=false))
        p23=abs(0.5*prob_four(3,2,1,eta;v=v,bin=false))
        return [p11 p12 p13; p21 p22 p23]
    end
end

function jp_six(r,eta;v=1,bin=true)
    x1,x2,y=0,0,0
    if r==1
        x1,x2,y=1,4,1
    elseif r==2
        x1,x2,y=2,5,2
    elseif r==3
        x1,x2,y=3,6,3
    end
    if bin
        p11=abs(0.5*prob_six(1,x1,y,eta;v=v,bin=true))
        p12=abs(0.5*prob_six(2,x1,y,eta;v=v,bin=true))
        p21=abs(0.5*prob_six(1,x2,y,eta;v=v,bin=true))
        p22=abs(0.5*prob_six(2,x2,y,eta;v=v,bin=true))
        return [p11 p12; p21 p22]
    else
        p11=abs(0.5*prob_six(1,x1,y,eta;v=v,bin=false))
        p12=abs(0.5*prob_six(2,x1,y,eta;v=v,bin=false))
        p13=abs(0.5*prob_six(3,x1,y,eta;v=v,bin=false))
        p21=abs(0.5*prob_six(1,x2,y,eta;v=v,bin=false))
        p22=abs(0.5*prob_six(2,x2,y,eta;v=v,bin=false))
        p23=abs(0.5*prob_six(3,x2,y,eta;v=v,bin=false))
        return [p11 p12 p13; p21 p22 p23]
    end
end

function conditional_entropy_four_A(eta;v=1,bin=true)
    jp=jp_four(eta;v=v,bin=bin)
    return conditional_entropy(jp)
end

function conditional_entropy_six_A(eta;v=1,bin=true)
    return sum(conditional_entropy(jp_six(r,eta;v=v,bin=bin)) for r in 1:3)/3
end

function conditional_entropy_four_B(eta;v=1,bin=true)
    jp=Matrix(jp_four(eta;v=v,bin=bin)')
    return conditional_entropy(jp)
end

function conditional_entropy_six_B(eta;v=1,bin=true)
    return sum(conditional_entropy(Matrix(jp_six(r,eta;v=v,bin=bin)')) for r in 1:3)/3
end