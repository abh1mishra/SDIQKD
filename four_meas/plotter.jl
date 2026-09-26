using DelimitedFiles
using Plots
using LaTeXStrings
include("./four_meas_corr.jl")

# Legend label for a data file, parsed from the G=... part of the filename.
# 4-state scans start at G = 2/4 = 0.5 and step by 0.01, so the 2-decimal
# filename value is exact: print it as is (0.5, 0.51, ...).
# 6-state scans start at G = 1/3 and step by 0.01, so the true value is a
# repeating decimal (0.333..., 0.3433..., ...): print the 2-decimal prefix
# followed by \bar{3} (renders as 0.33̄3, 0.34̄3, ...).
function G_label(file)
    m = match(r"G=([\d\.]+)\.txt$", file)
    m === nothing && return replace(file, ".txt" => "")
    g = m.captures[1]
    return file[1] == '6' ? latexstring("G=", g, "\\bar{3}") : "G=$(g)"
end

# Thin a sorted list of tick values so that no two kept ticks are closer than
# `min_gap`. The first and last values are always kept (so the axis range still
# shows its endpoints); interior ticks that would crowd a kept neighbour are
# dropped. Well-spread tick sets pass through unchanged.
function thin_ticks(vals; min_gap=0.03)
    length(vals) <= 2 && return vals
    kept = [vals[1]]
    for v in vals[2:end-1]
        if v - kept[end] >= min_gap
            push!(kept, v)
        end
    end
    # Make room for the final tick: drop kept ticks that would crowd it.
    while length(kept) > 1 && vals[end] - kept[end] < min_gap
        pop!(kept)
    end
    push!(kept, vals[end])
    return kept
end

# Probability (uniform x,y) that a round is key-generating, converting the
# per-sifted-round key rate to per-emitted-signal.
# four_meas 4-state: (x∈{1,2},y=1) + (x∈{3,4},y=4) → 2·(2/4)·(1/4) = 1/4.
# four_meas 6-state: pairs (1,4)↔y=1, (2,5)↔y=2, (3,6)↔y=3 → 3·(2/6)·(1/4) = 1/4.
p_succ(file) = 1/4

function process_club_eta(folder)
    for (root, dirs, files) in walkdir(folder)
        # Check if current directory name starts with "club"
        current_dir = basename(root)
        if startswith(current_dir, "club")
            # Parse visibility from this club's own path (e.g. ".../v=0.98/club_...")
            vm = match(r"v=([\d\.]+)", root)
            v = vm !== nothing ? parse(Float64, vm.captures[1]) : 1.0
            println("v=", v)

            # Extract title by removing "club_" prefix
            plot_title = replace(current_dir, r"^club_?" => "")

            all_eta = Float64[]

            # Collect per-series extremes
            x_extremes = Float64[]
            y_extremes = Float64[0.0]  # Always include 0 as y minimum

            # Initialize plot
            p = plot(xlabel="η", ylabel="key rate", tickfontsize=16, legendfontsize=12, guidefontsize=16, bottom_margin=10Plots.mm)

            for file in files
                if endswith(file, ".txt")
                    filepath = joinpath(root, file)
                    data = readdlm(filepath)
                    entropy = data[:, 2]
                    eta = data[:, 1]

                    # Calculate HAB based on file name
                    if parse(Int, file[1]) == 4
                        if file[2] == 'A' || file[2] == 'a'
                            HAB = conditional_entropy_four_A.(eta; v=v, bin=false)
                        elseif file[2] == 'B' || file[2] == 'b'
                            HAB = conditional_entropy_four_B.(eta; v=v, bin=true)
                        else
                            throw("Not yet implemented")
                        end
                    elseif parse(Int, file[1]) == 6
                        if file[2] == 'A' || file[2] == 'a'
                            HAB = conditional_entropy_six_A.(eta; v=v, bin=false)
                        elseif file[2] == 'B' || file[2] == 'b'
                            HAB = conditional_entropy_six_B.(eta; v=v, bin=true)
                        else
                            throw("Not yet implemented")
                        end
                    else
                        throw("Not yet implemented")
                    end

                    keyrate = (entropy - HAB) * p_succ(file)

                    # Sort data by eta values for smooth line plotting
                    perm = sortperm(eta)
                    eta_sorted = eta[perm]
                    keyrate_sorted = max.(keyrate[perm], 0)  # Replace negative values with 0
                    # Trim trailing zeros, keep only the first zero
                    last_pos = findfirst(keyrate_sorted .> 0)
                    if last_pos !== nothing && last_pos < length(keyrate_sorted)
                        start_idx = max(last_pos - 1, 1)
                        eta_sorted = eta_sorted[start_idx:end]
                        keyrate_sorted = keyrate_sorted[start_idx:end]
                    end

                    plot!(p, eta_sorted, keyrate_sorted, label=G_label(file))

                    # Keep track of per-series extremes
                    push!(x_extremes, round(eta_sorted[1], digits=3))
                    push!(x_extremes, round(eta_sorted[end], digits=3))
                    push!(y_extremes, round(minimum(keyrate_sorted), digits=3))
                    push!(y_extremes, round(maximum(keyrate_sorted), digits=3))
                    append!(all_eta, eta_sorted)
                end
            end

            # Save the combined plot
            if !isempty(all_eta)
                # X-axis: per-series extremes, thinned to avoid crammed ticks, rotated 90°
                xtick_vals = thin_ticks(sort(unique(x_extremes)))
                xticks!(p, xtick_vals)
                plot!(p, xrotation=90)

                # Y-axis: overall min to max with nice evenly spaced ticks
                y_min = minimum(y_extremes)
                y_max = maximum(y_extremes)
                ytick_vals = round.(range(y_min, y_max, length=6), digits=3)
                yticks!(p, ytick_vals)

                pdfname = joinpath(root, "$(plot_title).pdf")
                savefig(p, pdfname)
                println("Saved combined plot: $pdfname")
            end
        end
    end
end

function process_club_visibility(folder)
    for (root, dirs, files) in walkdir(folder)
        # Check if current directory name starts with "club"
        current_dir = basename(root)
        if startswith(current_dir, "club")
            # Parse detection efficiency from this club's own path (e.g. ".../eta=1.0/club_...")
            em = match(r"eta=([\d\.]+)", root)
            eta = em !== nothing ? parse(Float64, em.captures[1]) : 1.0
            println("eta=", eta)

            # Extract title by removing "club_" prefix
            plot_title = replace(current_dir, r"^club_?" => "")

            # Extract level from the first .txt file (should be same for all files in folder)
            level = ""
            for file in files
                if endswith(file, ".txt")
                    # Extract level between underscores (e.g., "1+B*E" from "4Alice_1+B*E_G=0.5.txt")
                    parts = split(file, '_')
                    if length(parts) >= 3
                        level = parts[2]  # The part between first and second underscore
                    end
                    break  # Only need to check first file since level is same for all
                end
            end

            # Update title to include level
            if !isempty(level)
                plot_title = "$(plot_title)_$(level)"
            end

            all_v = Float64[]

            # Collect per-series extremes
            x_extremes = Float64[]
            y_extremes = Float64[0.0]  # Always include 0 as y minimum

            # Initialize plot
            p = plot(xlabel="v", ylabel="key rate", tickfontsize=16, legendfontsize=12, guidefontsize=16, bottom_margin=10Plots.mm)

            for file in files
                if endswith(file, ".txt")
                    filepath = joinpath(root, file)
                    data = readdlm(filepath)
                    entropy = data[:, 2]
                    v = data[:, 1]

                    # Calculate HAB based on file name
                    if parse(Int, file[1]) == 4
                        if file[2] == 'A' || file[2] == 'a'
                            HAB = [conditional_entropy_four_A(eta; v=i, bin=false) for i in v]
                        elseif file[2] == 'B' || file[2] == 'b'
                            HAB = [conditional_entropy_four_B(eta; v=i, bin=true) for i in v]
                        else
                            throw("Not yet implemented")
                        end
                    elseif parse(Int, file[1]) == 6
                        if file[2] == 'A' || file[2] == 'a'
                            HAB = [conditional_entropy_six_A(eta; v=i, bin=false) for i in v]
                        elseif file[2] == 'B' || file[2] == 'b'
                            HAB = [conditional_entropy_six_B(eta; v=i, bin=true) for i in v]
                        else
                            throw("Not yet implemented")
                        end
                    else
                        throw("Not yet implemented")
                    end

                    keyrate = (entropy - HAB) * p_succ(file)

                    # Sort data by visibility values for smooth line plotting
                    perm = sortperm(v)
                    v_sorted = v[perm]
                    keyrate_sorted = max.(keyrate[perm], 0)  # Replace negative values with 0
                    # Trim trailing zeros, keep only the first zero
                    last_pos = findfirst(keyrate_sorted .> 0)
                    if last_pos !== nothing && last_pos < length(keyrate_sorted)
                        start_idx = max(last_pos - 1, 1)
                        v_sorted = v_sorted[start_idx:end]
                        keyrate_sorted = keyrate_sorted[start_idx:end]
                    end

                    plot!(p, v_sorted, keyrate_sorted, label=G_label(file))

                    # Keep track of per-series extremes
                    push!(x_extremes, round(v_sorted[1], digits=3))
                    push!(x_extremes, round(v_sorted[end], digits=3))
                    push!(y_extremes, round(minimum(keyrate_sorted), digits=3))
                    push!(y_extremes, round(maximum(keyrate_sorted), digits=3))
                    append!(all_v, v_sorted)
                end
            end

            # Save the combined plot
            if !isempty(all_v)
                # X-axis: per-series extremes, thinned to avoid crammed ticks, rotated 90°
                xtick_vals = thin_ticks(sort(unique(x_extremes)))
                xticks!(p, xtick_vals)
                plot!(p, xrotation=90)

                # Y-axis: overall min to max with nice evenly spaced ticks
                y_min = minimum(y_extremes)
                y_max = maximum(y_extremes)
                ytick_vals = round.(range(y_min, y_max, length=6), digits=3)
                yticks!(p, ytick_vals)

                pdfname = joinpath(root, "$(plot_title).pdf")
                savefig(p, pdfname)
                println("Saved combined plot: $pdfname")
            end
        end
    end
end
