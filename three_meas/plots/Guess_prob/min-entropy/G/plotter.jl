using DelimitedFiles
using Plots

function min_entropy(p)
    return -log2(max(p,1-p))
end

function process_folder(folder)
    for (root, dirs, files) in walkdir(folder)
        for file in files
            if endswith(file, ".txt")
                filepath = joinpath(root, file)
                data = readdlm(filepath)
                eta = data[:, 1]
                prob = data[:, 2]
                entropy = min_entropy.(prob)
                plot(eta, entropy, label="Min entropy", xlabel="G", ylabel="Min-Entropy", title=file)
                pngname = replace(filepath, ".txt" => ".png")
                savefig(pngname)
                # Removed close()
            end
        end
    end
end

process_folder("./adapt")
process_folder("./non-adapt")