# KomaMRI worker for PROAQMR.
# Listens on 127.0.0.1:$KOMA_PORT (default 8011, internal only; RunPod uses 8001 for its own proxy). The Python API sends it a Pulseq
# file and a phantom file; it runs the Bloch simulation and writes the raw
# signal (Float32 real/imag interleaved, ADC samples in playback order).
using KomaMRIBase, KomaMRICore, KomaMRIFiles
using CUDA
using HTTP
using JSON3

const GPU = try
    CUDA.functional()
catch
    false
end
println("KomaMRI worker starting. GPU available: ", GPU)

function load_phantom(path)
    raw = reinterpret(Float32, read(path))
    n = Int(raw[1])
    col(i) = Float64.(raw[2 + (i - 1) * n : 1 + i * n])
    x, y, rho, t1, t2, t2s, dw = (col(i) for i in 1:7)
    return Phantom(name = "proaqmr", x = x, y = y, z = zeros(n), ρ = rho,
                   T1 = t1, T2 = t2, T2s = t2s, Δw = dw)
end

function run_sim(seq_path, phantom_path, out_path, b0)
    obj = load_phantom(phantom_path)
    seq = read_seq(seq_path)
    sys = Scanner()
    sys.B0 = b0
    sim_params = Dict{String,Any}("return_type" => "mat", "gpu" => GPU)
    sig = simulate(obj, seq, sys; sim_params = sim_params)
    s = ComplexF32.(vec(sig))
    out = Vector{Float32}(undef, 2 * length(s))
    out[1:2:end] .= real.(s)
    out[2:2:end] .= imag.(s)
    write(out_path, out)
    return length(s)
end

function handle(req::HTTP.Request)
    if req.target == "/health"
        return HTTP.Response(200, JSON3.write((ok = true, gpu = GPU)))
    end
    try
        b = JSON3.read(String(req.body))
        t = @elapsed n = run_sim(String(b.seq), String(b.phantom), String(b.out), Float64(b.b0))
        return HTTP.Response(200, JSON3.write((ok = true, n = n, seconds = t, gpu = GPU)))
    catch e
        msg = sprint(showerror, e)
        @error "simulation failed" exception = (e, catch_backtrace())
        return HTTP.Response(500, JSON3.write((ok = false, error = msg)))
    end
end

# Warm-up so the first real request does not pay the compile cost.
warm = get(ENV, "PROAQMR_WARMUP", "")
if !isempty(warm) && isfile(warm * ".seq")
    try
        println("Warming up KomaMRI (first compile can take a few minutes)...")
        run_sim(warm * ".seq", warm * ".phantom", warm * ".out", 1.5)
        println("Warm-up done.")
    catch e
        @warn "warm-up failed" exception = e
    end
end

const PORT = parse(Int, get(ENV, "KOMA_PORT", "8011"))
println("KomaMRI worker listening on 127.0.0.1:", PORT)
HTTP.serve(handle, "127.0.0.1", PORT)
