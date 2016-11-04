module SimpleSequencer

export Delay, CosinePulse, RectanglePulse, sequence

@inline env_cos(t, a, d) = a*(1 + cos(2π*(t - d/2)/d))/2
@inline if_signal(t, frequency, phase) = exp(im*(2π*frequency*t + phase))

abstract Pulse

type Delay <: Pulse
    duration::Float64
end
(x::Delay)(t) = 0.0

type CosinePulse <: Pulse
    amplitude::Float64
    duration::Float64
    phase::Float64
end
(x::CosinePulse)(t) = env_cos(t, x.amplitude, x.duration)

type RectanglePulse <: Pulse
    amplitude::Float64
    duration::Float64
    phase::Float64
end
(x::RectanglePulse)(t) = x.amplitude

duration(x::Pulse) = x.duration
phase(x::Pulse) = :phase in fieldnames(x) ? x.phase : 0.0

function sequence(sample_rate, IF, v::Vector{Pulse}; marker_pts = 10)
    total_time = mapreduce(duration, +, 0.0, v)
    npts = Int(ceil(sample_rate * total_time))
    seq = SharedArray(Complex{Float64}, npts)
    markers = SharedArray(Bool, npts)

    @sync @parallel for i=1:npts
        markers[i] = false
    end

    idx, dur = 0, 0.0
    for p in v
        rng = dur:(1/sample_rate):(dur+duration(p))
        @sync @parallel for i in 1:length(rng)
            seq[idx+i] = p(rng[i]-dur)*if_signal(rng[i], IF, phase(p))
        end
        if !isa(p, Delay)
            @sync @parallel for i in 1:marker_pts
                markers[idx+i] = true
            end
        end
        idx += length(rng)
        dur += duration(p)
    end

    (seq, markers)
end

end # module

# Xpi2 = CosineEnvelope(0.5, 230e-9, 0)
# Ypi2 = CosineEnvelope(0.5, 230e-9, pi/2)
# Xpi = CosineEnvelope(1.0, 230e-9, 0)
# Ypi = CosineEnvelope(1.0, 230e-9, pi/2)
# readout = RectanglePulse(1.0, 500e-9, 0)
# XYpulses = [Xpi, Delay(t), Delay(duration(readout))]
# Rpulses = [Delay(duration(Xpi)), Delay(t), readout]
