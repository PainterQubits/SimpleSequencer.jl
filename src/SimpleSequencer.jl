module SimpleSequencer

import InstrumentControl: source
using InstrumentControl
using InstrumentControl.AWG5014C

const DEF_READ_DLY = 125e-6
const DEF_IF = 100e6

@inline env_cos(t, a, d) = a*(1 + cos(2π*(t - d/2)/d))/2
@inline if_signal(t, frequency, phase) = exp(-im*(2π*frequency*t + phase))

abstract Pulse

immutable Delay <: Pulse
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

@inline duration(x::Pulse) = x.duration
phase(x::Pulse) = :phase in fieldnames(x) ? x.phase : 0.0

function sequence(sample_rate, IF, v::Vector{Pulse}; marker_pts = 100)
    total_time = mapreduce(duration, +, 0.0, v)
    npts = Int(ceil(sample_rate * total_time)) + 1
    seq = SharedArray(Complex{Float64}, npts)
    markers = SharedArray(Bool, npts)

    @sync @parallel for i=1:npts
        seq[i] = 0.0+0.0im
        markers[i] = false
    end

    idx, dur = 0, 0.0
    for p in v
        rng = dur:(1/sample_rate):(dur+duration(p))
        if !isa(p, Delay)
            @sync begin @parallel for i in 1:length(rng)
                    seq[idx+i] = p(rng[i]-dur)*if_signal(rng[i], IF, phase(p))
                end
                @parallel for i in 1:marker_pts
                    markers[idx+i] = true
                end
            end
        end
        idx += length(rng)
        dur += duration(p)
    end

    (seq, markers)
end

function prepare(awg::InsAWG5014C)
    zeros = fill(0.0,250)
    falses = fill(false, 250)
    data = AWG5014CData(zeros, falses, falses)
    pushto_awg(awg, "simple_seq_XYI", data, :RealWaveform)
    pushto_awg(awg, "simple_seq_XYQ", data, :RealWaveform)
    pushto_awg(awg, "simple_seq_RI", data, :RealWaveform)
    pushto_awg(awg, "simple_seq_RQ", data, :RealWaveform)

    awg[RunMode] = :Sequence
    awg[SequenceLength] = 0
    awg[SequenceLength] = 1
    awg[SequenceWaveform,1,1] = "simple_seq_XYQ"
    awg[SequenceWaveform,1,2] = "simple_seq_XYI"
    awg[SequenceWaveform,1,3] = "simple_seq_RQ"
    awg[SequenceWaveform,1,4] = "simple_seq_RI"

    awg[SequenceGOTOTarget,1] = 1
    awg[SequenceGOTOState,1] = true
    awg[SequenceWaitTrigger,1] = true

    @allch awg[ChannelOutput] = true
    awg[Output] = true
    nothing
end

function sendpulses(awg::InsAWG5014C, xys, xym, rs, rm)
    pushto_awg(awg, "simple_seq_XYI", AWG5014CData(real(xys), xym, xym), :RealWaveform)
    pushto_awg(awg, "simple_seq_XYQ", AWG5014CData(imag(xys), xym, xym), :RealWaveform)
    pushto_awg(awg, "simple_seq_RI", AWG5014CData(real(rs), rm, rm), :RealWaveform)
    pushto_awg(awg, "simple_seq_RQ", AWG5014CData(imag(rs), rm, rm), :RealWaveform)
    @allch awg[ChannelOutput] = true
    awg[Output] = true
    nothing
end

type T1 <: Stimulus
    awg::InsAWG5014C
    Xpi::Pulse
    readout::Pulse
    IF::Float64
    finaldelay1::Float64
    finaldelay2::Float64
end
T1(awg, Xpi, readout; IF=DEF_IF,
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    T1(awg, Xpi, readout, IF, finaldelay1, finaldelay2)
function source(x::T1, t)
    awg = x.awg
    rate = awg[SampleRate]
    xys, xym = sequence(rate, x.IF,
        [x.Xpi,
         Delay(t),
         Delay(duration(x.readout)),
         Delay(x.finaldelay1),
         Delay(duration(x.readout)),
         Delay(x.finaldelay2)],
        marker_pts = 0)
    rs, rm = sequence(rate, x.IF,
        [Delay(duration(x.Xpi)),
         Delay(t),
         x.readout,
         Delay(x.finaldelay1),
         x.readout,
         Delay(x.finaldelay2)])
    sendpulses(awg, xys, xym, rs, rm)
end

type Rabi <: Stimulus
    awg::InsAWG5014C
    X::Pulse
    readout::Pulse
    IF::Float64
    finaldelay1::Float64
    finaldelay2::Float64
end
Rabi(awg, X, readout; IF=DEF_IF,
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    Rabi(awg, X, readout, IF, finaldelay1, finaldelay2)
function source(x::Rabi, t)
    awg = x.awg
    rate = awg[SampleRate]
    x.X.duration = t
    xys, xym = sequence(rate, x.IF,
        [x.X,
         Delay(duration(x.readout)),
         Delay(x.finaldelay1),
         Delay(duration(x.readout)),
         Delay(x.finaldelay2)],
        marker_pts = 0)
    rs, rm = sequence(rate, x.IF,
        [Delay(duration(x.X)),
         x.readout,
         Delay(x.finaldelay1),
         x.readout,
         Delay(x.finaldelay2)])
    sendpulses(awg, xys, xym, rs, rm)
end

type Ramsey <: Stimulus
    awg::InsAWG5014C
    Xpi2::Pulse
    readout::Pulse
    IF::Float64
    finaldelay1::Float64
    finaldelay2::Float64
end
Ramsey(awg, Xpi2, readout; IF=DEF_IF,
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    Ramsey(awg, Xpi2, readout, IF, finaldelay1, finaldelay2)
function source(x::Ramsey, t)
    awg = x.awg
    rate = awg[SampleRate]

    xys, xym = sequence(rate, x.IF,
        [x.Xpi2,
         Delay(t),
         x.Xpi2,
         Delay(duration(x.readout)),
         Delay(x.finaldelay1),
         Delay(duration(x.readout)),
         Delay(x.finaldelay2)],
        marker_pts = 0)
    rs, rm = sequence(rate, x.IF,
        [Delay(duration(x.Xpi2)),
         Delay(t),
         Delay(duration(x.Xpi2)),
         x.readout,
         Delay(x.finaldelay1),
         x.readout,
         Delay(x.finaldelay2)])
    sendpulses(awg, xys, xym, rs, rm)
end

type CPMG <: Stimulus
    awg::InsAWG5014C
    Xpi2::Pulse
    Ypi::Pulse
    mYpi::Pulse
    readout::Pulse
    IF::Float64
    finaldelay1::Float64
    finaldelay2::Float64
end
CPMG(awg, Xpi2, Ypi, mYpi, readout; IF=DEF_IF,
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    CPMG(awg, Xpi2, Ypi, mYpi, readout, IF, finaldelay1, finaldelay2)
function source(x::CPMG, v)
    nY, t_precess = v
    awg = x.awg
    rate = awg[SampleRate]
    t = t_precess / (2*nY)

    xys, xym = sequence(rate, x.IF,
        [x.Xpi2,
         Delay(t),
         x.Ypi,
         take(cycle([Delay(2*t), x.mYpi, Delay(2*t), x.Ypi]), 2*(nY-1))...,
         Delay(t),
         x.Xpi2,
         Delay(duration(x.readout)),
         Delay(x.finaldelay1),
         Delay(duration(x.readout)),
         Delay(x.finaldelay2)],
        marker_pts = 0)
    rs, rm = sequence(rate, x.IF,
        [Delay(duration(x.Xpi2)),
         Delay(t),
         Delay(duration(x.Ypi)),
         take(cycle([Delay(2*t),
                     Delay(duration(x.mYpi)),
                     Delay(2*t),
                     Delay(duration(x.Ypi))]), 2*(nY-1))...,
         Delay(t),
         Delay(duration(x.Xpi2)),
         x.readout,
         Delay(x.finaldelay1),
         x.readout,
         Delay(x.finaldelay2)])
    sendpulses(awg, xys, xym, rs, rm)
end

# For convenience
type CPMG_n <: Stimulus
    s::CPMG
    t::Float64
end
source(s::CPMG_n, nY) = source(s.s, (nY, s.t))

type CPMG_t <: Stimulus
    s::CPMG
    nY::Int
end
source(s::CPMG_t, t) = source(s.s, (s.nY, t))

end # module
