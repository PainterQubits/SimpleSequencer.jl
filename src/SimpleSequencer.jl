module SimpleSequencer

import InstrumentControl: source
using InstrumentControl
using InstrumentControl.AWG5014C
# using Unitful: ns,μs,ms,s, GHz,MHz,kHz,Hz

const DEF_READ_DLY = 125e-9
const DEF_IF = 100e6

@inline env_cos(t, a, d) = a*(1 + cos(2π*(t - d/2)/d))/2
@inline if_signal(t, frequency, phase) = exp(im*(2π*frequency*t + phase))

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
    step = 1/sample_rate

    total_time = mapreduce(duration, +, 0, v)
    npts = length(0:step:total_time)
    seq = SharedArray(Complex{Float64}, npts)
    markers = SharedArray(Bool, npts)

    @sync @parallel for i=1:npts
        seq[i] = 0.0+0.0im
        markers[i] = false
    end

    dur, idx = 0, 0
    for p in v
        sta = rem(dur, step) == 0 ? dur : (dur + step - rem(dur,step))
        rng = sta:step:(dur+duration(p))
        if !isa(p, Delay)
            @sync begin @parallel for i in 1:length(rng)
                    seq[idx+i] = p(rng[i]-dur)*if_signal(rng[i], IF, phase(p))
                end
                @parallel for i in 1:marker_pts
                    markers[idx+i] = true
                end
            end
        end
        idx += length(rng) - 1
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
    axisname::Symbol
    axislabel::String
end
T1(awg, Xpi, readout; IF=DEF_IF, axisname = :t1delay, axislabel = "Delay",
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    T1(awg, Xpi, readout, IF, finaldelay1, finaldelay2, axisname, axislabel)
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
    axisname::Symbol
    axislabel::String
end
Rabi(awg, X, readout; IF=DEF_IF, axisname=:xyduration, axislabel="XY pulse duration",
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    Rabi(awg, X, readout, IF, finaldelay1, finaldelay2, axisname, axislabel)
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
    axisname::Symbol
    axislabel::String
end
Ramsey(awg, Xpi2, readout; IF=DEF_IF, axisname=:ramseydelay,
    axislabel="Free precession time",
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    Ramsey(awg, Xpi2, readout, IF, finaldelay1, finaldelay2, axisname, axislabel)
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
    nY::Int
    t_precess::Float64
    axisname::Symbol
    axislabel::String
end
CPMG(awg, Xpi2, Ypi, mYpi, readout; IF=DEF_IF, nY=1, t_precess=1000,
    axisname=:n_and_tp,
    axislabel=:"(# of Y pulses, total free precession time)",
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    CPMG(awg, Xpi2, Ypi, mYpi, readout, IF, finaldelay1, finaldelay2,
        nY, t_precess, axisname, axislabel)
function source(x::CPMG, v)
    nY, t_precess = v
    x.nY = nY
    x.t_precess = t_precess
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
    cpmg::CPMG
    axisname::Symbol
    axislabel::String
end
CPMG_n(s::CPMG; axisname=:n_Ypulses, axislabel="# of Y pulses") =
    CPMG_n(s, axisname, axislabel)
source(s::CPMG_n, nY) = source(s.cpmg, (nY, s.cpmg.t_precess))

type CPMG_t <: Stimulus
    cpmg::CPMG
    axisname::Symbol
    axislabel::String
end
CPMG_t(s::CPMG; axisname=:precessiontime, axislabel="Total free precession time") =
    CPMG_t(s, axisname, axislabel)
source(s::CPMG_t, t) = source(s.cpmg, (s.cpmg.nY, t))

type PulseVariation <: Stimulus
    p::Pulse
    f::Symbol
    rabi::Rabi
    axisname::Symbol
    axislabel::String
end
PulseVariation(p, f, rabi; axisname=f, axislabel=string(f)) =
    PulseVariation(p, f, rabi, axisname, axislabel)

function source(s::PulseVariation, v)
    setfield!(p, f, v)
    source(s.rabi, 0.0)
end


end # module
