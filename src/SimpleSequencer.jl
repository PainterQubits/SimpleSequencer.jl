# TODO: why does `SimpleSequencer.prepare` hang when first executed?
# TODO: how to pass marker_pts through when sourcing stimuli?
# TODO: PulseVariation stimulus not fully functional.
# TODO: separate readout and XY IF.

__precompile__(true)
module SimpleSequencer

import ICCommon: source, Stimulus
using InstrumentControl
using InstrumentControl.AWG5014C

const DEF_READ_DLY = 125e-6
const DEF_IF = 100e6

@inline env_cos(t, a, d) = a*(1 + cos(2π*(t - d/2)/d))/2
@inline if_signal(t, frequency, phase) = exp(im*(2π*frequency*t + phase))

"""
    abstract type Pulse end
A pulse that can be sequenced by [`SimpleSequencer.sequence`](@ref). Concrete
subtypes of `Pulse` represent different envelopes. All Pulses are callable
and when given time as an argument, it will return the corresponding value
for the pulse's envelope.
"""
abstract type Pulse end

"""
    mutable struct Delay <: Pulse
        duration::Float64
    end
Generate a delay for a pulse sequence.
"""
mutable struct Delay <: Pulse
    duration::Float64
end

(x::Delay)(t) = 0.0

"""
    mutable struct CosinePulse <: Pulse
        amplitude::Float64
        duration::Float64
        phase::Float64
        IF::Float64
    end
Generate cosine envelope with amplitude and duration. Keep track of a phase to
allow phase shifting of the upconverted tone.
"""
mutable struct CosinePulse <: Pulse
    amplitude::Float64
    duration::Float64
    phase::Float64
    IF::Float64
end
(x::CosinePulse)(t) = env_cos(t, x.amplitude, x.duration)

"""
    mutable struct RectanglePulse <: Pulse
        amplitude::Float64
        duration::Float64
        phase::Float64
        IF::Float64
    end
Generate rectangle envelope with amplitude and duration. Keep track of a phase to
allow phase shifting of the upconverted tone.
"""
mutable struct RectanglePulse <: Pulse
    amplitude::Float64
    duration::Float64
    phase::Float64
    IF::Float64
end
(x::RectanglePulse)(t) = x.amplitude

"""
    duration(x::Pulse)
Query a pulse's duration.
"""
@inline duration(x::Pulse) = x.duration

"""
    phase(x::Pulse)
Query a pulse's phase (delays return 0.0).
"""
phase(x::Pulse) = x.phase
phase(x::Delay) = 0.0

"""
    IF(x::Pulse)
Query a pulse's IF.
"""
IF(x::Pulse) = x.IF

"""
    sequence(sample_rate, total_time, v::Vector{Pulse};
        readout=false, marker_pts = 100)
Creates a `Tuple{Complex{Float64}, Bool}` representing the IQ data and marker data
given a pulse sequence. Used internally, not called by user directly.
"""
function sequence(sample_rate, total_time, v::Vector{Pulse}; readout=false, marker_pts = 100)
    step = 1/sample_rate

    # `rng` is the sole time-base used for sequence generation.
    # total_time = mapreduce(duration, +, 0, v)
    rng = 0.0:step:total_time
    npts = length(rng)

    # Preallocate arrays, leaving the option of using multiple worker procs
    seq = Vector{Complex{Float64}}(npts)
    markers = Vector{Bool}(npts)

    # zero out the arrays
    for i=1:npts
        seq[i] = 0.0+0.0im
        markers[i] = false
    end

    dur, idx = 0.0, 0
    for p in v
        # Find points in our time-base that are within the duration of pulse `p`
        prng = rng[find(x->dur <= x < dur+duration(p), rng)]

        # For the readout tone, we don't care if it is phase-locked to the
        # XY time base. In fact, we probably want readout tones to always have
        # the same absolute phase, and more importantly, for successive readout
        # tones to start at the same exact phase.
        prng2 = length(prng)==0 ? prng : prng - prng[1] #don't getindex if len. 0
        ifrng = readout ? prng2 : prng

        if !isa(p, Delay)
            for i in 1:length(prng)
                seq[idx+i] = p(prng[i]-dur)*if_signal(ifrng[i], IF(p), phase(p))
            end
            for i in 1:marker_pts
                markers[idx+i] = true
            end
        end
        idx += length(prng)
        dur += duration(p)
    end

    (seq, markers)
end

"""
    prepare(awg::InsAWG5014C)
Prepare the AWG for use by SimpleSequencer. This generates waveforms with the
necessary names, switches run mode to sequenced, and loops over the waveforms
(with wait trigger enabled). It also turns on all outputs.
"""
function prepare(awg::InsAWG5014C)
    zeros = fill(0.0,250)
    falses = fill(false, 250)
    data = AWG5014CData(zeros, falses, falses)
    pushto_awg(awg, "simple_seq_XYI", data, :RealWaveform)
    pushto_awg(awg, "simple_seq_XYQ", data, :RealWaveform)
    pushto_awg(awg, "simple_seq_RI", data, :RealWaveform)
    pushto_awg(awg, "simple_seq_RQ", data, :RealWaveform)
    opc(awg)

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
    opc(awg)

    awg[ChannelOutput,1] = true
    opc(awg)
    awg[ChannelOutput,2] = true
    opc(awg)
    awg[ChannelOutput,3] = true
    opc(awg)
    awg[ChannelOutput,4] = true
    opc(awg)

    awg[Output] = true
    opc(awg)

    nothing
end

"""
    sendpulses(awg::InsAWG5014C, xys, xym, rs, rm)
Given complex arrays `xys` and `rs` (xy sequence and readout sequence, respectively)
and boolean arrays `xym` and `rm` (xy markers and readout markers, respectively), this
updates the simple sequencer waveforms on the AWG and turns on the outputs. This
is not called explicitly by the user, but is instead called by `source` methods.
"""
function sendpulses(awg::InsAWG5014C, xys, xym, rs, rm)
    pushto_awg(awg, "simple_seq_XYI", AWG5014CData(real(xys), xym, xym), :RealWaveform)
    pushto_awg(awg, "simple_seq_XYQ", AWG5014CData(imag(xys), xym, xym), :RealWaveform)
    pushto_awg(awg, "simple_seq_RI", AWG5014CData(real(rs), rm, rm), :RealWaveform)
    pushto_awg(awg, "simple_seq_RQ", AWG5014CData(imag(rs), rm, rm), :RealWaveform)
    opc(awg)

    awg[ChannelOutput,1] = true
    opc(awg)
    awg[ChannelOutput,2] = true
    opc(awg)
    awg[ChannelOutput,3] = true
    opc(awg)
    awg[ChannelOutput,4] = true
    opc(awg)

    awg[Output] = true
    opc(awg)

    nothing
end

mutable struct T1 <: Stimulus
    awg::InsAWG5014C
    Xpi::Pulse
    readout::Pulse
    finaldelay1::Float64
    finaldelay2::Float64
    axisname::Symbol
    axislabel::String
end

"""
    T1(awg, Xpi, readout; axisname = :t1delay,
        axislabel = "Delay", finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY)
Creates a T1 stimulus object given an AWG, pi pulse, and readout pulse.
Sourcing with a `Float64` will set the delay between the end of the pi pulse and
the start of the readout pulse, sequencing a T1 measurement.
"""
T1(awg, Xpi, readout; axisname = :t1delay, axislabel = "Delay",
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    T1(awg, Xpi, readout, finaldelay1, finaldelay2, axisname, axislabel)
function source(x::T1, t)
    awg = x.awg
    rate = awg[SampleRate]
    total_time = duration(x.Xpi)+t+2*duration(x.readout)+x.finaldelay1+x.finaldelay2
    xys, xym = sequence(rate, total_time,
        [x.Xpi,
         Delay(t),
         Delay(duration(x.readout)),
         Delay(x.finaldelay1),
         Delay(duration(x.readout)),
         Delay(x.finaldelay2)],
        marker_pts = 0)
    rs, rm = sequence(rate, total_time,
        [Delay(duration(x.Xpi)),
         Delay(t),
         x.readout,
         Delay(x.finaldelay1),
         x.readout,
         Delay(x.finaldelay2)],
         readout = true)
    sendpulses(awg, xys, xym, rs, rm)
end

mutable struct Rabi <: Stimulus
    awg::InsAWG5014C
    X::Pulse
    readout::Pulse
    finaldelay1::Float64
    finaldelay2::Float64
    axisname::Symbol
    axislabel::String
end

"""
    Rabi(awg, X, readout; axisname=:xyduration, axislabel="XY pulse duration",
        finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY)
Creates a Rabi stimulus object given an AWG, XY pulse, and readout pulse.
Sourcing with a `Float64` will set the duration of the XY pulse and sequence
a Rabi flop measurement.
"""
Rabi(awg, X, readout; axisname=:xyduration, axislabel="XY pulse duration",
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    Rabi(awg, X, readout, finaldelay1, finaldelay2, axisname, axislabel)
function source(x::Rabi, t)
    awg = x.awg
    rate = awg[SampleRate]
    x.X.duration = t
    total_time = t + 2*duration(x.readout) + x.finaldelay1 + x.finaldelay2
    xys, xym = sequence(rate, total_time,
        [x.X,
         Delay(duration(x.readout)),
         Delay(x.finaldelay1),
         Delay(duration(x.readout)),
         Delay(x.finaldelay2)],
        marker_pts = 0)
    rs, rm = sequence(rate, total_time,
        [Delay(duration(x.X)),
         x.readout,
         Delay(x.finaldelay1),
         x.readout,
         Delay(x.finaldelay2)],
         readout = true)
    sendpulses(awg, xys, xym, rs, rm)
end

mutable struct Ramsey <: Stimulus
    awg::InsAWG5014C
    Xpi2::Pulse
    readout::Pulse
    finaldelay1::Float64
    finaldelay2::Float64
    axisname::Symbol
    axislabel::String
end

"""
    Ramsey(awg, X, readout; axisname=:xyduration, axislabel="XY pulse duration",
        finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY)
Creates a Ramsey stimulus object given an AWG, pi/2 pulse, and readout pulse.
Sourcing with a `Float64` will set the delay between pi/2 pulses and sequence
a Ramsey (T2) measurement.
"""
Ramsey(awg, Xpi2, readout; axisname=:ramseydelay,
    axislabel="Free precession time",
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    Ramsey(awg, Xpi2, readout, finaldelay1, finaldelay2, axisname, axislabel)
function source(x::Ramsey, t)
    awg = x.awg
    rate = awg[SampleRate]

    total_time = 2*duration(x.Xpi2)+t+2*duration(x.readout)+x.finaldelay1+x.finaldelay2
    xys, xym = sequence(rate, total_time,
        [x.Xpi2,
         Delay(t),
         x.Xpi2,
         Delay(duration(x.readout)),
         Delay(x.finaldelay1),
         Delay(duration(x.readout)),
         Delay(x.finaldelay2)],
        marker_pts = 0)
    rs, rm = sequence(rate, total_time,
        [Delay(duration(x.Xpi2)),
         Delay(t),
         Delay(duration(x.Xpi2)),
         x.readout,
         Delay(x.finaldelay1),
         x.readout,
         Delay(x.finaldelay2)],
         readout = true)
    sendpulses(awg, xys, xym, rs, rm)
end

mutable struct CPMG <: Stimulus
    awg::InsAWG5014C
    Xpi2::Pulse
    Ypi::Pulse
    mYpi::Pulse
    readout::Pulse
    finaldelay1::Float64
    finaldelay2::Float64
    nY::Int
    t_precess::Float64
    δt1::Float64
    δt2::Float64
    axisname::Symbol
    axislabel::String
end

"""
    CPMG(awg, Xpi2, Ypi, mYpi, readout; nY=1, t_precess=duration(Ypi),
        δt1 = 0, δt2 = 0, axisname=:n_and_tp,
        axislabel=:"(# of Y pulses, total free precession time)",
        finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY)
Creates a CPMG (Carr-Purcell-Meiboom-Gill) stimulus object given an AWG, X pi/2
pulse, Y pi pulse, -Y pi pulse, and readout pulse. Sourcing with a tuple `(n,t_p)`
will set the number of pi pulses `n` and the total idle time between the first
and last pi/2 pulse. This is hard to use in a sweep directly, instead suggest
sourcing a [`CPMG_n`](@ref) or [`CPMG_t`](@ref) object.
"""
CPMG(awg, Xpi2, Ypi, mYpi, readout; nY=1, t_precess=duration(Ypi),
    δt1 = 0, δt2 = 0, axisname=:n_and_tp,
    axislabel=:"(# of Y pulses, total free precession time)",
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    CPMG(awg, Xpi2, Ypi, mYpi, readout, finaldelay1, finaldelay2,
        nY, t_precess, δt1, δt2, axisname, axislabel)
function source(x::CPMG, v)
    nY, t_precess, δt1, δt2 = v
    @assert (t_precess - nY*duration(x.Ypi) >= 0)
    @assert duration(x.Ypi) == duration(x.mYpi)

    x.nY = nY
    x.t_precess = t_precess
    x.δt1 = δt1
    x.δt2 = δt2

    awg = x.awg
    rate = awg[SampleRate]
    t = (t_precess - nY*duration(x.Ypi)) / (2*nY)
    @assert t-δt1 >= 0
    @assert 2*t-δt2 >= 0

    s = [x.Xpi2,
     Delay(t-δt1),
     x.Ypi,
     Iterators.take(Iterators.cycle([Delay(2*t-δt2), x.mYpi, Delay(2*t+δt2), x.Ypi]), 2*(nY-1))...,
     Delay(t+δt1),
     x.Xpi2,
     Delay(duration(x.readout)),
     Delay(x.finaldelay1),
     Delay(duration(x.readout)),
     Delay(x.finaldelay2)]

    total_time = mapreduce(duration, +, 0.0, s)

    xys, xym = sequence(rate, total_time, s, marker_pts = 0)
    rs, rm = sequence(rate, total_time,
        [Delay(duration(x.Xpi2)),
         Delay(t-δt1),
         Delay(duration(x.Ypi)),
         Iterators.take(Iterators.cycle([Delay(2*t-δt2),
                     Delay(duration(x.mYpi)),
                     Delay(2*t+δt2),
                     Delay(duration(x.Ypi))]), 2*(nY-1))...,
         Delay(t+δt1),
         Delay(duration(x.Xpi2)),
         x.readout,
         Delay(x.finaldelay1),
         x.readout,
         Delay(x.finaldelay2)],
         readout = true)
    sendpulses(awg, xys, xym, rs, rm)
end

# For convenience
mutable struct CPMG_n <: Stimulus
    cpmg::CPMG
    axisname::Symbol
    axislabel::String
end

"""
    CPMG_n(s::CPMG; axisname=:n_Ypulses, axislabel="# of Y pulses")
Creates a `CPMG_n` stimulus object using a [`CPMG`](@ref) object. When sourced,
this will change the number of pi pulses.
"""
CPMG_n(s::CPMG; axisname=:n_Ypulses, axislabel="# of Y pulses") =
    CPMG_n(s, axisname, axislabel)
source(s::CPMG_n, nY) = source(s.cpmg, (nY, s.cpmg.t_precess, s.cpmg.δt1, s.cpmg.δt2))

mutable struct CPMG_t <: Stimulus
    cpmg::CPMG
    axisname::Symbol
    axislabel::String
end

"""
    CPMG_t(s::CPMG; axisname=:precessiontime, axislabel="Total free precession time")
Creates a `CPMG_t` stimulus object using a [`CPMG`](@ref) object. When sourced,
this will change the free precession time.
"""
CPMG_t(s::CPMG; axisname=:precessiontime, axislabel="Total free precession time") =
    CPMG_t(s, axisname, axislabel)
source(s::CPMG_t, t) = source(s.cpmg, (s.cpmg.nY, t, s.cpmg.δt1, s.cpmg.δt2))

mutable struct CPMG_δt1 <: Stimulus
    cpmg::CPMG
    axisname::Symbol
    axislabel::String
end

"""
    CPMG_δt1(s::CPMG; axisname=:δt1, axislabel="Time offset 1")
Creates a `CPMG_δt1` stimulus object using a [`CPMG`](@ref) object. When sourced,
this will change the first time offset.
"""
CPMG_δt1(s::CPMG; axisname=:δt1, axislabel="Time offset 1") =
    CPMG_δt1(s, axisname, axislabel)
source(s::CPMG_δt1, t) = source(s.cpmg, (s.cpmg.nY, s.cpmg.t_precess, t, s.cpmg.δt2))

mutable struct CPMG_δt2 <: Stimulus
    cpmg::CPMG
    axisname::Symbol
    axislabel::String
end

"""
    CPMG_δt2(s::CPMG; axisname=:δt2, axislabel="Time offset 2")
Creates a `CPMG_δt2` stimulus object using a [`CPMG`](@ref) object. When sourced,
this will change the second time offset.
"""
CPMG_δt2(s::CPMG; axisname=:δt2, axislabel="Time offset 2") =
    CPMG_δt2(s, axisname, axislabel)
source(s::CPMG_δt2, t) = source(s.cpmg, (s.cpmg.nY, s.cpmg.t_precess, s.cpmg.δt1, t))

mutable struct StarkShift <: Stimulus
    awg::InsAWG5014C
    Xpi::Pulse
    readout::Pulse
    finaldelay1::Float64
    finaldelay2::Float64
    axisname::Symbol
    axislabel::String
end

"""
    StarkShift(awg, Xpi, readout; axisname=:t_offset, axislabel=:"Offset time",
        finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY)
Creates a `StarkShift` stimulus object given an AWG, pi pulse, and readout pulse.
Sourcing with a time will set the starting time of the pi pulse with respect
to the starting time of the first readout tone. `source` will throw an `AssertionError`
if you source a time less than minus the pi pulse duration.
"""
StarkShift(awg, Xpi, readout; axisname=:t_offset, axislabel=:"Offset time",
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    StarkShift(awg, Xpi, readout, finaldelay1, finaldelay2,
        axisname, axislabel)
function source(x::StarkShift, t)
    awg = x.awg
    rate = awg[SampleRate]
    @assert t >= -duration(x.Xpi)

    s = [Delay(duration(x.Xpi)),
     x.readout,
     Delay(x.finaldelay1),
     x.readout,
     Delay(x.finaldelay2)]
    total_time = mapreduce(duration, +, 0.0, s)

    xys, xym = sequence(rate, total_time,
        [Delay(t+duration(x.Xpi)),
         x.Xpi,
         Delay((duration(x.Xpi)+2*duration(x.readout)+x.finaldelay1+x.finaldelay2) -
            (t+2*duration(x.Xpi)))],
        marker_pts = 0)
    rs, rm = sequence(rate, total_time, s, readout = true)
    sendpulses(awg, xys, xym, rs, rm)
end

mutable struct LongStarkShift <: Stimulus
    awg::InsAWG5014C
    Xpi::Pulse
    readout1::Pulse
    readout2::Pulse
    finaldelay1::Float64
    finaldelay2::Float64
    axisname::Symbol
    axislabel::String
end

"""
    LongStarkShift(awg, Xpi, readout1, readout2; axisname=:t_offset,
        axislabel=:"Offset time", finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY)
Creates a `LongStarkShift` stimulus object given an AWG, pi pulse, and two
distinct readout pulses. Sourcing with a time will set the starting time of the
pi pulse with respect to the starting time of the first readout tone.
`source` will throw an `AssertionError` if you source a time less than minus the
pi pulse duration.

The utility of this object is that you can keep your actual readout the same
(no tweaks to the digitizer parameters) while still investigating what happens
if you drive the resonator a long time.

Do take care that the resonator has rung down before doing the second readout pulse
(the readout pulses are not kept phase coherent).
"""
LongStarkShift(awg, Xpi, readout1, readout2;
    axisname=:t_offset,
    axislabel=:"Offset time",
    finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    LongStarkShift(awg, Xpi, readout1, readout2,
        finaldelay1, finaldelay2, axisname, axislabel)
function source(x::LongStarkShift, t)
    awg = x.awg
    rate = awg[SampleRate]
    @assert t >= -duration(x.Xpi)

    s = [Delay(duration(x.Xpi)),
     x.readout1,
     Delay(x.finaldelay1),
     x.readout2,
     Delay(x.finaldelay2)]
    total_time = mapreduce(duration, +, 0.0, s)

    xys, xym = sequence(rate, total_time,
        [Delay(t+duration(x.Xpi)),
         x.Xpi,
         Delay((duration(x.Xpi)+duration(x.readout1)+duration(x.readout2)+
            x.finaldelay1+x.finaldelay2) -(t+2*duration(x.Xpi)))],
        marker_pts = 0)
    rs, rm = sequence(rate, total_time, s, readout = true)
    sendpulses(awg, xys, xym, rs, rm)
end

mutable struct AnharmonicitySearch <: Stimulus
    awg::InsAWG5014C
    X01::Pulse
    X12::Pulse
    readout::Pulse
    finaldelay1::Float64
    finaldelay2::Float64
    axisname::Symbol
    axislabel::String
end

"""
    AnharmonicitySearch(awg, X01, X12, readout; axisname = :X12IF,
        axislabel = "X12 IF (Hz)", finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY)
Creates an AnharmonicitySearch stimulus object given an AWG, pulse for 0-1
transition, pulse for 1-2 transition, and readout pulse. This will do a 0-1 pulse,
immediately followed by a 1-2 pulse, immediately followed by readout.
Sourcing with a `Float64` will set the IF for the 1-2 transition pulse.
"""
AnharmonicitySearch(awg, X01, X12, readout; axisname = :X12IF,
    axislabel = "X12 IF (Hz)", finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    AnharmonicitySearch(awg, X01, X12, readout,
        finaldelay1, finaldelay2, axisname, axislabel)
function source(x::AnharmonicitySearch, IF)
    awg = x.awg
    rate = awg[SampleRate]
    x.X12.IF = IF
    total_time = duration(x.X01) + duration(x.X12) +
        2*duration(x.readout) + x.finaldelay1 + x.finaldelay2
    xys, xym = sequence(rate, total_time,
        [x.X01,
         x.X12,
         Delay(duration(x.readout)),
         Delay(x.finaldelay1),
         Delay(duration(x.readout)),
         Delay(x.finaldelay2)],
        marker_pts = 0)
    rs, rm = sequence(rate, total_time,
        [Delay(duration(x.X01)),
         Delay(duration(x.X12)),
         x.readout,
         Delay(x.finaldelay1),
         x.readout,
         Delay(x.finaldelay2)],
         readout = true)
    sendpulses(awg, xys, xym, rs, rm)
end

mutable struct T1_12 <: Stimulus
    awg::InsAWG5014C
    X01::Pulse
    X12::Pulse
    readout::Pulse
    finaldelay1::Float64
    finaldelay2::Float64
    axisname::Symbol
    axislabel::String
end
T1_12(awg, X01, X12, readout; axisname = :t1delay,
    axislabel = "Delay", finaldelay1=DEF_READ_DLY, finaldelay2=DEF_READ_DLY) =
    T1_12(awg, X01, X12, readout,
        finaldelay1, finaldelay2, axisname, axislabel)

function source(x::T1_12, t::Float64)
    awg = x.awg
    rate = awg[SampleRate]
    total_time = duration(x.X01) + duration(x.X12) + t +
        2*duration(x.readout) + x.finaldelay1 + x.finaldelay2
    xys, xym = sequence(rate, total_time,
        [x.X01,
         x.X12,
         Delay(t),
         Delay(duration(x.readout)),
         Delay(x.finaldelay1),
         Delay(duration(x.readout)),
         Delay(x.finaldelay2)],
        marker_pts = 0)
    rs, rm = sequence(rate, total_time,
        [Delay(duration(x.X01)),
         Delay(duration(x.X12)),
         Delay(t),
         x.readout,
         Delay(x.finaldelay1),
         x.readout,
         Delay(x.finaldelay2)],
         readout = true)
    sendpulses(awg, xys, xym, rs, rm)
end
# type PulseVariation <: Stimulus
#     p::Pulse
#     f::Symbol
#     rabi::Rabi
#     axisname::Symbol
#     axislabel::String
# end
# PulseVariation(p, f, rabi; axisname=f, axislabel=string(f)) =
#     PulseVariation(p, f, rabi, axisname, axislabel)
#
# function source(s::PulseVariation, v)
#     setfield!(p, f, v)
#     source(s.rabi, 0.0)
# end


end # module
