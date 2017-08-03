using SimpleSequencer
using Base.Test

import SimpleSequencer: duration, phase

@testset "Pulses" begin
    d = Delay(10.0e-6)
    @test duration(d) == 10.0e-6
    @test phase(d) == 0.0
    @test d(0.0) == 0.0
    c = CosinePulse(1.0, 100e-9, 0.1)
    @test duration(c) == 100e-9
    @test phase(c) == 0.1
    @test c(50e-9) ≈ 1.0
    r = RectanglePulse(1.0, 500e-9, 0.2)
    @test duration(r) == 100e-9
    @test phase(r) == 0.2
    @test r(0.0) == 1.0
end
