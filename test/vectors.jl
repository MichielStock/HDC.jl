@testset "Vectors" begin
    N = 1000
    using SparseArrays, LinearAlgebra
    
    @testset "BitHDV" begin
        # Binary
        v = hdv(N)
        @test v isa BinaryHDV
        @test length(v) == N
        @test sum(v) < N
        @test eltype(v) == Bool

        # Bipolar
        v = bphdv(N)
        @test v isa BipolarHDV
        @test eltype(v) <: Integer
        @test all(∈((-1,1)), v)
        @test -N/2 < sum(v) < N/2  # usually...
    end

    @testset "SparseHDV" begin
        v = sphdv(p=0.2)
        @test v isa SparseHDV
        @test v.v isa SparseVector
        @test eltype(v) == Bool
        @test sum(v) isa Int
    end

    @testset "DenseHDV" begin
        # real
        v = realhdv(N)
        @test eltype(v) == Float64
        @test -N/2 < sum(v) < N/2 
        @test eltype(realhdv(N, Float32)) == Float32
        @test norm(v) > 0

        # graded
        v = gradhdv(N)
        @test eltype(v) == Float64
        @test all(e->0≤e≤1, v)

        # graded bipolar
        v = gradbphdv(N)
        @test eltype(v) == Float64
        @test all(e->-1≤e≤1, v)
    end

end