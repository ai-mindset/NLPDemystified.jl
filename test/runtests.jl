using NLPDemystified 
using Test

@testset "Test app" begin

    @testset "Tokenizer tests" begin
        include("tokenizer_tests.jl")
    end

    @testset "Summarise tests" begin
        include("summarise_tests.jl")
    end
end
