module NLPDemystified
include("Preprocessing.jl")
include("Vectorization.jl")

##
# Imports
using TextAnalysis: Corpus, StringDocument
using Plots: scatter!
using Random: seed!

##
# Load corpus
corpus::Corpus, classes::Vector{String} = Preprocessing.load_and_standardise_transcript_corpus()
corpus = Preprocessing.preprocess_corpus(corpus)

##
corpus_terms_bm25, sorted_corpus_terms_desc_bm25, clusters = Vectorization.bm25_term_frequencies(corpus)

##
# Split corpus to train / test / validate
seed!(123)
train::Vector{StringDocument{String}}, val::Vector{StringDocument{String}}, test::Vector{StringDocument{String}} = Preprocessing.train_test_valid_split(corpus)

end # module NLPDemystified
