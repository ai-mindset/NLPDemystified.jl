module NLPDemystified
include("Preprocessing.jl")
include("Vectorization.jl")

##
# Imports
using TextAnalysis: Corpus, StringDocument
using Plots: scatter!

##
# Load corpus
crps::Corpus{StringDocument{String}} = Preprocessing.load_and_standardise_transcript_corpus()
crps = Preprocessing.preprocess_corpus(crps)

##
crps_terms, sorted_crps_terms_desc, clusters = Vectorization.term_frequencies(crps)

end # module NLPDemystified
