module NLPDemystified
include("Preprocessing.jl")

#
using TextAnalysis: Corpus, StringDocument
# See [ElixirConf 2023 - Andrés Alejos - Nx Powered Decision Trees](https://youtu.be/rbmviKT6HkU)
# Regex + TF - IDF + XGBoost + make graphs?

# Load corpus
crps::Corpus{StringDocument{String}} = Preprocessing.load_and_standardise_transcript_corpus()
crps = Preprocessing.preprocess(crps, false)

end # module NLPDemystified
