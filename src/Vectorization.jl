module Vectorization
using TextAnalysis: bm_25, Corpus, StringDocument, update_lexicon!, DocumentTermMatrix
using SparseArrays: SparseMatrixCSC
using Clustering: dbscan, DbscanResult
using Statistics: mean
using Plots: scatter

##
"""
    bm25_term_frequencies(d::Dict{Int64, String})

# Arguments
- `crps::Corpus{StringDocument{String}}`: A corpus of `StringDocument`s, as generated in `Preprocessing.preprocess()`

# Returns
- `m.terms::Vector{String}`: Corpus terms
- `terms::Vector{String}`: Corpus terms sorted in descending frequencies
- `clusters::KmeansResult{Matrix{Float64}, Float64, Int64}}`: Clusters found in the corpus' BM25 scores
when k = 5
"""
function bm25_term_frequencies(corpus::Corpus{StringDocument{String}})::Tuple{
        Vector{String}, Vector{String}, DbscanResult}
    # BM25 takes into account word length per document
    # BM25 = IDF(query) * ( (TF(query, Document) * (κ + 1) / (TF(query, Document) * κ * (1 - β + β * θ))),
    # where θ = abs(Document) / mean(allDocuments), κ = free parameter typically ∈ [1.2, 2], β = free parameters typically 0.75
    update_lexicon!(corpus)
    m::DocumentTermMatrix = DocumentTermMatrix(corpus)
    bm25_scores::SparseMatrixCSC{Float64, Int64} = bm_25(m.dtm)
    # Each row = a word vector
    word_vectors = Matrix(bm25_scores)
    clusters::DbscanResult = dbscan(
        word_vectors, 0.05, min_neighbors = 3, min_cluster_size = 10)

    # Summing up the BM25 scores for each term across all documents
    term_freq::Matrix{Float64} = sum(m.dtm .* bm25_scores, dims = 1)
    # Convert term frequencies to a dense vector for easier manipulation
    term_freq_dense::Vector{Float64} = vec(term_freq)
    # Get indices that would sort the term frequencies in descending order
    sorted_terms_desc_freq::Vector{Int64} = sortperm(term_freq_dense, rev = true)

    sorted_terms_desc::Vector{String} = m.terms[sorted_terms_desc_freq]

    return m.terms, sorted_terms_desc, clusters
end

end
