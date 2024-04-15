module Vectorization
using TextAnalysis: bm_25, Corpus, StringDocument, update_lexicon!, DocumentTermMatrix
using SparseArrays: SparseMatrixCSC
using Clustering: dbscan, DbscanResult
using Statistics: mean
using Plots: scatter

##
"""
    term_frequencies(d::Dict{Int64, String})

# Arguments
- `crps::Corpus{StringDocument{String}}`: A corpus of `StringDocument`s, as generated in `Preprocessing.preprocess()`

# Returns
- `terms::Vector{String}`: Corpus terms sorted in descending order
- `clusters::KmeansResult{Matrix{Float64}, Float64, Int64}}`: Clusters found in the corpus' BM25 scores
when k = 5
"""
function term_frequencies(crps::Corpus{StringDocument{String}})::Tuple{
        Vector{String}, DbscanResult}
    # BM25 takes into account word length per document
    # BM25 = IDF(query) * ( (TF(query, Document) * (κ + 1) / (TF(query, Document) * κ * (1 - β + β * θ))),
    # where θ = abs(Document) / mean(allDocuments), κ = free parameter typically ∈ [1.2, 2], β = free parameters typically 0.75
    update_lexicon!(crps)
    m::DocumentTermMatrix = DocumentTermMatrix(crps)
    bm25_scores::SparseMatrixCSC{Float64, Int64} = bm_25(m.dtm)
    # Each row = a word vector
    word_vectors = Matrix(bm25_scores)
    clusters::DbscanResult = dbscan(
        word_vectors, 0.05, min_neighbors = 3, min_cluster_size = 10)

    # Summing up the BM25 scores for each term across all documents
    term_frequencies::Matrix{Float64} = sum(m.dtm .* bm25_scores, dims = 1)
    # Convert term frequencies to a dense vector for easier manipulation
    term_frequencies_dense::Vector{Float64} = vec(term_frequencies)
    # Get indices that would sort the term frequencies in descending order
    sorted_indices::Vector{Int64} = sortperm(term_frequencies_dense, rev = true)

    terms::Vector{String} = m.terms[sorted_indices]

    return terms, clusters
end

end
