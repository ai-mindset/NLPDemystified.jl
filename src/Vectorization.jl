module Vectorization
using TextAnalysis: bm_25, Corpus, StringDocument, update_lexicon!, DocumentTermMatrix
using SparseArrays: SparseMatrixCSC
using DataFrames: DataFrame



##
#  Document Vectorization:
# 1. Create a document-term matrix: Use the TextAnalysis package to create a document-term matrix.
# 2. Calculate TF-IDF values: Compute TF-IDF values for each term in the documents.


"""
    term_frequencies(d::Dict{Int64, String})

# Arguments
- `d::Dict{Int64, String}`: Dictionary containing dose rules per entry, in free text

# Returns

"""
function term_frequencies(d::Dict{Int64,String})::Vector{String}
    # BM25 takes into account word length per document
    # BM25 = IDF(query) * ( (TF(query, Document) * (κ + 1) / (TF(query, Document) * κ * (1 - β + β * θ))),
    # where θ = abs(Document) / mean(allDocuments), κ = free parameter typically ∈ [1.2, 2], β = free parameters typically 0.75
    corpus::Corpus = Corpus([StringDocument(v) for (_, v) in d])
    update_lexicon!(corpus)
    m::DocumentTermMatrix = DocumentTermMatrix(corpus)
    bm25_scores::SparseMatrixCSC{Float64,Int64} = bm_25(m.dtm)
    # Summing up the BM25 scores for each term across all documents
    term_frequencies::Matrix{Float64} = sum(m.dtm .* bm25_scores, dims=1)
    # Convert term frequencies to a dense vector for easier manipulation
    term_frequencies_dense::Vector{Float64} = vec(term_frequencies)
    # Get indices that would sort the term frequencies in descending order
    sorted_indices::Vector{Int64} = sortperm(term_frequencies_dense, rev=true)

    return m.terms[sorted_indices]
end



end
