module Preprocessing
include("OllamaAI.jl")

##
# Imports
using TextAnalysis: Corpus, StringDocument, DocumentMetadata, NGramDocument,
                    standardize!, tokenize, Languages, update_lexicon!, lexical_frequency,
                    update_inverse_index!, inverse_index, stem!, remove_case!,
                    remove_corrupt_utf8!, remove_whitespace!, remove_nonletters!,
                    remove_punctuation!, remove_html_tags!, remove_numbers!,
                    remove_sparse_terms!, remove_frequent_terms!, remove_stop_words!,
                    remove_prepositions!, remove_articles!, remove_indefinite_articles!,
                    remove_definite_articles!, remove_pronouns!,
                    DocumentTermMatrix, lexicon, dtv

using Glob: glob
using JSON: parse
using HTTP: request
using Random: shuffle
using RegularExpressions

##
# Const
const PROJECT_ROOT = splitpath(Base.active_project())[end - 1]
const DATA_DIR = "data"
const MISTRAL_INSTRUCT_7B_TOKEN_CONTEXT = 32768 # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
const MISTRAL_INSTRUCT_7B_WORD_CONTEXT = round(
    Int64, MISTRAL_INSTRUCT_7B_TOKEN_CONTEXT * 0.75)
# https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct
# "<s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]"
const MISTRAL_INSTRUCT_SYSTEM_MESSAGE = """
    <s>[INST]
    Instruction: You are a world class NLP (Natural Language Processing) expert.
    Your job is to:
    i) extract the most important insights per document,
    ii) summarise the highlights
    and
    iii) retrieve the information the user asks for.
    Be precise.
    Say "I don't know" whenever you don't know the answer.
    [/INST]</s>"""
const DM = DocumentMetadata(
    Languages.English(),
    "NLP Demystified",
    "Nate Parker",
    "N/A"
)

##
"""
    word_and_token_count(vector::Vector{String})

Approximate count of the total number of tokens in a vector of strings.
1 token = 0.75 words per [OpenAI API documentation](https://platform.openai.com/docs/introduction)

# Arguments
- `vector::Vector{String}`: A vector of strings.

# Returns
- `Int64`: The total number of _tokens_ in the vector of strings
- `Int64`: The total number of _words_ in the vector of strings
"""
function word_and_token_count(vector::Vector{String})::Tuple{Int64, Int64}
    token_estimate::Float64 = 0
    total_words::Int64 = 0

    for text in vector
        words = split(text)
        total_words += length(words)
        token_estimate += total_words / 0.75
    end

    return round(Int64, token_estimate), total_words
end

"""
    word_and_token_count(text::String)
Approximately count the total number of tokens in a string of text.
1 token = 0.75 words per [OpenAI API documentation](https://platform.openai.com/docs/introduction)

# Arguments
- `text::String`: A string of text

# Returns
- `Int64`: The total number of tokens in the string.
"""

function word_and_token_count(text::String)::Int64
    token_estimate = length(split(text)) / 0.75

    return round(Int64, token_estimate)
end

##
"""
    segment_input(vector::Vector{String})
Segment text into `$(MISTRAL_INSTRUCT_7B_WORD_CONTEXT)\` word chunks.
Chunk length is calculated using the token = 0.75 word conversion,
according to [OpenAI API documentation](https://platform.openai.com/docs/introduction).

# Arguments
- `vector::Vector{String}`: A vector of strings

# Returns
- `d::Dict{Int64, String}`: Chunks of text divided into
"""
function segment_input(vector::Vector{String})::Dict{Int64, String}
    d = Dict{Int64, String}()
    i = 1
    chunk = ""
    for text in vector
        chunk *= text * " "
        if word_and_token_count(chunk) >= (MISTRAL_INSTRUCT_7B_TOKEN_CONTEXT - 10)
            d[i] = chunk
            chunk = ""
            i += 1
        end
    end

    if !isempty(chunk)
        d[i] = chunk
    end

    return d
end

##
"""
    summarise_text(chunks::Dict{Int64, String})::Vector{String}

Summarise text using the Mistral 7b LLM as locally served by Ollama

# Arguments
- `chunks::Dict{Int64, String})`:

# Returns
- `::Vector{String}`:
"""
function summarise_text(chunks::Dict{Int64, String})::Vector{String}
    url = "http://localhost:11434/api/generate"
    summaries = Vector{String}()

    for (_, v) in chunks
        prompt = MISTRAL_INSTRUCT_SYSTEM_MESSAGE
        prompt *= "Context: $v"
        prompt *= """\n[INST]Summarise the most important knowledge in the transcript above.
            Only return the summary, wrapped in single quotes (' '), and nothing else.
            [/INST]"""
        req = OllamaAI.send_request(prompt)
        res = request("POST", url, [("Content-type", "application/json")], req)
        if res.status == 200
            body = parse(String(res.body))
            push!(summaries, body["response"])
        else
            error("LLM returned status $(res.status)")
        end
    end

    return summaries
end

##
"""
    load_and_standardise_transcript_corpus()::Vector{Dict{Int64, String}}

Load all transcripts and [standardise](https://juliatext.github.io/TextAnalysis.jl/stable/corpus/)
them into a `TextAnalysis.Corpus`

# Arguments
nothing

# Returns
- `Corpus{StringDocument{String}}`: A corpus of `StringDocument`s, that contains all the transcripts in
`$(DATA_DIR)/`, segmented to fit the model's `$(MISTRAL_INSTRUCT_7B_TOKEN_CONTEXT)` token context
- `classes::Vector{String}`: Vector of classes
"""
function load_and_standardise_transcript_corpus()::Tuple{
        Corpus{StringDocument{String}}, Vector{String}}
    files::Vector{String} = glob(DATA_DIR * "/*.text")
    vec_docs = Vector{StringDocument{String}}()

    for file in files
        num_str::RegexMatch = match(r"\d{1,2}", file)
        num::Int64 = Base.parse(Int64, num_str.match)
        doc::String = open(file) |> readchomp
        _, no_words::Int64 = word_and_token_count([doc])
        println("Segmenting $(num_str.match) transcript ($no_words words)...")
        push!(vec_docs, StringDocument(doc, DM))
    end

    # TODO: Populate dynamically, based on filenames
    classes::Vector{String} = ["Intro",
        "Tokenization",
        "Basic preproc",
        "Advanced preproc",
        "BOW",
        "TF-IDF",
        "Models",
        "Classification",
        "Latent Dirichlet",
        "Neural Nets",
        "Training",
        "Embeddings",
        "RNNs",
        "Seq2Seq",
        "Transformers"]

    return Corpus(vec_docs), classes
end

##
"""
    preprocess_corpus(crps::Corpus{StringDocument{String}}; n_gram_docs::Bool=false)::Corpus{StringDocument{String}}

Preprocessing corpus' documents

# Arguments
- `corpus::Corpus{StringDocument{String}}`: A corpus of `StringDocument`s, that contains all the transcripts in
`$(DATA_DIR)/`, segmented to fit the model's `$(MISTRAL_INSTRUCT_7B_TOKEN_CONTEXT)` token context

# Keywords
- `freq_terms::Bool = false`: Flag to remove frequent terms in each document
- `n_gram_docs::Bool = false`: Flag to standardise corpus as `NGramDocument`s
- `articles::Bool = false`: Flag to remove definite & indefinite articles, prepositions and pronouns

# Returns
- `crps::Corpus`: A `StringDocument`s corpus, with the following transformations applied to every document:
* Stem document
* Remove
    > punctuation
    > whitespace
    > non letters
    > HTML tags
    > numbers
if `freq_terms` is true, remove frequent terms from corpus
if `n_gram_docs` is true, standardise the corpus into `NGramDocument`s
if `articles` is true, remove definite and indefinite articles, prepositions, pronouns and stopwords
"""
function preprocess_corpus(
        corpus::Corpus; freq_terms::Bool = false, n_gram_docs::Bool = false,
        articles::Bool = false)::Corpus
    remove_corrupt_utf8!(corpus)
    remove_case!(corpus)
    remove_punctuation!(corpus)
    remove_whitespace!(corpus)
    remove_nonletters!(corpus)
    remove_html_tags!(corpus)
    remove_numbers!(corpus)

    # FIXME: Conditionals are a bit arbitrary. Find a better way
    if freq_terms
        remove_sparse_terms!(corpus)
        remove_frequent_terms!(corpus)
        remove_stop_words!(corpus)
    end

    # FIXME: Conditionals are a bit arbitrary. Find a better way
    if articles
        remove_indefinite_articles!(corpus)
        remove_definite_articles!(corpus)
        remove_pronouns!(corpus)
        remove_prepositions!(corpus)
    end

    stem!(corpus)
    update_lexicon!(corpus)
    update_inverse_index!(corpus)

    if n_gram_docs
        standardize!(corpus, NGramDocument)
    end

    return corpus
end

##
"""
    train_test_valid_split(corpus::Corpus)::Tuple{Corpus, Corpus, Corpus}

Split corpus into train, test and validate chunks.

# Arguments
- `corpus::Corpus{StringDocument{String}}`: A corpus of `StringDocument`s, that contains all the transcripts in
`$(DATA_DIR)/`, segmented to fit the model's `$(MISTRAL_INSTRUCT_7B_TOKEN_CONTEXT)` token context

# Keywords
- `train_size::Float64 = 0.7`: Training set size %
- `val_size::Float64 = 0.1`: Validation set size %. Test set size will be 1 - (train_size - val_size)

# Returns
- `train_corpus::Corpus`: Training corpus
- `val_corpus::Corpus`: Validation corpus
- `test_corpus::Corpus`: Testing corpus
"""
function train_test_valid_split(corpus::Corpus; train_size::Float64 = 0.7,
        val_size::Float64 = 0.1)::Tuple{
        Vector{StringDocument{String}}, Vector{StringDocument{String}},
        Vector{StringDocument{String}}}
    n_docs::Int64 = length(corpus)
    test_size::Float64 = 1 - (train_size + val_size)

    train_pivot = Int(floor(train_size * n_docs))
    val_pivot = Int(floor((train_size + val_size) * n_docs))

    shuffled_idx::Vector{Int64} = shuffle(1:n_docs)
    train_idx = shuffled_idx[1:train_pivot]
    val_idx = shuffled_idx[(train_pivot + 1):val_pivot]
    test_idx = shuffled_idx[(val_pivot + 1):end]

    train_corpus::Vector{StringDocument{String}} = corpus[train_idx]
    val_corpus::Vector{StringDocument{String}} = corpus[val_idx]
    test_corpus::Vector{StringDocument{String}} = corpus[test_idx]

    return train_corpus, val_corpus, test_corpus
end

end
