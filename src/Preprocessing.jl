module Preprocessing
include("OllamaAI.jl")

##
# Imports
using TextAnalysis: Corpus, StringDocument, DocumentMetadata, NGramDocument,
                    standardize!, tokenize, Languages, remove_corrupt_utf8!,
                    prepare!, update_lexicon!, lexical_frequency, update_inverse_index!,
                    inverse_index, remove_case!, stem!,
                    strip_whitespace, strip_punctuation, strip_articles,
                    strip_indefinite_articles, strip_definite_articles, strip_prepositions,
                    strip_pronouns, strip_stopwords, strip_numbers, strip_non_letters,
                    strip_sparse_terms, strip_frequent_terms, strip_html_tags

using Glob: glob
using JSON: parse
using HTTP: request
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

Approximately count the total number of tokens in a vector of strings.
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
"""
function load_and_standardise_transcript_corpus()::Corpus{StringDocument{String}}
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

    return Corpus(vec_docs)
end

##
"""
    preprocess(crps::Corpus{StringDocument{String}};
        n_gram_docs::Bool=false)::Corpus{StringDocument{String}}

Preprocessing corpus' documents

# Arguments
- `crps::Corpus{StringDocument{String}}`: A corpus of `StringDocument`s, that contains all the transcripts in
`$(DATA_DIR)/`, segmented to fit the model's `$(MISTRAL_INSTRUCT_7B_TOKEN_CONTEXT)` token context

# Keywords
- `n_gram_docs::Bool = false`: Flag to standardise corpus as `NGramDocument`s. Default: false

# Returns
- `crps::Corpus{StringDocument{String}}`: A transformed corpus of `StringDocument`s,
with the following transformations applied to every document:
* Stem document
* Remove
    > utf-8 characters
    > case
    > punctuation
    > whitespace
    > articles (indefinite and definite)
    > prepositions
    > pronouns
    > stopwords
    > numbers
    > non letters
    > sparse terms
    > frequent terms
    > HTML tags
    > update lexicon
    > update inverse index
`if` n_gram_docs is true, standardise the corpus into `NGramDocument`s
"""
function preprocess(crps::Corpus{StringDocument{String}};
        n_gram_docs::Bool = false)::Corpus{StringDocument{String}}
    remove_corrupt_utf8!(crps)
    remove_case!(crps)
    prepare!(crps,
        strip_punctuation | strip_whitespace | strip_indefinite_articles |
        strip_definite_articles | strip_prepositions |
        strip_pronouns | strip_stopwords | strip_numbers | strip_non_letters |
        strip_sparse_terms | strip_frequent_terms | strip_html_tags)
    stem!(crps)
    update_lexicon!(crps)
    update_inverse_index!(crps)
    if n_gram_docs
        standardize!(crps, NGramDocument)
    end

    return crps
end

##
function named_entity_recognition()
end

##
function parsing()
end

##
function bag_of_words()
end

##
function doc_similarity()
end

end
