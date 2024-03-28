module Preprocessing
include("OllamaAI.jl")

##
# Imports
using Tokenize
using Glob: glob
using JSON: parse
using HTTP: request

##
# Const
const PROJECT_ROOT = splitpath(Base.active_project())[end-1]
const DATA_DIR = "data"
const MISTRAL_INSTRUCT_7B_TOKEN_CONTEXT = 32768 # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
const MISTRAL_INSTRUCT_7B_WORD_CONTEXT = round(Int64, MISTRAL_INSTRUCT_7B_TOKEN_CONTEXT * 0.75)
# https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct
# "<s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]"
const MISTRAL_INSTRUCT_SYSTEM_MESSAGE = """
    <s>[INST]
    Instruction: You are a world class NLP (Natural Language Processing) expert. Your job is
    to extract the most important insights per document, summarise the highlights and retrieve
    any information the user asks for.
    Be precise. Say "I don't know" whenever you don't know the answer.
    [/INST]</s>"""

##
# Text Preprocessing:
# 1. Tokenize the text: Use the TextAnalysis package to tokenize the text.
# 2. Remove stopwords: Utilize the TextAnalysis package to remove stopwords.
# 3. Stemming: Implement stemming using the TextAnalysis package.

##
"""
    token_count(vector::Vector{String})

Approximately count the total number of tokens in a vector of strings.
1 token = 0.75 words per [OpenAI API documentation](https://platform.openai.com/docs/introduction)

# Arguments
- `vector::Vector{String}`: A vector of strings.

# Returns
- `Int64`: The total number of tokens in the vector of strings
- `Int64`: The total number of words in the vector of strings
"""
function word_and_token_count(vector::Vector{String})::Tuple{Int64,Int64}
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
    token_count(text::String)
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
- `Dict{Int64, String}`: Chunks of text divided into
"""
function segment_input(vector::Vector{String})
    d = Dict{Int64,String}()
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
    summarise_text(chunks::Dict{Int64,String})::Vector{String}

Summarise text using an Ollama LLM

# Arguments
- `chunks::Dict{Int64,String})`:

# Returns
- `::Vector{String}`:
"""
function summarise_text(chunks::Dict{Int64,String})::Vector{String}
    local url = "http://localhost:11434/api/generate"
    local summaries = Vector{String}()

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

"""
# FIXME: WIP, incomplete implementation
function load_corpora()::Vector{String}::Dict{Int64,String}
    files::Vector{String} = glob(DATA_DIR * "/*.text")
    for file in files
        text::Vector{String} = open(file) |> readlines |> lowercase
        _, no_words::Int64 = word_and_token_count(text)
        println("Original transcript contains $no_words words")
        d::Dict{Int64,String} = segment_input(text)
    end

end


# toks = collect(tokenize(t))


end
