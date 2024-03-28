module OllamaAI

using JSON: json

"""
    send_request(prompt::String, model::String)

Send request to Ollama LLM. `mistral` 7b instruct is hardcoded due to its excellent performance

# Arguments
- `prompt::String`: Prompt incl. context to be sent to Ollama LLM

# Credit
@rcherukuri12 for the [solution](https://discourse.julialang.org/t/using-julia-to-connect-to-local-llms/106137)
"""
function send_request(prompt::String)::String
    println("Sending request to LLM...")
    data = Dict() # declare empty to support Any type.
    data["model"] = "mistral"
    data["prompt"] = prompt
    data["stream"] = false  # turning off streaming response.
    data["temperature"] = 0.0

    return json(data)
end

end
