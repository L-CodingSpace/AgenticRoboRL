"""The RL behaviour policy: plain temperature-1 sampling of the model's own distribution.

rl/proxy.py writes these values into every chat request (overriding the planner's), so the
logprobs vLLM returns (logprobs_mode raw_logprobs, its default) are the sampling distribution.
"""
RL_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,  # vLLM: 0 disables top-k
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
}
