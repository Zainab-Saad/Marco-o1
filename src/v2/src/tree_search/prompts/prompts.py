# systems prompts and user prompts for all datasets
# TODO: right not all these prompts are for chain of thought only do these prompts for exploration as well.

# system prompts
SYSTEM_COT_PROMPT = (
    "You are a careful math problem solver. "
    "Solve problems using a clear step by step reasoning process. "
    "Follow the requested final answer format exactly."
)

# user prompts for COT
GSM8K_COT_PROMPT = (
    "Solve the following math problem step by step. Show all your reasoning clearly. At the end, write your final numerical answer after '####'.\n\n"
    "Problem: {question}\n\n"
    "Solution:"
)

AIME_COT_PROMPT = (
    "Solve the following AIME problem step by step. Show all your reasoning clearly. "
    "Your final answer must be an integer between 000 and 999. "
    "Write it after '####' at the end.\n\n"
    "Problem: {question}\n\n"
    "Solution:"
)

MATH_COT_PROMPT = (
    "Solve the following math problem step by step. Show all your reasoning clearly. "
    "At the end, write your final answer inside \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
    "Solution:"
)
