from main import graph

policy = graph.invoke({"query": "How much is delivery below INR 149?"})
general = graph.invoke({"query": "Tell me a joke."})

assert policy["intent"] == "policy_question"
assert policy["sources"]
assert policy["answer"].startswith("Based on the retrieved context:")
assert general["intent"] == "general_question"
assert general["sources"] == []
assert general["answer"] == "I can only answer questions about Zepto policies right now."

print("Support assistant mock-mode tests passed.")
