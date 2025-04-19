# Run the command 10 times in parallel
for i in {1..10}; do
  uv run python text_evals.py deepcoder &
done
wait