# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python prompt_evals.py foo 19 &
done
wait