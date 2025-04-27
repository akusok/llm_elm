# Run the command 10 times in parallel
for i in {1..2}; do
  uv run python prompt_evals.py foo 50 &
done
wait