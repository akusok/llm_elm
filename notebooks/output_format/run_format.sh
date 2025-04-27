# Run the command 10 times in parallel
for i in {1..3}; do
  uv run python format_evals.py foo 30 &
done
wait