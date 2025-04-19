# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py cogito:8b 10 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py qwen2.5-coder:7b 10 &
done
wait

# Run the command 10 times in parallel
for i in {1..4}; do
  uv run python text_evals.py cogito:14b 5 &
done
wait

# Run the command 10 times in parallel
for i in {1..4}; do
  uv run python text_evals.py qwen2.5-coder:14b 5 &
done
wait

# Run the command 10 times in parallel
for i in {1..2}; do
  uv run python text_evals.py cogito:32b 5 &
done
wait

# Run the command 10 times in parallel
for i in {1..2}; do
  uv run python text_evals.py qwen2.5-coder:32b 5 &
done
wait
