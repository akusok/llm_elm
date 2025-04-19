# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py qwen2.5-coder:1.5b 10 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py cogito:3b 15 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py granite3.3:2b 15 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py qwen2.5-coder:3b 18 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py granite3.3 18 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py llama3.1 18 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py qwen2.5-coder:7b 10 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py cogito:8b 10 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py phi4 16 &
done
wait


# Run the command 10 times in parallel
for i in {1..2}; do
  uv run python text_evals.py cogito:14b 40 &
done
wait

# Run the command 10 times in parallel
for i in {1..2}; do
  uv run python text_evals.py qwen2.5-coder:14b 40 &
done
wait

# Run the command 10 times in parallel
for i in {1..2}; do
  uv run python text_evals.py cogito:32b 15 &
done
wait

# Run the command 10 times in parallel
for i in {1..2}; do
  uv run python text_evals.py qwen2.5-coder:32b 15 &
done
wait

# Run the command 10 times in parallel
for i in {1..2}; do
  uv run python text_evals.py cogito:32b 30 &
done
wait

# Run the command 10 times in parallel
for i in {1..2}; do
  uv run python text_evals.py qwen2.5-coder:32b 30 &
done
wait
