# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py yandex/YandexGPT-5-Lite-8B-instruct-GGUF 20 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py exaone3.5 20 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py command-r7b 20 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py marco-o1 10 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py gemma3:12b-it-qat 20 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py exaone-deep 2 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py tulu3 20 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py opencoder 20 &
done
wait

# Run the command 10 times in parallel
for i in {1..5}; do
  uv run python text_evals.py falcon3 20 &
done
wait
