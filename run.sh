if [ $(uname -s) = "Linux" ]; then
  USER_OPTION="--user $(id -u):$(id -g)"
fi
docker run -it --rm --platform linux/amd64 -p 8888\:8888 \
$USER_OPTION \
-v $(pwd)/../rl-tools/include/rl_tools/\:/usr/local/include/rl_tools/\:ro \
-v $(pwd):/workspace/\:rw -w /workspace \
$IMAGE \
"$@"
