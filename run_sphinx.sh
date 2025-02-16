docker run -it --rm --platform linux/amd64 \
-v $(pwd):/workspace/\:rw -w /workspace \
rltools/documentation-builder \
"$@"
