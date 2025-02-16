docker build . -t rltools/documentation-builder -f Dockerfile_sphinx $@
docker run -it --rm -v $(pwd):/mount rltools/documentation-builder cp /environment_sphinx.yml /mount