#!/bin/bash

docker build --rm  $@ -t project_docker:latest -f "$(dirname "$0")/../../docker/project.Dockerfile" "$(dirname "$0")/../.."