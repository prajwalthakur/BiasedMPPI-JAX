#!/bin/bash
absolute_path=/home/prajwal/projects/github-projects/ros_projects/jax_mppi/
# run_docker() {
#     # -it is for interactive, tty
#     # --privileged for accessing /dev contents
#     # --net=host to share the same network as host machine. TL;DR same IP.
#     docker run -it --privileged --net=host \
#     --name f1tenth_humble \
#     --env="DISPLAY" \
#     --env="QT_X11_NO_MITSHM=1" \
#     -v $absolute_path/scripts/deploy/app.sh:/root/app.sh \
#     -v $absolute_path/scripts/deploy/cyclonedds.xml:/root/cyclonedds.xml \
#     -v /etc/udev/rules.d:/etc/udev/rules.d \
#     -v /dev:/dev \
#     --env="CYCLONEDDS_URI=file:///root/cyclonedds.xml" \
#     $@
# }
run_docker() {
    # -it is for interactive, tty
    # --privileged for accessing /dev contents
    # --net=host to share the same network as host machine. TL;DR same IP.
    xhost +local:root # giving display privilages
    docker run -it --privileged --net=host \
    --name project_docker \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    -v $absolute_path/scripts/deploy/app.sh:/root/app.sh \
    $@
}



stop_docker() {
    docker stop project_docker && docker rm project_docker
}