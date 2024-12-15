FROM  nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get upgrade -y

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get install --no-install-recommends -y \
    software-properties-common \
    vim \
    python3-pip\
    tmux \
    git 

# Added updated mesa drivers for integration with cpu - https://github.com/ros2/rviz/issues/948#issuecomment-1428979499
RUN add-apt-repository ppa:kisak/kisak-mesa && \
    apt-get update && apt-get upgrade -y &&\
    apt-get install libxcb-cursor0 -y && \
    apt-get install ffmpeg python3-opengl -y

RUN pip3 install matplotlib PyQt5 dill pandas pyqtgraph

RUN add-apt-repository universe

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get install -y curl && \
     curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
     echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
     apt-get update && apt-get install -y ros-humble-ros-base


RUN apt-get install --no-install-recommends -y ros-humble-rviz2

ENV ROS_DISTRO=humble

# # Cyclone DDS
# RUN apt-get update --fix-missing 
RUN apt-get install --no-install-recommends -y \
    ros-$ROS_DISTRO-cyclonedds \
    ros-$ROS_DISTRO-rmw-cyclonedds-cpp

# Use cyclone DDS by default
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Source by default
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc

RUN pip3 install -U colcon-common-extensions \
    && apt-get install -y build-essential python3-rosdep

RUN \ 
    pip3 install --no-cache-dir Cython


RUN pip3 install  "jax[cuda12]" 

RUN pip3 install cuda-python && pip3 install numba
RUN pip3 install \
    'torch>=2.1.2,<2.2' \
    'gym>=0.26.2,<0.27' \
    'urdfenvs>=0.9.12,<0.10' \
    'mypy>=1.8.0,<1.9' \
    'ghalton==0.6.1' \
    'tqdm>=4.66.1,<4.67'

RUN git clone https://github.com/giaf/blasfeo.git && cd blasfeo && make shared_library -j 4 && sudo make install_shared
RUN git clone https://github.com/giaf/hpipm.git  && cd hpipm && git checkout 6a0267dca70d6377859efc82dca8a5a1c509c2ae && make shared_library -j 4 && sudo make install_shared  && cd  /hpipm/interfaces/python/hpipm_python/ && pip3 install .
#RUN git clone https://github.com/prajwalthakur/hpipm-master.git  && mv hpipm-master hpipm && cd hpipm && make shared_library -j 4 && sudo make install_shared  && cd  /hpipm/interfaces/python/hpipm_python/ && pip3 install .

# Add library paths to LD_LIBRARY_PATH
RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib:/blasfeo/lib:/hpipm/lib" >> /root/.bashrc
RUN echo "source /hpipm/examples/python/env.sh">>/root/.bashrc

ENV WORKSPACE_PATH=/root/workspace

COPY workspace/ $WORKSPACE_PATH/src/

SHELL ["/bin/bash", "-c"]


# RUN rosdep update && cd $WORKSPACE_PATH && \
#     rosdep install --from-paths src -y --ignore-src

# COPY scripts/setup/ /root/scripts/setup
# RUN  /root/scripts/setup/workspace.sh

