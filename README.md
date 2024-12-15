# BIASED-MPPI for Obstacle Avoidance

This is a dockerized project for running Model Predictive Path Integral (MPPI) control in PyBullet.  
[Biased-MPPI](https://arxiv.org/abs/2401.09241) has been implemented in JAX-Python for obstacle avoidance.


![mppi_image](https://github.com/user-attachments/assets/9ac9d165-0fe6-4ef6-8dad-7762273eec17)


https://github.com/user-attachments/assets/1d86231f-71e8-419d-9b92-ca5939365d75

## Steps to Reproduce Results

The project is containerized with Docker. To reproduce the results, follow these steps:

1. Clone the repository.
2. From the root directory of the repository, update the absolute path in `scripts/deploy/base.sh`.
3. Build the Docker image by running the following command:  
   ```bash
   ./scripts/build/project_docker.sh
4 . to run the docker image , execute : ./scripts/deploy/devel.sh

5 Once inside the Docker container, navigate to /root/workspace/src and run python3 main.py
