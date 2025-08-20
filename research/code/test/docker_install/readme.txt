sudo docker build -t zmedia .
sudo docker run -p 8501:8501 zmedia

docker system prune --all --force --volumes
docker system df

docker system df
docker system prune -a
docker volume rm $(docker volume ls -qf dangling=true)

docker image prune [-af if you want to force remove all images]

docker images --no-trunc | awk '/<none>/ { print $3 }' \
| xargs docker rmi

---
2


Updating /usr/lib/systemd/system/docker.service to add -g /apps/newdocker/docker as per the other answers DID NOT WORK for me (on rhel8)

You can check it with docker info -f '{{ .DockerRootDir }}'

Following baeldung's instructions did work:

$ sudo vim /etc/docker/daemon.json
Insert the the path as you like:

{ 
   "data-root": "/tmp/new-docker-root-dir"
}
Then the usual docker restart

sudo systemctl stop docker
sudo systemctl start docker

---

How to Fix "Docker: No Space Left on Device" Error
#
docker
The "Docker: No Space Left on Device" error is a common roadblock for developers using Docker containers. This error occurs when Docker runs out of available disk space, halting your containerization efforts. Let's dive into the causes of this error and explore effective solutions to get your Docker environment back on track.

Understanding the "No Space Left on Device" Error in Docker
The "No Space Left on Device" error in Docker indicates that the filesystem where Docker stores its data has run out of free space. Docker uses disk space to store images, containers, volumes, and other data in the /var/lib/docker directory by default. When this directory fills up, Docker operations—such as pulling images, creating containers, or writing to volumes—fail.

Docker's space usage can quickly balloon due to:

Accumulated unused images and containers
Orphaned volumes
Build cache from image creation
Logs and other operational data
This error significantly impacts Docker operations, preventing you from creating new containers, pulling images, or even running existing containers in some cases.

Quick Fix: Freeing Up Space in Docker
To quickly resolve the "No Space Left on Device" error, use these commands to clean up Docker resources:

Remove unused Docker objects:


docker system prune


This command removes all stopped containers, unused networks, dangling images, and build cache.

Image description

Remove unused images:


docker image prune


Clean up volumes:


docker volume prune


After running these commands, restart the Docker service:



sudo systemctl restart docker


Monitoring Docker Disk Usage
To prevent future space issues, regularly monitor your Docker disk usage:

Check overall disk space:


df -h


View Docker-specific usage:


docker system df


Set up alerts for low disk space and implement a regular maintenance schedule to keep your Docker environment healthy.

Root Causes of Docker Space Issues
Understanding the root causes helps in long-term prevention:

Accumulation of unused resources: Over time, unused images, containers, and volumes consume significant space.
Large or numerous Docker volumes: Data-heavy applications can quickly fill up volumes.
Default storage location limitations: The partition containing /var/lib/docker may be too small.
System-wide disk space shortage: Your entire system might be running low on space, affecting Docker operations.
Advanced Solutions for Docker Space Management
For more robust space management:

Configure a different storage driver: Some drivers are more space-efficient. Research options like overlay2 or devicemapper.

Move Docker's data directory: Relocate to a larger partition:



   # Stop Docker
   sudo systemctl stop docker

   # Move the data
   sudo mv /var/lib/docker /path/to/new/location

   # Update Docker's configuration
   sudo nano /etc/docker/daemon.json
   # Add: {"data-root": "/path/to/new/location"}

   # Restart Docker
   sudo systemctl start docker


Implement image lifecycle policies: Automatically remove old or unused images based on age or usage patterns.

Use Docker Compose: Better manage resources across multiple containers and services.

Optimizing Dockerfiles for Space Efficiency
Efficient Dockerfiles lead to smaller images and reduced space usage:

Use multi-stage builds: Separate build-time dependencies from runtime dependencies.

Minimize layer size:

Combine RUN commands
Clean up in the same layer where files are added
Leverage .dockerignore: Exclude unnecessary files from the build context.

Choose appropriate base images: Use slim or alpine versions when possible.

Preventing Future "No Space Left" Errors
Implement these strategies to avoid recurring space issues:

Set up regular cleanup cron jobs:


   0 0 * * * docker system prune -af --volumes


Use disk space monitoring tools: Consider tools like Prometheus with node_exporter for comprehensive monitoring.

Educate team members: Share Docker best practices across your development team.

Consider container orchestration: Solutions like Kubernetes can help with resource allocation and management.

Key Takeaways
The "No Space Left on Device" error often stems from Docker's storage management.
Regular cleanup and monitoring are crucial for preventing space issues.
Advanced solutions involve reconfiguring Docker's storage setup.
Optimizing Dockerfiles and implementing best practices can significantly reduce space usage.
FAQs
How often should I run Docker cleanup commands?
Run cleanup commands weekly or bi-weekly, depending on your Docker usage. For high-traffic environments, daily cleanups might be necessary.

Can I move Docker's data directory without losing my containers and images?
Yes, if you follow the correct procedure of stopping Docker, moving the data, updating the configuration, and restarting Docker.

What's the difference between docker system prune and docker image prune?
docker system prune removes unused data including stopped containers, unused networks, dangling images, and build cache. docker image prune only removes dangling images.

How do I increase Docker's default storage limit on Windows?
On Windows, you can increase Docker's storage limit through the Docker Desktop settings. Navigate to Settings > Resources > Advanced and adjust the "Disk image size" slider.

Resources
Docker official documentation on disk space issues
Advanced Docker storage driver configuration guide
Best practices for writing Dockerfiles

---
sudo apt update
sudo apt install ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
      "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null    

sudo apt update
    sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin      

sudo usermod -aG docker $USER

docker run hello-world