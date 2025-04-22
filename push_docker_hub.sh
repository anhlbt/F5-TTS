#!/bin/bash

# Define variables
IMAGE_NAME="f5-tts-f5-tts"         # Local image name
IMAGE_TAG="latest"                   # Local image tag (default is latest)
DOCKER_USERNAME="anhlbt"  # Docker Hub username
DOCKER_REPO="anhlbt/f5-tts-f5-tts"        # Docker Hub repository name (e.g., "yourusername/your-repo")

# Login to Docker Hub
echo "Logging in to Docker Hub..."
docker login -u "$DOCKER_USERNAME" || { echo "Docker login failed"; exit 1; }

# Tag the Docker image
echo "Tagging image..."
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$DOCKER_REPO:$IMAGE_TAG" || { echo "Failed to tag image"; exit 1; }

# Push the Docker image
echo "Pushing image to Docker Hub..."
docker push "$DOCKER_REPO:$IMAGE_TAG" || { echo "Failed to push image"; exit 1; }

echo "Image pushed successfully to Docker Hub!"
