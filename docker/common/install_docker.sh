#!/bin/bash

if ! command -v docker &> /dev/null; then
  echo "Installing Docker..."

  if [ ! -f /usr/share/keyrings/docker-archive-keyring.gpg ]; then
    sudo apt-get update -qq > /dev/null
    sudo apt-get install -qq -y curl apt-transport-https ca-certificates software-properties-common > /dev/null
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg > /dev/null
    sudo echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq > /dev/null
  fi

  sudo apt-get install -qq -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin > /dev/null
fi

if ! service docker status > /dev/null; then
  sudo service docker start
  sudo service docker status
  sleep 1
fi
