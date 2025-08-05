#!/bin/bash

# Docker management script for Wealth Assistant (Production)
# Usage: ./docker.sh [command]

set -e

help() {
    echo "Available commands:"
    echo "  build     - Build all Docker images"
    echo "  up        - Start services in production mode"
    echo "  down      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  logs      - Show logs from all services"
    echo "  status    - Check service status"
    echo "  clean     - Remove all containers, images, and volumes"
    echo "  help      - Show this help message"
}

build() {
    echo "Building all Docker images..."
    docker-compose build
}

up() {
    echo "Starting services in production mode..."
    docker-compose up -d
}

down() {
    echo "Stopping all services..."
    docker-compose down
}

restart() {
    echo "Restarting all services..."
    docker-compose restart
}

logs() {
    echo "Showing logs from all services..."
    docker-compose logs -f
}

status() {
    echo "Checking service status..."
    docker-compose ps
}

clean() {
    echo "Cleaning up all containers, images, and volumes..."
    docker-compose down -v --rmi all --remove-orphans
    docker system prune -f
}

# Main script logic
case "${1:-help}" in
    build)
        build
        ;;
    up)
        up
        ;;
    down)
        down
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    clean)
        clean
        ;;
    help|*)
        help
        ;;
esac
