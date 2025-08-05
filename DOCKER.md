# Docker Setup for Wealth Assistant (Production)

This project has been dockerized for production deployment with separate containers for the backend (FastAPI) and frontend (React/Vite).

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Copy `.env.example` to `.env` and configure your environment variables

### Production Deployment

```bash
docker-compose up -d
```

### Using the Shell Script (Alternative)

```bash
# Make executable first
chmod +x docker.sh

# Start production deployment
./docker.sh up
```

## Available Commands

Using docker-compose directly:

- `docker-compose build` - Build all Docker images
- `docker-compose up -d` - Start services in production mode
- `docker-compose down` - Stop all services
- `docker-compose logs -f` - Show logs from all services
- `docker-compose restart` - Restart all services
- `docker-compose down -v --rmi all --remove-orphans` - Remove all containers, images, and volumes
- `docker-compose ps` - Check service status

**Alternative: Using the provided shell script:**

```bash
# Make script executable (one time)
chmod +x docker.sh

# Available commands
./docker.sh help     # Show all commands
./docker.sh build    # Build all images
./docker.sh up       # Start production services
./docker.sh down     # Stop all services
./docker.sh logs     # Show logs
./docker.sh status   # Check service status
./docker.sh clean    # Clean up everything
```

## Services

### Backend (FastAPI)

- **Port**: 8000
- **Health Check**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs

### Frontend (React/Vite)

- **Port**: 80
- **URL**: http://localhost

## Environment Variables

Create a `.env` file based on `.env.example`:

```env
GOOGLE_API_KEY=your_google_api_key_here
MONGO_URI=your_mongodb_connection_string_here
MONGO_DB_NAME=Project0
DEBUG_MODE=False
REACT_APP_API_URL=http://localhost:8000
```

## Production Configuration

Your application is configured with production-ready features:

- **Optimized builds**: Multi-stage Docker builds for smaller images
- **Security**: Non-root users in containers
- **Monitoring**: Health checks for service monitoring
- **Performance**: Resource limits and optimizations
- **Reliability**: Automatic restart policies
- **Networking**: Isolated container network

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend       │
│   (React/Vite)  │     │   (FastAPI)     │
│   Port: 80      │     │   Port: 8000    │
└─────────────────┘     └─────────────────┘
        │                       │
        │                       │
        ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   Nginx         │     │   MongoDB       │
│   (Production)  │     │   (External)    │
└─────────────────┘     └─────────────────┘
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Make sure ports 80 and 8000 are not in use
2. **Permission issues**: Ensure Docker has proper permissions
3. **Build failures**: Run `docker-compose down -v --rmi all --remove-orphans` and rebuild

### Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs backend
docker-compose logs frontend
```

### Health Checks

```bash
# Check service status
docker-compose ps

# Manual health check
curl http://localhost:8000/
curl http://localhost/
```

## Security Features

- Non-root users in containers
- Health checks for service monitoring
- Resource limits in production
- Minimal base images
- Environment variable security
- Volume mounts for persistent data

## Performance Optimizations

- Multi-stage builds for smaller images
- Layer caching optimization
- Resource limits
- Health checks for reliability
- Separate dev/prod configurations
