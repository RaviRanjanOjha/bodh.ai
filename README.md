# Wealth Management AI Assistant

![Project Logo](https://placehold.co/600x200?text=Wealth+AI+Assistant)

A comprehensive AI-powered system to support financial advisors with portfolio analysis, compliance checking, and client interactions.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [API Documentation](#api-documentation)
7. [Development](#development)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview

The Wealth Management AI Assistant is designed to:
- Analyze client portfolios with AI insights
- Provide natural language chat assistance
- Process financial documents automatically
- Ensure regulatory compliance
- Generate interactive visualizations

![System Architecture](https://placehold.co/600x300?text=System+Architecture+Diagram)

## Features

### Core Capabilities
✅ AI-Powered Financial Chat Assistant  
✅ Automated Document Processing  
✅ Compliance Verification Engine  
✅ Portfolio Visualization Tools  
✅ Market Data Integration  

### Technical Components
- **API Layer**: FastAPI endpoints for all services
- **Core Services**: Business logic and AI integration
- **Database**: Client data models and storage
- **UI**: Gradio-based web interface

## Installation

### Prerequisites
- Python 3.13.0
- MongoDB
- Gemini API key

### Setup
```bash
git clone https://github.com/your-repo/wealth-assistant.git
cd wealth-assistant
pip install -r requirements.txt
```

## Configuration
Essential environment variables:

```
GOOGLE_API_KEY=your_google_api_key
MONGO_URI=your_mongodb_uri
MONGO_DB_NAME=your_db_name
DEBUG_MODE=False
API_HOST=0.0.0.0
API_PORT=8000
```

## Usage
### Running the Application

```
uvicorn main:app --reload
```

Access the web interface:

```
python -m ui.gradio_interface
```

### Example API Calls
```python
from wealth_assistant import ChatService

# Initialize chat service
chat = ChatService()

# Get portfolio analysis
response = chat.generate_response(
    "What is the risk exposure for client 123?",
    client_id="123"
)
```

## API Documentation
### Key Endpoints

| Endpoint          | Method | Description                  |
|-------------------|--------|------------------------------|
| /chat             | POST   | AI conversation interface    |
| /clients/{id}     | GET    | Client portfolio data        |
| /documents        | POST   | Upload financial documents   |
| /visualizations   | GET    | Generate portfolio charts    |

Full API docs available at `/docs` when running locally.

## Development
### Code Structure

```
wealth_assistant/
├── api/             # API endpoints
├── config/          # Configuration
├── database/        # Data models
├── services/        # Business logic
├── ui/              # User interface
└── visualization/   # Charting tools
```

## Contributing
- Fork the repository
- Create a feature branch
- Submit a pull request
- Include tests for new features

## Troubleshooting
### Document Processing Fails
- Verify file size limits
- Check PDF libraries are installed

### LLM Not Responding
- Confirm API key validity
- Test with simpler prompts

### Chart Rendering Issues
- Validate input data structure
- Check Plotly version compatibility

