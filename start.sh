#!/bin/bash

# UGE - Grammatical Evolution for Classification
# Startup script for easy deployment and release management

set -e

VERSION="1.0.0"
RELEASE_DATE="2025-09-14"

echo "🚀 UGE v${VERSION} - Grammatical Evolution for Classification"
echo "=================================================="
echo "Release Date: ${RELEASE_DATE}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed (V2)
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p results/setups datasets grammars

# Set permissions
echo "🔐 Setting permissions..."
chmod -R 755 results/ datasets/ grammars/

# Build and start the application
echo "🐳 Building UGE v${VERSION} Docker image..."
docker compose build

echo "🚀 Starting UGE v${VERSION} application..."
docker compose up -d

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is running
if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
    echo "✅ UGE v${VERSION} is running successfully!"
    echo ""
    echo "🎉 UGE v${VERSION} Release Complete!"
    echo "   Release Date: ${RELEASE_DATE}"
    echo "   Docker Images: uge:${VERSION}, uge:latest"
    echo "   Container: uge-application-v1.0.0"
    echo ""
    echo "📋 Usage Information:"
    echo "🌐 Access the application at: http://localhost:8501"
    echo "📊 View logs with: docker compose logs -f"
    echo "🛑 Stop the application with: docker compose down"
    echo ""
    echo "📚 For more information, see INSTALLATION.md"
else
    echo "❌ Application failed to start. Check logs with: docker compose logs"
    exit 1
fi
