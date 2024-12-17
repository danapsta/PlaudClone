import requests
import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ollama_connection():
    """Test Ollama connection and API endpoints."""
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True)
        logger.info("Ollama installation check passed")
        logger.info(f"Available models:\n{result.stdout}")
    except FileNotFoundError:
        logger.error("Ollama is not installed or not in PATH")
        return False
        
    # Test API endpoints
    endpoints = [
        "http://localhost:11434",
        "http://127.0.0.1:11434"
    ]
    
    working_endpoint = None
    for endpoint in endpoints:
        try:
            # Test basic endpoint
            logger.info(f"Testing {endpoint}...")
            response = requests.get(f"{endpoint}/api/tags")
            
            if response.status_code == 200:
                logger.info(f"Successfully connected to {endpoint}")
                working_endpoint = endpoint
                break
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to connect to {endpoint}: {str(e)}")
            
    if not working_endpoint:
        logger.error("Could not connect to any Ollama API endpoint")
        return False
        
    # Test model generation
    try:
        logger.info("Testing model generation...")
        response = requests.post(
            f"{working_endpoint}/api/generate",
            json={
                "model": "llama3.2",
                "prompt": "Say hello",
                "stream": False
            }
        )
        
        if response.status_code == 200:
            logger.info("Successfully generated response from model")
            logger.info(f"Response: {response.json()['response']}")
            return True
        else:
            logger.error(f"Model generation failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Model generation test failed: {str(e)}")
        return False

if __name__ == "__main__":
    if test_ollama_connection():
        logger.info("All Ollama connection tests passed!")
        sys.exit(0)
    else:
        logger.error("Ollama connection tests failed")
        sys.exit(1)