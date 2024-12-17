# chat_cli.py
from src.chat.transcript_query import ChatInterface
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize chat interface
    db_path = Path("data/transcripts.db")
    print(f"Initializing chat interface with database: {db_path}")
    
    if not db_path.exists():
        print("Database not found!")
        return
        
    chat = ChatInterface(db_path)
    
    print("\nTranscript Chat Interface")
    print("------------------------")
    print("Type your questions about the transcripts (or 'quit' to exit)")
    print("Example questions:")
    print("- What was discussed in the latest meeting?")
    print("- Did anyone mention project deadlines?")
    print("- What did [speaker name] say about [topic]?")
    print("- Type 'list' to see all available transcripts")
    print()
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            print("\nSearching transcripts...")
            response = chat.process_query(query)
            print("\nResponse:")
            print("-" * 80)
            print(response)
            print("-" * 80)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print("Sorry, there was an error processing your query. Please try again.")

if __name__ == "__main__":
    main()