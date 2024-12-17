import argparse
from pathlib import Path
import json
import logging
import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.chat.transcript_query import TranscriptQuery

def main():
    parser = argparse.ArgumentParser(description='Transcript Chat CLI')
    parser.add_argument('--model', default='llama3.2', 
                       help='Ollama model name (default: llama3.2)')
    parser.add_argument('--db', required=True, help='Path to transcript database')
    parser.add_argument('--mode', choices=['chat', 'actions', 'speaker'], 
                       default='chat', help='Operation mode')
    parser.add_argument('--speaker', help='Speaker name for speaker analysis mode')
    
    args = parser.parse_args()
    
    try:
        # Initialize query system
        query_system = TranscriptQuery(
            db_path=Path(args.db),
            model_name=args.model
        )
        
        if args.mode == 'chat':
            print(f"Chat mode using model {query_system.model_name}")
            print("Ask questions about the transcripts (type 'quit' to exit)")
            while True:
                question = input("\nYour question: ").strip()
                if question.lower() == 'quit':
                    break
                    
                response = query_system.query_transcripts(question)
                print("\nResponse:", response)
                
        elif args.mode == 'actions':
            print("Extracting action items from all transcripts...")
            actions = query_system.get_action_items()
            print("\nAction Items:")
            print(json.dumps(actions, indent=2))
            
        elif args.mode == 'speaker' and args.speaker:
            print(f"Analyzing contributions from speaker: {args.speaker}")
            summary = query_system.get_speaker_summary(args.speaker)
            print("\nSpeaker Analysis:")
            print(json.dumps(summary, indent=2))
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()