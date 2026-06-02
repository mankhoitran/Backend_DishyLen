import csv
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

class GeminiKeyManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GeminiKeyManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.keys = []
        self.current_index = 0
        self.google_sheet_url = "https://docs.google.com/spreadsheets/d/1vdEUol-xwwv7oyImB8cWgLUzg7IJivAGkxgMtDEPi6c/export?format=csv"
        self.load_keys()

    def load_keys(self):
        try:
            req = urllib.request.Request(self.google_sheet_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                lines = [l.decode('utf-8') for l in response.readlines()]
                reader = csv.reader(lines)
                for row in reader:
                    if len(row) >= 2:
                        gmail = row[0].strip()
                        api_key = row[1].strip()
                        if api_key and api_key != "API" and api_key != "API_KEY":  # Skip headers randomly if any
                            self.keys.append({"gmail": gmail, "key": api_key})
            
            logger.info(f"Loaded {len(self.keys)} Gemini API keys.")
        except Exception as e:
            logger.error(f"Failed to load keys from Google Sheets: {e}")

    def get_current_key(self) -> Optional[str]:
        if not self.keys:
            return None
        return self.keys[self.current_index]["key"]
    
    def switch_to_next_key(self) -> Optional[str]:
        """Switch to another key if current is rate-limited or busy."""
        if not self.keys:
            return None
        
        self.current_index = (self.current_index + 1) % len(self.keys)
        logger.info(f"Switched Gemini API key to account: {self.keys[self.current_index]['gmail']}")
        return self.get_current_key()
