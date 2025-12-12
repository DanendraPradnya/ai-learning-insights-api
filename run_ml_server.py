from waitress import serve
from app import app
import os

if __name__ == '__main__':
  print("Running ML Server for AI Learning Insight with Waitress...")  
  port = int(os.environ.get("PORT", 5000))
  print(f"Starting Waitress server on 0.0.0.0:{port}")
    
  serve(
    app,
    host='0.0.0.0',
    port=port,
    threads=4 
    )