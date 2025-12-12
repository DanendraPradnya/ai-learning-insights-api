from waitress import serve
from app import app

if __name__ == '__main__':
  print("Running ML Server for AI Learning Insight with Waitress...")
  print("Serving on http://localhost:5000")
  
  serve(
    app,
    host='localhost',
    port=5000,
    threads=4,
    connection_limit=1000,
    channel_timeout=60
  )