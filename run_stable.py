# Run this to start the server WITHOUT auto-reload
# This prevents constant restarts from package file changes

import os
os.environ['FLASK_DEBUG'] = '0'  # Disable debug mode to prevent auto-reload

from app import app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SafeHome Server - Production Mode (No Auto-Reload)")
    print("="*60)
    print("\nServer will start at: http://127.0.0.1:5000/dashboard")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
