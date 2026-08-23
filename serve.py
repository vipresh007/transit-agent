"""Run the web app.

    pip install fastapi uvicorn
    python serve.py            ->  http://127.0.0.1:8000

A three-line launcher, like the others. The app lives in transit/web/app.py.
"""

from transit.web.app import main

if __name__ == "__main__":
    main()
