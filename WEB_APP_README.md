# Live Trading Scanner - Web App

Web interface for the live trading scanner with real-time monitoring and control.

## Features

- 📊 Real-time dashboard showing market status and scanner state
- ➕ Add/remove stocks from hotlist
- 🔄 Revalidate hotlist with Pipeline 1
- ▶️ Start/stop live scanner
- 🚨 View buy signals generated today
- 🌐 Accessible from anywhere via web browser

## Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```bash
TWELVEDATA_API_KEY=your_key_here
```

3. Run the web app:
```bash
python src/web_app/app.py
```

4. Open browser to http://localhost:5000

## Deploying to Render

1. Push code to GitHub

2. Go to https://render.com and sign up

3. Create new Web Service:
   - Connect your GitHub repo
   - Name: `stock-scanner` (or your choice)
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --chdir src/web_app app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2`

4. Add environment variables in Render dashboard:
   - `TWELVEDATA_API_KEY`: your TwelveData API key

5. Deploy! Your app will be live at `https://your-app-name.onrender.com`

## API Endpoints

- `GET /api/status` - Get scanner and market status
- `GET /api/hotlist` - Get current hotlist
- `POST /api/hotlist/add` - Add ticker to hotlist
- `POST /api/hotlist/remove` - Remove ticker from hotlist
- `POST /api/hotlist/revalidate` - Revalidate all tickers
- `POST /api/hotlist/clear` - Clear entire hotlist
- `GET /api/signals` - Get signals generated today
- `POST /api/scanner/start` - Start live scanner
- `POST /api/scanner/stop` - Stop live scanner

## Architecture

- **Backend**: Flask API with background threading
- **Frontend**: Vanilla JavaScript with auto-refresh
- **Deployment**: Gunicorn WSGI server on Render
- **Storage**: JSON files (persisted on Render disk)

## Notes

- Scanner runs during market hours (9:35 AM - 4:00 PM ET)
- Auto-refreshes dashboard every 5 seconds
- Mobile-friendly responsive design
- Free tier on Render stays awake 24/7 if you keep using it
