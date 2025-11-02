from fastapi import APIRouter, HTTPException
from typing import List
import os
import json
import yfinance as yf

router = APIRouter()

# Assuming the API is run from the project root, this relative path should work.
WATCHLIST_FILE = "Watchlist/watchlist.json"

@router.get("/")
async def get_watchlist():
    """Get user's watchlist"""
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add/{symbol}")
async def add_to_watchlist(symbol: str):
    """Add a single stock to the watchlist"""
    try:
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or 'regularMarketPrice' not in info:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        watchlist = []
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, "r") as f:
                watchlist = json.load(f)
        
        if symbol not in watchlist:
            watchlist.append(symbol)
            with open(WATCHLIST_FILE, "w") as f:
                json.dump(watchlist, f)
        
        return {"message": f"{symbol} added to watchlist"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-symbols")
async def add_symbols_to_watchlist(symbols: List[str]):
    """Add multiple symbols to the watchlist"""
    try:
        watchlist = []
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, "r") as f:
                watchlist = json.load(f)
        
        added_count = 0
        for symbol in symbols:
            if symbol not in watchlist:
                watchlist.append(symbol)
                added_count += 1
        
        if added_count > 0:
            with open(WATCHLIST_FILE, "w") as f:
                json.dump(watchlist, f)
        
        return {"message": f"{added_count} new symbol(s) added to watchlist."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))