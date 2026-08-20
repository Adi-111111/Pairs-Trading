from fastapi import FastAPI, HTTPException, Query
from pymongo import MongoClient
from datetime import datetime
from typing import List

app = FastAPI(title="SSE Stock Data API")
client = MongoClient("mongodb://localhost:27017/")
db = client["sse_market_data"]

@app.get("/data/")
def get_stock_data(start_date: str, end_date: str, ticker: str, fields: List[str] = Query(...)):
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        collection = db[ticker]

        proj = {"_id": 0, "Date": 1}
        for f in fields:
            proj[f] = 1

        query = {"Date": {"$gte": start_dt, "$lte": end_dt}}
        records = list(collection.find(query, proj).sort("Date", 1))

        if not records:
            raise HTTPException(status_code=404, detail="No data found for the given parameters.")

        for r in records:
            r['Date'] = r['Date'].strftime('%Y-%m-%d')

        return records

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
