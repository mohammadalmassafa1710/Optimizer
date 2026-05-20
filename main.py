import io
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from optimizer import run_cutting_stock_optimization

app = FastAPI(
    title="M&D Optimizer API",
    description="Backend API for Cutting Stock Operations Research optimization.",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    orders: List[Dict[str, Any]]
    params: Dict[str, Any]

class ExportRequest(BaseModel):
    summary: Dict[str, Any]
    production_plan: List[Dict[str, Any]]
    deckle_detail: List[Dict[str, Any]]
    remaining_by_customer: List[Dict[str, Any]]
    widths_used: List[Dict[str, Any]]

# Sample dataset
SAMPLE_ORDERS = [
    {"Customer": "Apex Packaging", "Product": "Kraft Liner", "GSM": 125, "Roll Width": 1850, "Quantity": 150.0},
    {"Customer": "Durabox Ltd", "Product": "Testliner", "GSM": 140, "Roll Width": 1200, "Quantity": 300.0},
    {"Customer": "Global Corrugated", "Product": "Kraft Liner", "GSM": 125, "Roll Width": 1100, "Quantity": 120.0},
    {"Customer": "EcoPack Co", "Product": "Fluting Medium", "GSM": 112, "Roll Width": 950, "Quantity": 90.0}
]

@app.get("/api/sample-data")
def get_sample_data():
    return SAMPLE_ORDERS

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Parses an uploaded Excel file and returns a list of orders.
    Expected columns: Customer, Product, GSM, Roll Width, Quantity
    """
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported.")
    
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Validate columns
        required_cols = ['Customer', 'Product', 'GSM', 'Roll Width', 'Quantity']
        # Case insensitive mapping
        col_mapping = {}
        for req in required_cols:
            found = False
            for col in df.columns:
                if str(col).strip().lower() == req.lower():
                    col_mapping[col] = req
                    found = True
                    break
            if not found:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required column: '{req}'. File must contain: {', '.join(required_cols)}"
                )
        
        df = df[list(col_mapping.keys())].rename(columns=col_mapping)
        df = df.dropna()
        
        # Format types
        df['Roll Width'] = df['Roll Width'].astype(int)
        df['Quantity'] = df['Quantity'].astype(float)
        
        return df.to_dict(orient='records')
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Excel file: {str(e)}")

@app.post("/api/optimize")
def optimize(payload: OptimizeRequest):
    """
    Runs the PuLP cutting stock solver and returns the plan.
    """
    if not payload.orders:
        raise HTTPException(status_code=400, detail="No orders provided.")
    
    try:
        status_str, summary, plan, detail, remaining, widths_used = run_cutting_stock_optimization(
            payload.orders, payload.params
        )
        
        if status_str in ["Infeasible", "Undefined"]:
            return {
                "success": False,
                "status": status_str,
                "error": f"The model is {status_str.upper()}. Please adjust parameters like Min Fill or increase Tolerance."
            }
            
        return {
            "success": True,
            "status": status_str,
            "summary": summary,
            "production_plan": plan,
            "deckle_detail": detail,
            "remaining_by_customer": remaining,
            "widths_used": widths_used
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.post("/api/export")
def export_excel(payload: ExportRequest):
    """
    Generates and returns an Excel workbook containing the results.
    """
    try:
        plan_df = pd.DataFrame(payload.production_plan)
        detail_df = pd.DataFrame(payload.deckle_detail)
        remaining_df = pd.DataFrame(payload.remaining_by_customer)
        widths_df = pd.DataFrame(payload.widths_used)
        summary_df = pd.DataFrame([payload.summary])
        
        # Write to in-memory Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            plan_df.to_excel(writer, sheet_name="Production_Plan", index=False)
            detail_df.to_excel(writer, sheet_name="Deckle_Detail", index=False)
            remaining_df.to_excel(writer, sheet_name="Remaining_By_Customer", index=False)
            widths_df.to_excel(writer, sheet_name="Deckle Width Used", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            # Auto-fit columns
            for sheet in writer.sheets.values():
                sheet.autofit()
                
        output.seek(0)
        
        filename = f"deckle_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return StreamingResponse(
            output, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# Serve Static files - MUST be at the end so it doesn't hijack API routes
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
