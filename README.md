# EcoWatt

EcoWatt forecasts next-day household electricity consumption from the UCI
Individual Household Electric Power Consumption dataset. The source is
minute-level active power in kW; invalid `?` readings are ignored, incomplete
days are dropped, retained days with at least 95% valid minutes are
time-interpolated, and daily kWh is calculated as the sum of kW readings
divided by 60. The model uses lagged consumption and calendar features, with a
chronological 80/20 test split and a previous-day baseline.

## Run locally

From the repository root, using the existing environment:

```powershell
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000
```

Serve `index.html` from the frontend directory (for example with VS Code
Live Server) on port 5500. The dashboard calls `http://localhost:8000`.
Generated model artifacts are intentionally ignored by Git and can always be
recreated with the training command.

Run tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The What-If Simulator remains a transparent scenario estimate based on its
temperature, AC, appliance, occupancy, and day-type sliders. Those variables
are not present in the UCI training data and are not sent to the trained model.
