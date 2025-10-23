import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from model import predict_future_incidents, get_device_alerts, get_incident_summary
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Incident Trends Dashboard", layout="wide")
st.title("🛢️ Real-Time Oil & Gas Incident Forecasting")

summary = get_incident_summary()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Incidents", summary["total_incidents"])
col2.metric("Plants Monitored", summary["plants"])
col3.metric("Zones Monitored", summary["zones"])
col4.metric("Devices Monitored", summary["devices"])

device_alerts = get_device_alerts()
device_df = pd.DataFrame(device_alerts)
if not device_df.empty:
    device_df = device_df.sort_values(by="is_incident", ascending=False)
    fig_device = px.bar(
        device_df,
        x="DeviceName",
        y="is_incident",
        title="🚨 Incidents per Device",
        color="is_incident",
        color_continuous_scale=["#00cc96", "#ffa600", "#ff0000"],
        text="is_incident"
    )
    fig_device.update_traces(textposition="outside")
    fig_device.update_layout(
        xaxis_title="Device",
        yaxis_title="Incident Count",
        template="plotly_dark",
        font=dict(color="white")
    )
    st.plotly_chart(fig_device, use_container_width=True)
else:
    st.warning("No device alerts data available.")

st.subheader("📈 Forecasted Incidents")
period_option = st.selectbox(
    "Select Prediction Period",
    ["Today", "Next Week", "Next Month", "Next Quarter", "Next Year"]
)
period_days = {
    "Today": 1,
    "Next Week": 7,
    "Next Month": 30,
    "Next Quarter": 90,
    "Next Year": 365
}
num_days = period_days[period_option]
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

try:
    preds = predict_future_incidents(current_timestamp=current_time, num_days=num_days)
    pred_df = pd.DataFrame(preds)
    if not pred_df.empty:
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=pred_df["date"],
            y=pred_df["predicted_incident_count"],
            mode="lines+markers",
            line=dict(color="#ffa600", width=3),
            marker=dict(size=8, color=pred_df["predicted_incident_count"], colorscale="RdYlGn_r"),
            name="Predicted Incidents",
            hovertemplate="<b>Date:</b> %{x}<br><b>Incidents:</b> %{y}<extra></extra>"
        ))
        fig_forecast.add_trace(go.Scatter(
            x=pred_df["date"],
            y=pred_df["predicted_incident_count"],
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(255,166,0,0.2)",
            name="Incident Density"
        ))
        fig_forecast.update_layout(
            title=f"⛽ Predicted Incidents - {period_option}",
            xaxis_title="Date",
            yaxis_title="Incident Count",
            template="plotly_dark",
            font=dict(color="white"),
            hovermode="x unified"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.dataframe(pred_df)
    else:
        st.info("No forecast data available.")
except Exception as e:
    st.error(f"Prediction failed: {str(e)}")