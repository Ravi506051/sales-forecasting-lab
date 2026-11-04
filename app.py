import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
st.title("Sales Forecasting Lab"); st.caption("Synthetic monthly series with a holdout backtest.")
r=np.random.default_rng(11); dates=pd.date_range("2019-01-01",periods=84,freq="MS"); y=300+np.linspace(0,350,84)+80*np.sin(2*np.pi*np.arange(84)/12)+r.normal(0,35,84)
d=pd.DataFrame({"date":dates,"sales":y}).set_index("date")
f=pd.DataFrame({"y":d.sales})
for lag in [1,2,3,6,12]: f[f"lag{lag}"]=f.y.shift(lag)
f["month"]=f.index.month; f=f.dropna(); split=int(len(f)*.8); tr,te=f.iloc[:split],f.iloc[split:]
m=RandomForestRegressor(n_estimators=200,random_state=42).fit(tr.drop(columns="y"),tr.y); pred=m.predict(te.drop(columns="y"))
st.metric("Backtest MAE",f"{mean_absolute_error(te.y,pred):.1f}"); st.plotly_chart(px.line(d.reset_index(),x="date",y="sales"),use_container_width=True)
h=st.slider("Months to forecast",1,12,6); hist=list(d.sales); out=[]
for i in range(h):
 row={f"lag{lag}":hist[-lag] for lag in [1,2,3,6,12]}; row["month"]=(dates[-1]+pd.offsets.MonthBegin(i+1)).month; val=float(m.predict(pd.DataFrame([row]))[0]); out.append(val); hist.append(val)
fc=pd.DataFrame({"date":pd.date_range(dates[-1]+pd.offsets.MonthBegin(1),periods=h,freq="MS"),"forecast":out})
st.plotly_chart(px.line(fc,x="date",y="forecast",markers=True),use_container_width=True); st.dataframe(fc)
