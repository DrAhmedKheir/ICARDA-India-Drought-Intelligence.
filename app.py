import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
st.set_page_config(page_title="India Drought Intelligence",page_icon="🌾",layout="wide")
st.title("India | RS–ML Drought Intelligence")
st.caption("Research prototype · CHIRPS + ERA5-Land · 0.25° · SPEI-6 · Direct ML leads 1–3 months")
DATA=Path(__file__).resolve().parent
@st.cache_data(ttl=3600)
def read(name):
    path=DATA/name
    if not path.exists():return None
    return pd.read_csv(path)
if st.sidebar.button('Refresh published data'):
    st.cache_data.clear();st.rerun()
f=read('latest_spei6_forecasts.csv');scores=read('independent_hindcast_scores.csv')
if f is None or f.empty:
    st.error('No forecast data published. Run the Colab processing pipeline first.');st.stop()
f['issue_date']=pd.to_datetime(f.issue_date);f['verify_date']=pd.to_datetime(f.verify_date)
lead=st.sidebar.select_slider('Forecast lead (months)',options=[1,2,3],value=3)
sub=f[f.lead_months==lead].copy()
if sub.empty:st.error('Missing forecast lead');st.stop()
issue=sub.issue_date.max();sub=sub[sub.issue_date==issue]
target=sub.verify_date.iloc[0]
st.write(f'**Issue month:** {issue:%B %Y} · **Target month:** {target:%B %Y} · **India, 0.25° grid**')
if issue < pd.Timestamp.now()-pd.DateOffset(months=2):
    st.error('HISTORICAL / STALE FORECAST: The input dataset has not been updated recently. Do not interpret this as a current drought warning.')
if scores is None:
    st.warning('No independent model evaluation available.')
else:
    sc=scores[scores.lead_months==lead]
    ml=sc[sc.model=='ML_HGB']
    if not ml.empty and ml.rmse_skill_vs_climatology.iloc[0]<=0:
        st.warning('At this lead, the ML model does not outperform historical climatology. Forecast skill is not established.')
a,b,c=st.columns(3)
a.metric('Grid cells',f'{len(sub):,}')
b.metric('Mean forecast SPEI-6',f'{sub.forecast_spei6.mean():.2f}')
c.metric('Cells with predicted SPEI-6 ≤ −1',f'{100*(sub.forecast_spei6<=-1).mean():.1f}%')
fig=px.scatter_map(sub,lat='lat',lon='lon',color='forecast_spei6',color_continuous_scale='RdBu',
    range_color=(-2.5,2.5),zoom=3,center={'lat':22,'lon':79},height=620,
    hover_data=['forecast_spei6','verify_date'])
fig.update_traces(marker={'size':7})
fig.update_layout(map_style='carto-positron',margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(fig,use_container_width=True)
st.caption('Grid-center visualization, not a field-scale forecast. No calibrated event probabilities or uncertainty intervals are claimed.')
if scores is not None:
    st.subheader('Independent 2021–2025 hindcast scores')
    st.dataframe(sc[['model','n','rmse','mae','rmse_skill_vs_climatology']],hide_index=True,use_container_width=True)
st.download_button('Download forecast CSV',sub.to_csv(index=False),file_name=f'India_lead{lead}_SPEI6.csv',mime='text/csv')
st.info('Live website ≠ automatically updated forecasts. New observations must be ingested and the Colab forecasting pipeline rerun; publish refreshed CSVs to this app repository. SPEI and PET methods require independent scientific verification.')
