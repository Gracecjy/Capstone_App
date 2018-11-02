
# coding: utf-8


import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect
import requests
from fbprophet import Prophet
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar

import bokeh
from bokeh.plotting import figure
from bokeh.io import show,output_notebook
from bokeh.embed import components
from bokeh.models import HoverTool
bv = bokeh.__version__


# In[ ]:

app = Flask(__name__)
app.vars={}


# In[ ]:

@app.route('/')
def main():
    return redirect('/index')


# In[ ]:

@app.route('/index',methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        app.vars['start_date'] = request.form['start_date']
        return redirect('/graph')


# In[ ]:

@app.route('/graph', methods=['GET', 'POST'])
def graph():
    #get data from local store_info.csv
    raw_df = pd.read_csv('store_info.csv')
    raw_df['date'] = raw_df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    start_date = datetime.datetime.strptime(app.vars['start_date'],'%Y-%m-%d')
    df = raw_df[raw_df['date'] < start_date]

    #create holiday calendar for prophet
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2014-01-31', end='2015-12-09').to_pydatetime()
    holiday = pd.DataFrame(holidays)
    holiday.columns = ['ds']
    holiday['holiday'] = np.array('federal_holiday')
    
    
    #predict using prophet and plot graphs
    def prophet_predict(df,start_date=start_date):
        df.columns = ['ds','y']
        model = Prophet(interval_width = 0.80, holidays = holiday,yearly_seasonality=True, daily_seasonality=True )
        model.fit(df)
        future_dates = model.make_future_dataframe(periods = 14)
        forecast = model.predict(future_dates)
        pred = forecast[['ds','yhat']]
        pred.columns = ['date','prediction']
        mask = pred['date'] > start_date
        new_pred = pred.loc[mask]
        return new_pred
    

    def bokeh_plot(pred,item,color):
        hover = HoverTool(tooltips = [("Value", "$y{int}")],  mode='vline')
        tools = [hover]
        p = figure(plot_width=800, plot_height=500,
               title="Two-Week Projection for {} start from {}".format(item, app.vars['start_date']), x_axis_type="datetime",tools=tools)
        p.line(pred['date'], pred['prediction'], line_width=3,
               line_color=color, legend=item)
        p.xaxis.axis_label = "Date"
        p.yaxis.axis_label = "Projection"
        p.xaxis.axis_label_text_font_style = 'bold'
        p.yaxis.axis_label_text_font_style = 'bold'
        p.xaxis.bounds = (pred.date.iloc[-1], pred.date.iloc[0])
        script, div = components(p)
        return script, div


    sales_df = df[['date','Net_Sales']]
    pred = prophet_predict(sales_df)
    script_sales, div_sales = bokeh_plot(pred,'Net Sales',"#658b33")
        

    trans_df = df[['date','Trans']]
    pred = prophet_predict(trans_df)
    script_trans, div_trans = bokeh_plot(pred,'Transactions',"#949150")
        

    Visits_df = df[['date','Visits']]
    pred = prophet_predict(Visits_df)
    script_visits, div_visits = bokeh_plot(pred,'Visits',"#dbc69d")

    return render_template('graph.html', bv=bv, period=app.vars['start_date'], 
                           script_sales=script_sales, div_sales=div_sales,
                           script_trans=script_trans, div_trans=div_trans,
                           script_visits=script_visits,div_visits=div_visits
                          )


# In[ ]:

if __name__ == '__main__':
    
    #run the app
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)





