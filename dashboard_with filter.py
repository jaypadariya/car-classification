# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:35:22 2021

@author: JPadariya
"""

###########dashboard for datashowing with flask############

#dash show drop down 
import dash
import dash_html_components as html
import dash_core_components as dcc
df2=pd.read_csv(r"C:\Users\jpadariya\Downloads\assignment\assignment\res\training_file.csv",encoding= 'unicode_escape')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df2.columns =[column.replace(" ", "_") for column in df2.columns]

df2.query("petrol == True", inplace = True)
petrol=df2["car_model"]


df2.query("cng == True", inplace = True)
cng=df2["car_model"]

df2.query("diesel == True", inplace = True)
diesel=df2["car_model"]


df2.query("roof == True", inplace = True)
sunroof=df2["car_model"]

df2.query("climate_control == True", inplace = True)
climate_control=df2["car_model"]


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'petrol', 'value': '{}'.format(petrol)},
            {'label': 'cng', 'value': '{}'.format(cng)},
            {'label': 'diesel', 'value': '{}'.format(diesel)},
            {'label': 'sunroof', 'value': '{}'.format(sunroof)},
            {'label': 'climate_control', 'value': '{}'.format(climate_control)}
        ],
        value='{}'.format(petrol)
    ),
    html.Div(id='dd-output-container')
])


@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False) 