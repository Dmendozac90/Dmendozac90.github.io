---
layout: post
title:  "Data Scraping and Visualization with Dash"
date:   2022-04-02 -
description: "The internet is a rich source of data waiting to be explored for research or personal interests. This data is encountered in a variety of formats which may not always be amenable to analysis and modeling. It is often the case that more meticulous methods are required to gather and structure this data. Fortunately, there are several libraries in Python that are intended to solve this issue and thus unlock these datasets to anyone who takes time to explore these libraries. This post demonstrates a quick use of these python libraries as well as a brief introduction to dynamic data visualization."
categories: Python Web_Scrapping Selenium Dash API Web_App Plotly
html_url: /assets/img/Web Scraping/Web Scraping.jpg
---

**Outline**
-   [Introduction](#introduction)
-   [ Interacting with APIs Web Scraping](#interacting-with-apis-and-web-scraping)
-   [Data Visualization with Dash](#data-visualization-with-dash)
-   [Conclusion](#conclusion)


## Introduction

Accessing data is the necessary first step for any data-related project and is often a trivial process because the data exists in a format that is easy to load like a csv file, JSON object, or other text file formats. A more difficult scenario arises when that data is not in a previously mentioned format. What if the data is provided by a website API, exists within a HTML table or other HTML components or attributes, or is executed as a client-side script? When this is the case, more robust methods are required. 

Fortunately, python contains various libraries that are designed to address these complex scenarios and allow users to successfully retreive multiple data formats encountered on the internet. Some functionality of these python libraries will be demonstrated using **[fiscal data](https://fiscaldata.treasury.gov/)** provided by the United States Department of the Treasury's website. 

After the data has been successfully retreived, it is important to become familiarized with the structure of the data especially if it is the first time interacting with it. This is an easy task if one is dealing with a single dataset, but what if multiple datasets are encountered and these datasets can be further parsed by selecting a subset of features? It would be highly beneficial to have method that can quickly provide dynamic visual and tabular displays of the data as it may give one insight into any preprocessing requirements that the data will need during data cleanining and preparation, or interesting trends or observations that may be further analyzed when building models. For this purpose, a dynamic web application will be constructed using **[Dash](https://plotly.com/)** in which all available datasets will incorporated and will be made available for display using a dynamic scatterplot and table. 


## Interacting with APIs and Web Scraping 

Before exploring the topic of web scraping, please know that many websites do not allow users to retrieve data via scraping for various reasons and thus one should respect website policy. Many websites have public APIs that facilitate access to available data and thus deter the need for web scraping. API stands for Application Programming Interface and is a mechanism that enables two software components to communicate with one another using a set of well-defined protocols. In this case, python will be used to send a request to a server and the server will return data as a response to the executed request. 

The United States Department of the Treasury has a well-documented public **[API](https://fiscaldata.treasury.gov/api-documentation/)**. This is where one will find information on the structure of valid requests a user can make, available datasets, and sufficient examples to get started. Interacting with an API is typically done using the **`requests`** library. The **`requests`** library facilitates sending HTTP requests for users and is extremely easy to use, especially when interacting with APIs. Notice that the United States Department of the Treasury's API documentation specifies that the components of a full API request are comprised of a **Base URL**, an **Endpoint**, and optional **Parameters and Filters**. Using this information, a valid request can be made as follows:

```python
import requests
BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
endpoint = "/v2/accounting/od/debt_to_penny"
URL = BASE_URL + endpoint
resp = requests.get(URL)
resp.status_code

200
```

The status code $$\small 200$$ indicates that a successful request has been made. This can be compared to an invalid API request and its resulting status code.

```python
BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
#change endpoint to ensure invalid API request
endpoint = "this_is_not_valid"
URL = BASE_URL + endpoint
resp = requests.get(URL)
resp.status_code

404
```

The status code $$\small 404$$ means that the server could not find the request that was made. Checking the status code is a quick way to check if one has made a proper API request. When encountering a $$\small 404$$ status code, refer to the  API documentation to determine the source of the error. According to the API documentation site, if a valid API request is made, the server returns a JSON response. The **`Request`** object that is called **`resp`** has a method that decodes JSON data and returns a Python **`dict`**. If this method is called, the following is returned (the majority of the dictionary is omitted for demonstration purposes).


```python
resp.json()

{'data': [{'record_date': '1993-04-01',
   'debt_held_public_amt': 'null',
   'intragov_hold_amt': 'null',
   'tot_pub_debt_out_amt': '4225873987843.44',
   'src_line_nbr': '1',
   'record_fiscal_year': '1993',
   'record_fiscal_quarter': '3',
   'record_calendar_year': '1993',
   'record_calendar_quarter': '2',
   'record_calendar_month': '04',
   'record_calendar_day': '01'}, ...

   'links': {'self': '&page%5Bnumber%5D=1&page%5Bsize%5D=100',
  'first': '&page%5Bnumber%5D=1&page%5Bsize%5D=100',
  'prev': None,
  'next': '&page%5Bnumber%5D=2&page%5Bsize%5D=100',
  'last': '&page%5Bnumber%5D=73&page%5Bsize%5D=100'}}
```

Unless one is after a specific dataset described by a single endpoint, then the request library provides sufficient functionality to retrieve the desired dataset. But what if more than one dataset is required, or if one wants to explore other available datasets to determine if they are appropriate for a current or future analysis? This task could be accomplished manually by going through the API documentation and exploring all available datasets and making a table of records for future reference. Looking at the API documentation website, such a table already exists. This table contains a link to the dataset, the dataset name, the corresponding endpoint for an API request, and a brief description of the dataset. Having immediate access to this table without having to open this webpage and scroll through all the entries might be beneficial for making future API requests. 

Retrieving the information in this table will require more functionality that what is provided by the **`requests`** library. This will be made evident by inspecting either the content or text methods of the new **`Response`** object. Instead of passing in the base URL and endpoint into the **`requests`** **`get`** method, the API documentation URL will be passed.  

```python
URL = "https://fiscaldata.treasury.gov/api-documentation/"
resp = requests.get(URL)
resp.content

'Endpoint Description</th></tr></thead><tbody></tbody></table></div></div>'
```
Please note that the output above omits most of the contents returned by the content method to illustrate that **`requests`** does not have sufficient functionality for the aforementioned task. The information of the table should be contained within the table body component shown as <tbody></tbody>; however, no data is observed. This can be verified by using the browser's developer tools. The developer tools can be accessed by running the command **`Ctrl+Shift+I`** if one is using Google Chrome as the web browser. The developer tools can be used to locate any element within the webpage. The corresponding table body component is displayed in the image below.

<img src="/assets/img/Web Scraping/tbody.png" width="100%">.

Notice that hovering over the table body component highlights the corresponding table on the webpage. Although the content method shows that there is no information within the table, the developer tools contradict this finding. The reason for this is because dynamic webpages require time for client-side scripts to be executed and thus all information of the webpage will not be immediately accessible. This can be verified by using the developer tools and pausing the website from fully loading. The image below displays the table of interest. Note that it is empty as was previously observed by the content that was returned by the **`Requests`** object content method. 

<img src="/assets/img/Web Scraping/blank table.png"  width="100%" />

A python library that allows for dynamic webpages to be fully loaded before webpage content can be retrieved is Selenium. Selenium has a wide array of tools, and it is encouraged that users fully explore their documentation as the functionality displayed in this post is just a minor component of what Selenium provides. To retrieve the information that is wanted, the following script is executed.

```python
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

URL = "https://fiscaldata.treasury.gov/api-documentation/#methods"
PATH = "Insert Your Driver Path Here"

driver = webdriver.Chrome(PATH)
driver.get(URL)

tbody = WebDriverWait(driver, 10).until(
    lambda x: x.find_element_by_tag_name("tbody"))
  
tbody

<selenium.webdriver.remote.webelement.WebElement (session="cee1504ffea3eb80c6b9b56639777066", 
element="e63e206f-fe76-4824-9a32-5de96202dc1f")>
```

This gives sufficient time to retrieve the information within the table body component. Using the developer tool, one can observe that the desired information is found within the table row component. Each table row contains four table data (td) components that contain the entries of the data table. These are retrieved as follows.

```python
td = tbody.find_elements_by_tag_name('td')
a = tbody.find_elements_by_tag_name('a')
#Data dictionary
data = {
    "Dataset": [],
    "Table Name": [],
    "Endpoints": [],
    "Description": []
}
    
for i in range(len(td)):
    if i % 4 == 0:
        data["Dataset"].append(a[int(i / 4)].get_attribute("href"))
    elif i % 4 == 1:
        data["Table Name"].append(td[i].text)
    elif i % 4 == 2:
        data["Endpoints"].append(td[i].text)
    elif i % 4 == 3:
        data["Description"].append(td[i].text)
```

The entries are arranged in a table and displayed below. 

<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="/assets/img/Web Scraping/Endpoints.html" height="525" width="100%"></iframe>

Although the information displayed in the API documentation's webpage has been retrieved, additional work is required. Notice that only ten entries are displayed in the table by default and thus only these ten entries were retrieved when the code block above was executed. Notice that at the bottom of the table, a dropdown menu and a button are included that allow access to the remaining entries of the table. Thus, interaction with these components is required. Selenium gives users the necessary tools to interact with these components and thus one can develop methods to retrieve information that would otherwise require manual interaction with the webpage.

The developer tools will be required to determine the location of the dropdown menu and/or the next button. This highlights a major phase of retrieving data from a webpage. When a user initially attempts to retrieve data from a webpage, they will dedicate a large portion of time to understanding the various components that comprise the webpage and the information that they contain. Once a user fully understands the layout of the webpage, they can develop a succinct strategy to retrieve this information. After exploring the webpage with the developer tools, the dropdown is located and is displayed in the image below. 

<img src="/assets/img/Web Scraping/button.png" width="100%">

To interact with the dropdown, its **`XPath`** will be required. This can be easily accessed by right-clicking the component, hovering to the "Copy" option, and selecting "Copy XPath". This **`XPath`** only corresponds to the button; however, the **`XPath`**  that corresponds to the dropdown is also required. Again, this is located using the developer tools and once it is found, all the necessary elements to retrieve all available endpoints are obtained. The code below retrieves the entire table of endpoints. 

```python
import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

URL = "https://fiscaldata.treasury.gov/api-documentation/#methods"
PATH = r'Your Path Here'
button_xpath = '//*[@id="list-of-endpoints-table"]/div/div[2]/div[2]/div[1]/button'
select_xpath = '//*[@id="rows-per-page"]/div[2]/ul/li[5]'

driver = webdriver.Chrome(PATH)
driver.get(URL)

tbody = WebDriverWait(driver, 10).until(
    lambda x: x.find_element_by_tag_name("tbody"))

button = WebDriverWait(driver, 10).until(
    lambda x: x.find_element_by_xpath(button_xpath))

select = WebDriverWait(driver, 10).until(
    lambda x: x.find_element_by_xpath(select_xpath))

button.click()
time.sleep(1.5)
select.click()
time.sleep(2.5)

td = tbody.find_elements_by_tag_name('td')
a = tbody.find_elements_by_tag_name('a')

#Data dictionary
data = {
    "Dataset": [],
    "Table Name": [],
    "Endpoints": [],
    "Description": []
}

for i in range(len(td)):
    if i % 4 == 0:
        data["Dataset"].append(a[int(i/4)].get_attribute("href"))
    elif i % 4 == 1:
        data["Table Name"].append(td[i].text)
    elif i % 4 == 2:
        data["Endpoints"].append(td[i].text)
    elif i % 4 == 3:
        data["Description"].append(td[i].text)
```

Now that the table information was been retrieved, generating valid API requests is greatly simplified. All that is required is the base URL and any of the endpoints stored in the data dictionary. This is demonstrated below.

```python
BASE_URL + data["Endpoints"][0], \
BASE_URL + data["Endpoints"][15], \
BASE_URL + data["Endpoints"][39]

('https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/debt/tror/data_act_compliance',
 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/statement_net_cost',
 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/mts/mts_table_6e')
```

If more information about the dataset is required, the dataset URL can be quickly retrieved as follows 

```python
print(data["Dataset"][40])

https://fiscaldata.treasury.gov/datasets/monthly-treasury-statement/
```

Selenium truly provides some great tools for retrieving data from dynamic and static web pages. With a small understanding of some of Selenium's components and the API documentation webpage layout, a general method for obtaining all the valid API requests was formulated as well as additional information about available datasets. Again, this is just a small preview of what can be accomplished using Selenium and those interested in other applications are encouraged to explore this library. 

Now that all valid API requests can be made, all datasets provided by the United States Department of the Treasury can easily be accessed. Naturally, one might ponder how this data is structured or if it possesses any characteristics that merit further analysis. A great way to visualize a large set of distinct datasets is using a dynamic web application that has all the desired functionalities. The following section explores this task using Dash.

## Data Visualization with Dash

Dash provides the necessary tools to quickly build dynamic and interactive web applications that data scientists and engineers can use to communicate findings efficiently. Dash apps possess components that allow the user to define how the application will look like and the interactivity of the application. The layout property is usually defined as a tree of components and the callbacks are functions that are automatically called by Dash when an input is changed. The example below demonstrates the use of the layout property and callback functionality.

```python
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd


app = dash.Dash(__name__)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

#Specify app layout
app.layout = html.Div([
    html.H1(children='Hello Dash, this is a sample!!'),
        dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        min=df['year'].min(),
        max=df['year'].max(),
        step=None,
        value=df['year'].min(),
        marks={str(year): str(year) for year in df['year'].unique()},
        id='year-slider'
    )
])

#Add callbacks
@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):
    filtered_df = df[df.year == selected_year]

    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
                     size="pop", color="continent", hover_name="country",
                     log_x=True, size_max=55)

    fig.update_layout(transition_duration=500)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
```

<img src="/assets/img/Web Scraping/Dash_Sample.gif" width="100%">.

Notice that below the wrapper **`app.callback`** there is an **`Output`** and an associated **`Input`**. The callback is initiated when there is a change to the input value which is linked to the value of the slider component. Thus whenever the slider value changes, Dash automatically updates the Output which is a graph. In this example, the slider passes a value corresponding to a specific year. The function underneath the wrapper filters the dataset according to the user-specified year and then creates a Plotly Express trace object that is then returned to the dcc.Graph in the app's layout. 

To observe the datasets retrieved by the API requests, a scatter plot and table will be designed along with the interactive componenets that will controll the functionality of the graph and table. The code below demonstrates the construction of such web application.

```python
import os
import json
import requests
import plotly.graph_objects as go
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_table
from dash.dependencies import Input, Output 

with open("json_endpoints.json", "r") as o:
    endpoint_data = json.load(o)

endpoints_vals = endpoint_data["Endpoints"]
table_names = endpoint_data["Table Name"]

app = dash.Dash(
    external_stylesheets=[dbc.themes.FLATLY],
    prevent_initial_callbacks=True
)

select_data = dbc.Card(
    html.Div(
        children=[
            dbc.Label("Endpoints List", className="pl-4 pt-1"),
            dcc.Dropdown(
                id="endpoints",
                options=[{"label": name, "value": endpoint} for name, endpoint in zip(table_names, endpoints_vals)],
                value=endpoints_vals[0],
                className="pl-4 pr-4 mb-2 outline-primary",
                searchable=True
            ),
            dbc.Label("Fields", className="pl-4 pt-1"),
            dcc.Dropdown(
                id="fields",
                className="pl-4 pr-4 mb-2 outline-primary",
            ),
            dbc.Label("Record Length", className="pl-4 pt-1"),
            html.Div(
                children=[
                    daq.Slider(id="size", size=350, min=1, max=10000, color="#74d6c3", value=100,
                    handleLabel=dict(color="rgb(44,62,80)", showCurrentValue=True, label="Value"))
                ],
                className="ml-4 mb-2 mt-5"
            ),
            html.Div(
                children=[
                    dbc.Button("Plot Field", id="plot graph", color="white", className="btn-outline-success")
                ],
                className="d-grid col-4 mx-auto mb-2 mt-3"
            ),
        ], 
        className="card text-primary outline-primary border border-4"
    ),
    className="border border-primary"
)

data_table_info = dbc.Card(
    html.Div(
        children=[
            dbc.Label("Fields", className="pl-4 pt-1"),
            dcc.Dropdown(
                id="fields_2",
                className="pl-4 pr-4 mb-2 outline-primary",
                multi=True
            ),
            dbc.Label("Record Length", className="pl-4 pt-1"),
            html.Div(
                children=[
                    daq.Slider(id="size_2", size=350, min=1, max=10000, color="#74d6c3", value=100,
                    handleLabel=dict(color="rgb(44,62,80)", showCurrentValue=True, label="Value"))
                ],
                className="ml-4 mb-2 mt-5"
            ),
        ],
        className="border border-primary rounded rounded-5 p-1"
    )
)

data_table = dbc.Card(
    html.Div(
        children=[
            dash_table.DataTable(id="table", 
            style_header=dict(
                backgroundColor="rgba(149,165,166,0.5)", 
                color="#2c3e50", 
                fontWeight="bold", 
                textAlign="center", 
                fontFamily="arial",
                border="1px solid black"
            ),
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "rgba(149,165,166,0.2)",
                }
            ],
            style_data=dict(textAlign="center", fontFamily="arial")
            ),
            html.Div(id="test")
        ],
        className="border border-primary rounded rounded-5 p-1",
        style=dict(overflow="scroll")),
)

app.layout=(
    dbc.Container(
        [
            html.H1(children="US Fiscal Data", className="text-white bg-primary border border-secondary \
            rounded pl-4 pt-2 pb-2 pr-1 mt-2"),
            html.Hr(className="bg-primary mb-4 mr-1", style=dict(height="1px")),
            dbc.Row(
                children = [
                    dbc.Col(select_data, md=4),
                    dbc.Col(html.Div(dcc.Graph(id="Fiscal Data Graph", className="p-1"), 
                    className="border border-3 border border-primary rounded rounded-5 p-1"), md=8)    
                ],
                className="p-1"
            ),
            html.Hr(className="bg-secondary mt-5 mr-1", style=dict(height="1px")),
            dbc.Row(
                children = [
                    dbc.Col(data_table_info, md=4),
                    dbc.Col(data_table, md=8),
                ],
                className="p-1"
            ),
        ],
        fluid=True
    )
)

@app.callback(
    Output("fields", "options"),
    Input("endpoints", "value"),
    prevent_initial_call=True   
)
def retrieve_fields(endpoint):
    NUMERICAL_TYPES = [
        "DATE",
        "DAY",
        "MONTH",
        "QUARTER",
        "YEAR",
        "NUMBER",
        "CURRENCY",
        "PERCENTAGE"
    ]
    numeric_fields = []
    if endpoint is not None:
        BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
        GET = BASE_URL + endpoint
        resp = requests.get(GET)
        for key, value in resp.json()["meta"]["dataTypes"].items():
            if value in NUMERICAL_TYPES:
                numeric_fields.append(key)
        dropdown_fields = [{"label": field, "value": field} for field in numeric_fields]
        return dropdown_fields

@app.callback(
    Output("Fiscal Data Graph", "figure"),
    [
        Input("endpoints", "value"),
        Input("fields", "value"),
        Input("size", "value"),
        Input("plot graph", "n_clicks")
    ],
    prevent_initial_call=True
)
def plot_data(endpoint, field, size, clicked):
    fig = {}
    BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
    FIELDS_URL = "?fields="
    FILTER_URL = "&filter="
    PAGE_NUM = "&page[number]=1"
    PAGE_SIZE = "&page[size]=" + str(size)
    if clicked is not None and field is not None:
        GET = BASE_URL + endpoint + FIELDS_URL + "record_date" + "," + field + PAGE_NUM + PAGE_SIZE
        data = requests.get(GET).json()
        date, ys = [], []
        for element in data['data']:
            for key in element.keys():
                if key == "record_date":
                    date.append(element[key])
                else:
                    ys.append(element[key])
        
        fig = go.Figure()
        trace = go.Scatter(
            name=field,
            x=date,
            y=ys,
            mode="markers",
            marker=dict(symbol="circle-open", color="rgb(44,62,80)")
        )
        fig.add_trace(trace)
        fig.update_yaxes(type="linear", title=dict(text=f"{field}", font=dict(color="rgb(44,62,80)")))
        fig.update_xaxes(title=dict(text="Date", font=dict(color="rgb(44,62,80)")))
    return fig

@app.callback(
    Output("fields_2", "options"),
    Input("endpoints", "value"),
    prevent_initial_call=True   
)
def retrieve_fields(endpoint):
    NUMERICAL_TYPES = [
        "DATE",
        "DAY",
        "MONTH",
        "QUARTER",
        "YEAR",
        "NUMBER",
        "CURRENCY",
        "PERCENTAGE"
    ]
    numeric_fields = []
    if endpoint is not None:
        BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
        GET = BASE_URL + endpoint
        resp = requests.get(GET)
        for key, value in resp.json()["meta"]["dataTypes"].items():
            if value in NUMERICAL_TYPES:
                numeric_fields.append(key)
        dropdown_fields = [{"label": field, "value": field} for field in numeric_fields]
        return dropdown_fields

@app.callback(
    Output("table", "columns"),
    [
        Input("endpoints", "value"),
        Input("fields_2", "value"),
    ],
    prevent_initial_callback=True
)
def populate_columns(endpoint, field,):
    BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
    FIELDS_URL = "?fields="
    FILTER_URL = "&filter="
    PAGE_NUM = "&page[number]=1"
    PAGE_SIZE = "&page[size]=1"
    if field is not None:
        fields = ",".join(field)
        GET = BASE_URL + endpoint + FIELDS_URL + fields + PAGE_NUM + PAGE_SIZE
        columns = requests.get(GET).json()["meta"]["labels"].keys()
        column_data = [{"name": i, "id": i} for i in columns]
        return column_data

@app.callback(
    Output("table", "data"),
    [
        Input("endpoints", "value"),
        Input("fields_2", "value"),
        Input("size_2", "value")
    ],
    prevent_initial_callback=True
)
def populate_dash_table(endpoint, field, size,):
    BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
    FIELDS_URL = "?fields="
    FILTER_URL = "&filter="
    PAGE_NUM = "&page[number]=1"
    PAGE_SIZE = "&page[size]=" + str(size)
    if field is not None:
        fields = ",".join(field)
        GET = BASE_URL + endpoint + FIELDS_URL + fields + PAGE_NUM + PAGE_SIZE
        data = requests.get(GET).json()['data']
        data_dict={}
        fields = data[0].keys()
        for field in fields:
            data_dict[field] = {}
        for i, element in enumerate(data):
            for field in fields:
                data_dict[field][i] = element[field]
        return data

if __name__ == "__main__":
    app.run_server(debug=True)
```

The functionality is demonstrated below.

<img src="/assets/img/Web Scraping/Fiscal App.gif" width="100%">.

This application has two dropdown menus that contain the endpoint and attribute that is to be displayed on the graph. The user can specify the number of datapoints to be included with a slider. After these components are set to the desired values, the data is displayed on the scatter plot by clicking on the button labeled as "Plot Field". The table is controlled in a similar manner. Note that the dropdown menu accepts multiple selections which correspond to the columns on the table. This functionality is specified with the five callbacks shown in the code above. The remaining code specifies the layout of the graph and the styling that was used for the aesthetics of the application.  

This is a great method to quickly visualize the data and identify any interesting features. This application can quickly be modified to incorporate different analyses and allow users that may not be familiar using python, to quickly observe the results. This is one of the great features of building web applications with Dash. 


## Conclusion

Data is abundant on the internet. Developing the skills to retrieve this data and quickly display it can be greatly beneficial for any organization. The combination of the **`requests`**, **`Selenium`**, and **`Dash`** libraries allows for python users to interact with the data and create powerful web applications. Hopefully users that are unfamiliar with interacting with web APIs, web scraping, and building web applications are motivated by the content in this post to explore this topic and develop their skills. If you have any questions about these libraries or the code that was displayed, please feel free to contact me at my email! 