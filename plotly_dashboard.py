# Import required libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# Set up the app with Bootstrap styles
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
app.title = 'Assignment - Meharaz Hossain (232-25-002)'
# Define the layout of the web application with CSS styles
app.layout = dbc.Container([

#Header text show    
    dbc.Row([
        dbc.Col(html.H1("Text Similarity Calculator", className="text-center mt-5 mb-4"), width={"size": 6, "offset": 3}),
    ]),
#select box for 2 similarity
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='similarity-measure',
            options=[
                {'label': 'Cosine Similarity', 'value': 'cosine'},
                {'label': 'Jaccard Similarity', 'value': 'jaccard'}
            ],
            value='cosine',
            style={'width': '100%', 'marginBottom': '20px'}
        ), width={"size": 6, "offset": 3}),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Checklist(
                id='normalize-checkbox',
                options=[{'label': 'Normalize Text', 'value': 'normalize'}],
                value=[],
                inline=True,
                style={'display': 'none'}
            ),
        ], width={"size": 4, "offset": 4}),
    ]),
    # for text 1 field
    dbc.Row([
        dbc.Col(dcc.Textarea(
            id='text-input-1',
            placeholder='Enter text 1...',
            rows=5,
            style={'width': '100%', 'backgroundColor': '#F0F0F0', 'color': 'black', 'border': 'none', 'borderRadius': '5px', 'padding': '10px'}
        ), width=6),
    # for text 2 field
        dbc.Col(dcc.Textarea(
            id='text-input-2',
            placeholder='Enter text 2...',
            rows=5,
            style={'width': '100%', 'backgroundColor': '#F0F0F0', 'color': 'black', 'border': 'none', 'borderRadius': '5px', 'padding': '10px'}
        ), width=6),
    ]),
    # for chart color
    dbc.Row([
        dbc.Col(html.Div([
            html.H6("Select Color What color You want to show for your graph",style={'width': '100%',  'color': 'black', 'border': 'none', 'borderRadius': '5px', 'padding': '10px'}, className="text-center"),
            html.H6("Note: Every time when you select color. Click the button.",style={'width': '100%', 'font-size':'8px' ,'color': 'red', 'border': 'none', 'borderRadius': '5px', 'padding': '10px'}, className="text-center"),
           
            dcc.Dropdown(
                id='chart-style-dropdown',
                options=[
                    {'label': 'Default(''blue & red'')', 'value': 'default'},
                    {'label': 'Blue & Red', 'value': 'color-scheme-1'},
                    {'label': 'Green & Orange', 'value': 'color-scheme-2'},
                    {'label': 'purple & pink', 'value': 'color-scheme-3'},
                    {'label': 'cyan & magenta', 'value': 'color-scheme-4'},
                ],
                value='default',
                style={'width': '100%', 'marginBottom': '20px'}
            ),
        # for button code
            dbc.Button(
                'Calculate Similarity',
                id='calculate-button',
                n_clicks=0,
                className="btn btn-primary",
                style={'backgroundColor': '#007BFF', 'border': 'none', 'borderRadius': '5px', 'padding': '10px'}
            ),
            html.Div(id='similarity-score', className="text-center mt-4", style={'backgroundColor': '#F0F0F0', 'padding': '10px', 'borderRadius': '5px'}),
        ]), width={"size": 4, "offset": 4, "order": 2}),
    ], className="mb-4 mt-4 text-center"),
    
    # heatmap chart show
    dbc.Row([
        dbc.Col([
            html.H2("Heatmap Chart - Similarity Score", className="text-center"),
            dcc.Graph(id='heatmap', config={'displayModeBar': False}),
        ], width={"size": 6}),
     #Horizontal bar chart show
        dbc.Col([
            html.H2("Horizontal Bar Chart - Similarity Score", className="text-center"),
            dcc.Graph(id='bar-chart', config={'displayModeBar': False}),
        ], width={"size": 6}),
    ]),
    # for group bar chart
    dbc.Row([
        dbc.Col([
            html.H2("Grouped Bar Chart - Similarity Score", className="text-center"),
            dcc.Graph(id='grouped-bar-chart', config={'displayModeBar': False}),
        ], width={"size": 6}),
# for pie chart
        dbc.Col([
            html.H2("Pie Chart - Similarity Score", className="text-center"),
            dcc.Graph(id='pie-chart', config={'displayModeBar': False}),
        ], width={"size": 6}),
    ]),
    # for word count chart
    dbc.Row([
        dbc.Col([
            html.H2("Word Count Bar Chart", className="text-center"),
            dcc.Graph(id='word-count-bar-chart', config={'displayModeBar': False}),
        ], width={"size": 6}),
# for donut chart
        dbc.Col([
            html.H2("Donut Chart - Similarity vs. Dissimilarity", className="text-center"),
            dcc.Graph(id='donut-chart', config={'displayModeBar': False}),
        ], width={"size": 6}),
    ]),
], fluid=True)

# Create a pie chart with adjusted values when similarity is very close to 1
def create_pie_chart(similarity_score):
    if abs(1 - similarity_score) < 0.01:  # Adjust the tolerance as needed
        adjusted_score = 1  # Adjust the value slightly
    else:
        adjusted_score = similarity_score

    pie_chart = px.pie(
        {'Measure': ['Similarity', 'Dissimilarity'], 'Score': [adjusted_score, 1 - adjusted_score]},
        names='Measure',
        values='Score',
        title='Similarity Score (Cosine)'
    )

    return pie_chart

# Define the callback to calculate text similarity and update the charts
@app.callback(
    [Output('heatmap', 'figure'), Output('bar-chart', 'figure'), Output('grouped-bar-chart', 'figure'),
     Output('pie-chart', 'figure'), Output('similarity-score', 'children'), Output('word-count-bar-chart', 'figure'),
     Output('donut-chart', 'figure')],
    [Input('calculate-button', 'n_clicks')],
    [dash.dependencies.State('text-input-1', 'value'),
     dash.dependencies.State('text-input-2', 'value'),
     dash.dependencies.State('similarity-measure', 'value'),
     dash.dependencies.State('normalize-checkbox', 'value'),
     dash.dependencies.State('chart-style-dropdown', 'value')]  # Include the selected chart style
)
def update_charts(n_clicks, text1, text2, similarity_measure, normalize, chart_style):
    # Check if both text areas have content
    if not text1 or not text2:
        empty_fig = px.bar(pd.DataFrame({'Measure': [], 'Score': []}), x='Measure', y='Score')
        return empty_fig, empty_fig, empty_fig, empty_fig, "Similarity Score (Cosine): N/A", empty_fig, empty_fig

    # Normalize text if selected
    if 'normalize' in normalize:
        text1 = text1.lower()
        text2 = text2.lower()

    # Create a list of input texts
    input_texts = [text1, text2]

    # Create a TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Transform the input texts into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(input_texts)

    # Calculate similarity based on the selected measure
    if similarity_measure == 'cosine':
        # Compute cosine similarity
        similarity_score = cosine_similarity(tfidf_matrix)
    elif similarity_measure == 'jaccard':
        # Compute Jaccard similarity using pairwise_distances
        similarity_score = 1 - pairwise_distances(tfidf_matrix, metric='cosine')
    else:
        similarity_score = 0

    # Update the color scheme or style based on the selected chart style
    if chart_style == 'color-scheme-1':
        chart_colors = ['blue', 'red']
    elif chart_style == 'color-scheme-2':
        chart_colors = ['green', 'orange']
    elif chart_style == 'color-scheme-3':
        chart_colors = ['purple', 'pink']
    elif chart_style == 'color-scheme-4':
        chart_colors = ['cyan', 'magenta']
    else:
        chart_colors = ['blue', 'red']  # Default color scheme

    # Create a heatmap with headers
   # Update the color scale based on the selected chart style
    if chart_style == 'color-scheme-1':
        custom_colorscale = [[0.0, 'blue'], [1.0, 'red']]
    elif chart_style == 'color-scheme-2':
        custom_colorscale = [[0.0, 'green'], [1.0, 'orange']]
    elif chart_style == 'color-scheme-3':
        custom_colorscale = [[0.0, 'purple'], [1.0, 'pink']]
    elif chart_style == 'color-scheme-4':
        custom_colorscale = [[0.0, 'cyan'], [1.0, 'magenta']]
    else:
        custom_colorscale = [[0.0, 'blue'], [1.0, 'red']]  # Default color scale

    headers = ['Similarity', 'Dissimilarity']
    heatmap = go.Figure(data=go.Heatmap(
        z=similarity_score,
        x=headers,
        y=headers,
        colorscale=custom_colorscale  # Use the custom color scale here
    ))
    heatmap.update_layout(
        title='Heatmap Chart - Similarity Score',
        xaxis=dict(title=''),
        yaxis=dict(title=''),
    )


    # Create a horizontal bar chart
    bar_chart = px.bar(
        {'Measure': ['Similarity'], 'Score': [similarity_score[0][1]]},
        x='Measure',
        y='Score',
        text='Score',
        title='Horizontal Bar Chart - Similarity Score ({})'.format(similarity_measure.capitalize())
    )
    bar_chart.update_traces(texttemplate='%{text:.2f}', textposition='outside', marker_color=chart_colors[0])

    # Create a grouped bar chart
    grouped_bar_chart = px.bar(
        {'Measure': ['Similarity', 'Dissimilarity'], 'Score': [similarity_score[0][1], 1 - similarity_score[0][1]]},
        x='Measure',
        y='Score',
        text='Score',
        title='Grouped Bar Chart - Similarity Score ({})'.format(similarity_measure.capitalize())
    )
    grouped_bar_chart.update_traces(texttemplate='%{text:.2f}', textposition='outside', marker_color=chart_colors)

    # Create a pie chart with adjusted values
    pie_chart = create_pie_chart(similarity_score[0][1])
    pie_chart.update_traces(marker=dict(colors=chart_colors))

    # Split texts into words
    words_text1 = set(text1.split())
    words_text2 = set(text2.split())

    # Calculate word counts for text 1 and text 2
    word_count_text1 = len(words_text1)
    word_count_text2 = len(words_text2)

    # Calculate the similarity count (count of common words) between the two texts
    similarity_count = len(words_text1.intersection(words_text2))

    # Create a bar chart for word counts
    word_count_bar_chart = px.bar(
        {'Text': ['Text 1', 'Text 2', 'Similarity word'], 'Word Count': [word_count_text1, word_count_text2,similarity_count]},
        x='Text',
        y='Word Count',
        text='Word Count',
        title='Word Count Comparison'
    )
    word_count_bar_chart.update_traces(texttemplate='%{text}', textposition='outside', marker_color=chart_colors)

    # Create a donut chart for similarity and dissimilarity
    donut_chart = px.pie(
        {'Measure': ['Similarity', 'Dissimilarity'], 'Score': [similarity_score[0][1], 1 - similarity_score[0][1]]},
        names='Measure',
        values='Score',
        title='Similarity vs. Dissimilarity'
    )

    donut_chart.update_traces(hole=0.4, marker=dict(colors=chart_colors))  # Set the size of the hole in the middle to create a donut chart

    # Format the similarity score
    similarity_text = "Similarity Score ({}): {:.2f}".format(similarity_measure.capitalize(), similarity_score[0][1])

    return heatmap, bar_chart, grouped_bar_chart, pie_chart, similarity_text, word_count_bar_chart, donut_chart

if __name__ == '__main__':
    app.run_server(debug=True)
