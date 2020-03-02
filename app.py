from dash.dependencies import Input, Output
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px


#--------- Pandas Dataframe
url = "https://raw.githubusercontent.com/SulmanK/PUBG-EDA-Dashboard-Univariate-App/master/data/PUBG_Train.csv"
train_data = pd.read_csv(url, nrows = 15000)


## Scale
scaler = StandardScaler()
X_train_std = scaler.fit_transform(train_data)







############# DBSCAN FUNCTION ########
def dbscan_predict(model, X):
    "Predict function created for DBSCAN"
    nr_samples = X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * -1

    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new








################################ Scatter 3D Plot Function
def scatter3d_cluster(df, x, y, z, code, title):
    scatter  =  px.scatter_3d(df, x = x, y = y, z = z, color  =  code,  
                            color_discrete_sequence = ['#33FF00', '#FF0000'])
    
    scatter.update_layout(title  =  title, title_font  =  dict(size  =  20),
                          scene  =  dict(
                              xaxis  =  dict(
                                  backgroundcolor = "rgb(200, 200, 230)",
                                  gridcolor = "white",
                                  showbackground = True,
                                  zerolinecolor = "white",
                                  nticks = 10, ticks = 'outside',
                                  tick0 = 0, tickwidth  =  4,
                                  title_font  =  dict(size  =  16)),
                              yaxis  =  dict(
                                  backgroundcolor = "rgb(230, 200,230)",
                                  gridcolor = "white",
                                  showbackground = True,
                                  zerolinecolor = "white",
                                  nticks = 10, ticks = 'outside',
                                  tick0 = 0, tickwidth  =  4,
                                  title_font  =  dict(size  =  16)),
                              zaxis  =  dict(
                                  backgroundcolor = "rgb(230, 230,200)",
                                  gridcolor = "white",
                                  showbackground = True,
                                  zerolinecolor = "white",
                                  nticks = 10, ticks = 'outside',
                                  tick0 = 0, tickwidth  =  4,
                                  title_font  =  dict(size  =  16),
                              ),
                          ),
                          width = 750, 
                         )
    return scatter

############################# Bar Plots
def bar_cluster(df, x, code, title):
    bar = px.histogram(df, x,
                 color = code,
                 color_discrete_sequence = ['#33FF00', '#FF0000'] )
    
    bar.update_xaxes(showline = True, linewidth = 1, linecolor = 'black', 
                          mirror = True, gridcolor = 'LightPink', automargin = True, 
                          zeroline = True, zerolinewidth = 2, zerolinecolor = 'LightPink', 
                          ticks = "outside", tickwidth = 2, tickcolor = 'black', ticklen = 10,
                          title = 'Clusters', title_font  =  dict(size  =  16)) 
    bar.update_yaxes(showline = True, linewidth = 2, linecolor = 'black', 
                          mirror = True, gridcolor = 'LightPink',
                          zeroline = True, zerolinewidth = 1, zerolinecolor = 'LightPink', 
                          ticks = "outside", tickwidth = 2, tickcolor = 'black', ticklen = 10,
                          title = 'Count', title_font  =  dict(size  =  16))
    
    
    bar.update_layout(
        title = title,
        title_font  =  dict(size  =  20),
        legend = dict(
            x = 1,
            y = 1,
            traceorder = "normal",
            font = dict(
                family = "sans-serif",
                size = 14,
                color = "black"
            ),
            bgcolor = "#e5ecf6",
            bordercolor = "Black",
            borderwidth = 2
        )
    )
    return bar
    
############################ Caching the Models ##################
### Models
## Kmeans
number_cluster = 2
kmeans = KMeans(n_clusters = number_cluster, init = 'k-means++',
               n_init = 10, random_state = 10).fit(X_train_std)
kmeans_labels = kmeans.labels_

## 3D Plot of Training Data
# Create and modify dataframe for the cluster column
df_X_train_std = pd.DataFrame(X_train_std)
df_X_train_std['Cluster_Kmeans'] = pd.Series(kmeans_labels, index = df_X_train_std.index)

# Rename Cluster label names from k-means
cluster_label_names = {0: "Human", 1: "Hacker"}
df_X_train_std['Cluster_Kmeans_Labels'] = df_X_train_std['Cluster_Kmeans'].map(cluster_label_names) 
        
df_X_train_std.columns = ['Kill Death Ratio', "Headshot Kill Ratio",
                                  'Win Ratio', "Top 10 Ratio",
                                  'Cluster_Kmeans', 'Cluster_Kmeans_Labels']

# Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
K_means_G1_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Kill Death Ratio',
                          y = 'Headshot Kill Ratio', z = 'Win Ratio', code = 'Cluster_Kmeans_Labels', 
                          title = 'Clustering of Kill-Death Ratio, Headshot-Kill Ratio, and Win Ratio')
        
K_means_G2_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Kill Death Ratio', 
                          y = 'Top 10 Ratio', z = 'Win Ratio', code = 'Cluster_Kmeans_Labels', 
                          title = 'Clustering of Kill-Death Ratio, Top 10 Ratio, and Win Ratio')
            
K_means_G3_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Headshot Kill Ratio',
                          y = 'Top 10 Ratio', z = 'Win Ratio', code = 'Cluster_Kmeans_Labels', 
                         title = 'Clustering of Headshot-Kill Ratio, Top 10 Ratio, and Win Ratio')  
            
K_means_G1_Bar = bar_cluster(df = df_X_train_std, x = 'Cluster_Kmeans_Labels',
                        code = 'Cluster_Kmeans_Labels'  , title = 'Cluster Distribution')
                  
K_means_G2_Bar =  bar_cluster(df = df_X_train_std, x = 'Cluster_Kmeans_Labels',
                        code = 'Cluster_Kmeans_Labels'  , title = 'Cluster Distribution')
        
K_means_G3_Bar = bar_cluster(df = df_X_train_std, x = 'Cluster_Kmeans_Labels',
                        code = 'Cluster_Kmeans_Labels'  , title = 'Cluster Distribution')


## DBSCAN
dbscan = DBSCAN( eps = 0.95, min_samples = 8).fit(X_train_std)
dbscan_labels = dbscan.labels_

        
## 3D Plot of Training Data
# Create and modify dataframe for the cluster column
df_X_train_std = pd.DataFrame(X_train_std)
df_X_train_std['Cluster_DBSCAN'] = pd.Series(dbscan_labels, index = df_X_train_std.index)

# Rename Cluster label names from DBSCAN
cluster_label_names = {0: "Human", -1: "Hacker"}
df_X_train_std['Cluster_DBSCAN_Labels'] = df_X_train_std['Cluster_DBSCAN'].map(cluster_label_names) 

df_X_train_std.columns = ['Kill Death Ratio', "Headshot Kill Ratio",
                          'Win Ratio', "Top 10 Ratio",
                          'Cluster_DBSCAN', 'Cluster_DBSCAN_Labels']

# Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
DBSCAN_G1_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Kill Death Ratio',
                          y = 'Headshot Kill Ratio', z = 'Win Ratio', code = 'Cluster_DBSCAN_Labels', 
                          title = 'Clustering of Kill-Death Ratio, Headshot-Kill Ratio, and Win Ratio')
        
DBSCAN_G2_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Kill Death Ratio', 
                          y = 'Top 10 Ratio', z = 'Win Ratio', code = 'Cluster_DBSCAN_Labels', 
                          title = 'Clustering of Kill-Death Ratio, Top 10 Ratio, and Win Ratio')
            
DBSCAN_G3_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Headshot Kill Ratio',
                          y = 'Top 10 Ratio', z = 'Win Ratio', code = 'Cluster_DBSCAN_Labels', 
                         title = 'Clustering of Headshot-Kill Ratio, Top 10 Ratio, and Win Ratio')  
            
DBSCAN_G1_Bar = bar_cluster(df = df_X_train_std, x = 'Cluster_DBSCAN_Labels',
                        code = 'Cluster_DBSCAN_Labels'  , title = 'Cluster Distribution')
                  
DBSCAN_G2_Bar =  bar_cluster(df = df_X_train_std, x = 'Cluster_DBSCAN_Labels',
                        code = 'Cluster_DBSCAN_Labels'  , title = 'Cluster Distribution')
        
DBSCAN_G3_Bar = bar_cluster(df = df_X_train_std, x = 'Cluster_DBSCAN_Labels',
                        code = 'Cluster_DBSCAN_Labels'  , title = 'Cluster Distribution')

## LOF
LOF = LocalOutlierFactor(n_neighbors = 100, contamination = 0.0058, novelty = False).fit(X_train_std)
LOF_labels = LOF.fit_predict(X_train_std)
LOF_predict = LocalOutlierFactor(n_neighbors = 100, contamination = 0.0058, novelty = True).fit(X_train_std)

## 3D Plot of Training Data
# Create and modify dataframe for the cluster column
df_X_train_std = pd.DataFrame(X_train_std)
df_X_train_std['Cluster_LOF'] = pd.Series(LOF_labels, index = df_X_train_std.index)

# Rename Cluster label names from LOF
cluster_label_names = {1: "Human", -1: "Hacker"}
df_X_train_std['Cluster_LOF_Labels'] = df_X_train_std['Cluster_LOF'].map(cluster_label_names)         
df_X_train_std.columns = ['Kill Death Ratio', "Headshot Kill Ratio",
                                  'Win Ratio', "Top 10 Ratio",
                                  'Cluster_LOF', 'Cluster_LOF_Labels']

# Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
LOF_G1_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Kill Death Ratio',
                          y = 'Headshot Kill Ratio', z = 'Win Ratio', code = 'Cluster_LOF_Labels', 
                          title = 'Clustering of Kill-Death Ratio, Headshot-Kill Ratio, and Win Ratio')
        
LOF_G2_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Kill Death Ratio', 
                          y = 'Top 10 Ratio', z = 'Win Ratio', code = 'Cluster_LOF_Labels', 
                          title = 'Clustering of Kill-Death Ratio, Top 10 Ratio, and Win Ratio')
            
LOF_G3_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Headshot Kill Ratio',
                          y = 'Top 10 Ratio', z = 'Win Ratio', code = 'Cluster_LOF_Labels', 
                         title = 'Clustering of Headshot-Kill Ratio, Top 10 Ratio, and Win Ratio')  
            
LOF_G1_Bar = bar_cluster(df = df_X_train_std, x = 'Cluster_LOF_Labels',
                        code = 'Cluster_LOF_Labels'  , title = 'Cluster Distribution')
                  
LOF_G2_Bar =  bar_cluster(df = df_X_train_std, x = 'Cluster_LOF_Labels',
                        code = 'Cluster_LOF_Labels'  , title = 'Cluster Distribution')
        
LOF_G3_Bar = bar_cluster(df = df_X_train_std, x = 'Cluster_LOF_Labels',
                        code = 'Cluster_LOF_Labels'  , title = 'Cluster Distribution')





## EE
EE = EllipticEnvelope(random_state = 10, contamination = 0.0058).fit(X_train_std)
EE_labels = EE.fit_predict(X_train_std)

## 3D Plot of Training Data
# Create and modify dataframe for the cluster column
df_X_train_std = pd.DataFrame(X_train_std)
df_X_train_std['Cluster_EE'] = pd.Series(EE_labels, index = df_X_train_std.index)

# Rename Cluster label names from EE
cluster_label_names = {1: "Human", -1: "Hacker"}
df_X_train_std['Cluster_EE_Labels'] = df_X_train_std['Cluster_EE'].map(cluster_label_names) 
        
df_X_train_std.columns = ['Kill Death Ratio', "Headshot Kill Ratio",
                                  'Win Ratio', "Top 10 Ratio",
                                  'Cluster_EE', 'Cluster_EE_Labels']

# Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
EE_G1_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Kill Death Ratio',
                          y = 'Headshot Kill Ratio', z = 'Win Ratio', code = 'Cluster_EE_Labels', 
                          title = 'Clustering of Kill-Death Ratio, Headshot-Kill Ratio, and Win Ratio')
        
EE_G2_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Kill Death Ratio', 
                          y = 'Top 10 Ratio', z = 'Win Ratio', code = 'Cluster_EE_Labels', 
                          title = 'Clustering of Kill-Death Ratio, Top 10 Ratio, and Win Ratio')
            
EE_G3_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Headshot Kill Ratio',
                          y = 'Top 10 Ratio', z = 'Win Ratio', code = 'Cluster_EE_Labels', 
                         title = 'Clustering of Headshot-Kill Ratio, Top 10 Ratio, and Win Ratio')  
            
EE_G1_Bar = bar_cluster(df = df_X_train_std, x = 'Cluster_EE_Labels',
                        code = 'Cluster_EE_Labels'  , title = 'Cluster Distribution')
                  
EE_G2_Bar =  bar_cluster(df = df_X_train_std, x = 'Cluster_EE_Labels',
                        code = 'Cluster_EE_Labels'  , title = 'Cluster Distribution')
        
EE_G3_Bar = bar_cluster(df = df_X_train_std, x = 'Cluster_EE_Labels',
                        code = 'Cluster_EE_Labels'  , title = 'Cluster Distribution')

## IF
IF = IsolationForest(max_samples = 256 ,random_state = 10, 
                                  contamination = .0058, n_estimators = 500).fit(X_train_std)
IF_anomalies = IF.predict(X_train_std)
IF_labels = IF_anomalies

## 3D Plot of Training Data
# Create and modify dataframe for the cluster column
df_X_train_std = pd.DataFrame(X_train_std)
df_X_train_std['Cluster_IF'] = pd.Series(IF_labels, index = df_X_train_std.index)

# Rename Cluster label names from IF
cluster_label_names = {1: "Human", -1: "Hacker"}
df_X_train_std['Cluster_IF_Labels'] = df_X_train_std['Cluster_IF'].map(cluster_label_names) 
df_X_train_std.columns = ['Kill Death Ratio', "Headshot Kill Ratio",
                                  'Win Ratio', "Top 10 Ratio",
                                  'Cluster_IF', 'Cluster_IF_Labels']

# Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
IF_G1_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Kill Death Ratio',
                          y = 'Headshot Kill Ratio', z = 'Win Ratio', code = 'Cluster_IF_Labels', 
                          title = 'Clustering of Kill-Death Ratio, Headshot-Kill Ratio, and Win Ratio')
        
IF_G2_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Kill Death Ratio', 
                          y = 'Top 10 Ratio', z = 'Win Ratio', code = 'Cluster_IF_Labels', 
                          title = 'Clustering of Kill-Death Ratio, Top 10 Ratio, and Win Ratio')
            
IF_G3_scatter = scatter3d_cluster(df = df_X_train_std , x = 'Headshot Kill Ratio',
                          y = 'Top 10 Ratio', z = 'Win Ratio', code = 'Cluster_IF_Labels', 
                         title = 'Clustering of Headshot-Kill Ratio, Top 10 Ratio, and Win Ratio')  
            
IF_G1_Bar = bar_cluster(df = df_X_train_std, x = 'Cluster_IF_Labels',
                        code = 'Cluster_IF_Labels'  , title = 'Cluster Distribution')
                  
IF_G2_Bar =  bar_cluster(df = df_X_train_std, x = 'Cluster_IF_Labels',
                        code = 'Cluster_IF_Labels'  , title = 'Cluster Distribution')
        
IF_G3_Bar = bar_cluster(df = df_X_train_std, x = 'Cluster_IF_Labels',
                        code = 'Cluster_IF_Labels'  , title = 'Cluster Distribution')


############################### DASHBOARD ###################
## Importing Logo and encoding it
image_filename = 'assets/PUBG_4K_Logo.jpg' 
encoded_image = base64.b64encode(
    open(image_filename, 'rb').read())



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Set server for web app
server = app.server
app.config['suppress_callback_exceptions'] = True


# Define layout of the app 
app.layout = html.Div([
    html.Div(
        [
            html.Div(
                [
                    html.Img(src = 'data:image/png;base64,{}'
                             .format(encoded_image.decode())
                            )
                ], style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
            ),
            html.H1('PUBG Hacker Detection'),
        ], style = {"border-bottom":"2px black ridge"}
    ),
    
    # Insert Markdown for Problem / Background Information   
        html.Div(
            [
                dcc.Markdown(
                    ''' 

Playerunknown's Battleground (PUBG) is a video game, which set the standard for preceding games in the Battle Royale genre. The main goal is to survive at all costs, as you are pitted against other human opponents in a large battlefield. With such fierce competition, some users may resort to cheating to gain an unfair advantage. PUBG uses a standard automation tool that collects the number of complaints on a player and then dispenses a punishment.
 
My approach is to see if we can use Machine Learning, specifically Unsupervised Learning to cluster player game data to address this hacker issue. We'll be targeting the following features based on domain experts - Kill-Death Ratio, Headshot-Kill Ratio, Win Ratio, and Top 10 Ratio.

In terms of performance: IF > EE > LOF > DBSCAN > K-means, more information on dataset details and assumptions can be seen on these EDA dashboards [1](https://pubg-eda-part1-dash.herokuapp.com/),[2](https://pubg-eda-part2-dash.herokuapp.com/).
                    '''
                )
            ], style = {'fontSize': 20, 'font-family': 'Helvetica'}
        ),

    
    html.H3('Clustering Algorithm'),
    dcc.Dropdown(
        id = 'Clustering_Method',
        options = [
            {'label': 'K-means Clustering', 'value': 'K'},
            {'label': 'Density-based Spatial Clustering of Applications with Noise (DBSCAN)', 'value': 'DBSCAN'},
            {'label': 'Local Outlier Factor (LOF)', 'value': 'LOF'},
            {'label': 'Elliptic Envelope (EE)', 'value': 'EE'},
            {'label': 'Isolation Forest (IF)', 'value': 'IF'},
            
        ],
        placeholder = "Select an algorithm",
        value = 'MTL', style = {'width': '60%', 'display': 'inline-block'}
    ), 
    
    html.Div(
        [
            html.Div(
                [
                    html.H4('Kill-Death Ratio (KDR) [0 - 100.0]'),
                    dcc.Input(id='KDR', value= '0', type='number'),
                    html.Div(id='KDR-div')
                ], className = 'six columns'
            ),
            
            html.Div(
                [
                    html.H4('Headshot-Kill Ratio (HKR) [0 - 1.0]'),
                    dcc.Input(id='HKR', value='0', type='number'),
                    html.Div(id='HKR-div')
                ], className = 'six columns'
            ),      
        ], className = 'row'
    ),
    
    html.Div(
        [
            html.Div(
                [
                    html.H4('Win Ratio (WR) [0 - 100.0]'),
                    dcc.Input(id='WR', value='0', type='number'),
                    html.Div(id='WR-div')
                ], className = 'six columns'
            ),
            html.Div(
                [
                    html.H4('Top 10 Ratio (T10R) [0 - 100.0]'),
                    dcc.Input(id='Top10', value='0', type='number'),
                    html.Div(id='Top10-div')
                ], className = 'six columns'
            ),
        ], className = 'row'
    ),
    
    html.Div(
        [
            dcc.Markdown(("""
            Insert KDR, HKR, WR, and T10R values to observe if a person has hackerlike or humanlike behavior.
            """))
        ], style = {'fontSize': 22}  
    ),
    
    html.Div(className='row', children=[
        html.Div(
            [
                html.Pre(id='Hacker-Detection')
            ], style={'backgroundColor': '#80dfff', 'fontSize': 24, "border-top":"2px black ridge" } 
        ), 
        
    ],
            ),
    
    html.Div(
        [
            html.H3('Cluster Visualizations'),
            dcc.Dropdown(
                id = 'Graphs',
                options = 
                [
                    {'label': 'KDR, HKR, and WR', 'value': 'G1'},
                    {'label': 'KDR, T10R, and WR', 'value': 'G2'},
                    {'label': 'HKR, T10R, and WR', 'value': 'G3'},
                ],
                placeholder="Select a plot",
                value='G123', style={'width': '40%', 'display': 'inline-block'}
    )
        ]
    ),
    
    html.Div(className = 'Row', children = 
             [
                 html.Div(
                     [
                         dcc.Graph(id = 'indicator-graphic'),
                     ], style = {'backgroundColor': '#FFFFFF'}, className = 'six columns'
                 ),
                 html.Div(
                     [
                         dcc.Graph(id='Outliers'),
                     ], style = {'backgroundColor': '#FFFFFF'}, className = 'six columns'
                 ),
             ]
            )
], style = {'background-image': 'linear-gradient(#ff99c9, #ffcc66)', 'font-family': 'Helvetica',
         } 
)


@app.callback(
    Output(component_id = 'Hacker-Detection', component_property = 'children'),   
    
    # Clustering Method
    [Input(component_id = 'Clustering_Method', component_property = 'value'),
    
    # KDR
    Input(component_id = 'KDR', component_property = 'value'),

    # HKR
    Input(component_id = 'HKR', component_property = 'value'),

    # WR
    Input(component_id = 'WR', component_property = 'value'),

    # Top 10
    Input(component_id = 'Top10', component_property = 'value')]
)    
    

def callback_Cluster_Detection(Clustering_Method, KDR, HKR, WR, Top10):
    """Function for Callback Cluster Detection to identify if someone is a hacker"""
    input_data = np.array([[ KDR, HKR, WR, Top10 ]])
    predict_data = scaler.transform(input_data)
    
    # Check the callbacks for whichever clustering algorithm was selected and then predict for hackers
    if Clustering_Method == 'K':
        
        predict_labels = kmeans.predict(predict_data)
        
        if predict_labels[0] == 1:
        
            Decision = 'Hacker'
        
        else:
        
            Decision = 'Human'
        
        return Decision

    elif Clustering_Method == 'DBSCAN':
        
        predict_labels = dbscan_predict(dbscan, predict_data)
        
        if predict_labels[0] == -1:
            
            Decision = 'Hacker'
        
        else:
        
            Decision = 'Human'
        
        return Decision
    
    elif Clustering_Method == 'LOF':
        
        predict_labels = LOF_predict.predict(predict_data)
        
        if predict_labels[0] == -1:
            
            Decision = 'Hacker'
        
        else:
            
            Decision = 'Human'
        
        return Decision
    
    elif Clustering_Method == 'EE':
        
        predict_labels = EE.predict(predict_data)
        
        if predict_labels[0] == -1:
            
            Decision = 'Hacker'
        
        else:
        
            Decision = 'Human'
        
        return Decision

    
    elif Clustering_Method == 'IF':
        IF_anomalies = IF.predict(predict_data)
        predict_labels = IF_anomalies
        
        if predict_labels[0] == -1:
        
            Decision = 'Hacker'
        
        else:
        
            Decision = 'Human'
        
        return Decision
    
    else:
        
        return 'Choose between K-means Clustering, DBSCAN, LOF, EE, or IF'


@app.callback(
    Output('indicator-graphic', 'figure'),
    
    # Clustering Method
    [Input(component_id = 'Clustering_Method', component_property = 'value'),
    
    # Graphs
    Input(component_id = 'Graphs', component_property = 'value'),]
)

    
def callback_Clusters(Clustering_Method, Graphs):
    """Returns 3D Plots of the Clusters based on which method was selected"""
    
    if Clustering_Method == 'K' :

        # Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
        if Graphs == 'G1':
            
            return K_means_G1_scatter
        
        elif Graphs == 'G2':

            return K_means_G2_scatter
        
        elif Graphs == 'G3':
            
            return K_means_G3_scatter
        
        else:
         
            return 'Select a plot!'

    elif Clustering_Method == 'DBSCAN':
        
        # Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
        if Graphs == 'G1':

            return DBSCAN_G1_scatter
        
        elif Graphs == 'G2':
            
            return DBSCAN_G2_scatter
        
        elif Graphs == 'G3':
            
            return DBSCAN_G3_scatter
        
        else:
            return 'Select a plot!'
    
    elif Clustering_Method == 'LOF':
                
        # Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
        if Graphs == 'G1':
            
            return LOF_G1_scatter
        
        elif Graphs == 'G2':
            
            return LOF_G2_scatter
        
        elif Graphs == 'G3':
            
            return LOF_G3_scatter
        
        else:
            return 'Select a plot!'

    elif Clustering_Method == 'EE':
        
        # Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
        if Graphs == 'G1':
            
            return EE_G1_scatter
        
        elif Graphs == 'G2':
            
            return EE_G2_scatter
        
        elif Graphs == 'G3':
            
            return EE_G3_scatter
        
        else:
            
            return 'Select a plot!'

    elif Clustering_Method == 'IF':

        # Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
        if Graphs == 'G1':
            
            return IF_G1_scatter
        
        elif Graphs == 'G2':
            
            return IF_G2_scatter
        
        elif Graphs == 'G3':
            
            return IF_G3_scatter
        
        else:
            
            return 'Select a plot!'
        
    else:
     
        return 'Choose between K-means Clustering, DBSCAN, LOF, EE, or IF'

    
    
    
@app.callback(
    Output('Outliers', 'figure'),
    
    # Clustering Method
    [Input(component_id = 'Clustering_Method', component_property = 'value'),
    
    # Graphs
    Input(component_id = 'Graphs', component_property = 'value'),]
)
def callback_Bar(Clustering_Method, Graphs):

    
    if Clustering_Method == 'K' :


        # Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
        if Graphs == 'G1':
            
            return K_means_G1_Bar
        
        elif Graphs == 'G2':
        
            return K_means_G2_Bar
        
        elif Graphs == 'G3':
            
            return K_means_G3_Bar
        
        else:
            return 'Select a plot'   

        

    elif Clustering_Method == 'DBSCAN':
        
        # Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
        if Graphs == 'G1':

            return DBSCAN_G1_Bar
        
        elif Graphs == 'G2':
            
            return DBSCAN_G2_Bar
        
        elif Graphs == 'G3':
            
            return DBSCAN_G3_Bar
        
        else:
            return 'Select a plot!'
    
    elif Clustering_Method == 'LOF':
                
        # Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
        if Graphs == 'G1':
            
            return LOF_G1_Bar
        
        elif Graphs == 'G2':
            
            return LOF_G2_Bar
        
        elif Graphs == 'G3':
            
            return LOF_G3_Bar
        
        else:
            return 'Select a plot!'
        
    elif Clustering_Method == 'EE':
        
        # Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
        if Graphs == 'G1':
            
            return EE_G1_Bar
        
        elif Graphs == 'G2':
            
            return EE_G2_Bar
        
        elif Graphs == 'G3':
            
            return EE_G3_Bar
        
        else:
         
            return 'Select a plot'

    elif Clustering_Method == 'IF':

        # Plots of Win Ratio, Kill Death Ratio, Headshott KIll Ratio
        if Graphs == 'G1':
            
            return IF_G1_Bar
        
        elif Graphs == 'G2':
            
            return IF_G2_Bar
        
        elif Graphs == 'G3':
            
            return IF_G3_Bar
        
        else:
        
            return 'Select a plot'
    
    else:
        
        return 'Choose between K-means Clustering, DBSCAN, LOF, EE, or IF'



if __name__ == '__main__':
    app.run_server(debug = True)