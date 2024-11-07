import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# Leer los datos desde el archivo Excel
df = pd.read_excel("ClientesBancoEjemplo.xlsx")

# Cargar los datos y construir el modelo predictivo
X = df[['Edad', 'IngresoAnual', 'VisitasWeb', 'ClicksPublicidad', 'TiempoEnSitio']]
y = df['AbreCuenta']

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Entrenar un modelo Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Predicción y evaluación del modelo
y_pred = modelo_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Guardar el modelo entrenado para futuras predicciones
joblib.dump(modelo_rf, 'modelo_random_forest_apertura_cuentas.pkl')

# Crear la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['No Abre', 'Abre'], columns=['No Abre', 'Abre'])

# Filtrar solo los clientes que sí abrieron cuenta
df_abrieron = df[df['AbreCuenta'] == 1]

# Aplicar K-Means Clustering solo en clientes que sí abrieron cuenta
kmeans = KMeans(n_clusters=3, random_state=42)
df_abrieron['Cluster'] = kmeans.fit_predict(df_abrieron[['Edad', 'IngresoAnual', 'TiempoEnSitio']])

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Crear la disposición de la aplicación
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Modelo de Predicción: Apertura de Cuentas (Random Forest)"))),

    dbc.Row([
        dbc.Col(html.Div(f"Exactitud del modelo: {accuracy:.2f}"), width=6),
        dbc.Col(html.Div(f"Clientes que abrieron cuenta: {report['1']['support']:.0f}"), width=6)
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(
            id='scatter-plot',
            figure=px.scatter(df, x='IngresoAnual', y='Edad', color='AbreCuenta',
                              title='Relación entre Ingreso Anual y Edad',
                              labels={'AbreCuenta': 'Abre Cuenta'},
                              color_continuous_scale=px.colors.sequential.Viridis)), width=6),

        dbc.Col(dcc.Graph(
            id='confusion-matrix',
            figure=go.Figure(data=go.Heatmap(
                z=conf_matrix_df.values,
                x=conf_matrix_df.columns,
                y=conf_matrix_df.index,
                colorscale='Viridis',
                text=conf_matrix_df.values,
                hoverinfo='text'
            )).update_layout(title='Matriz de Confusión', xaxis_title='Predicción', yaxis_title='Realidad')), width=6)
    ]),

    dbc.Row(dbc.Col(dcc.Graph(
        id='precision-recall',
        figure=px.bar(
            x=[key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']],
            y=[report[key]['precision'] for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']],
            labels={'x': 'Clases', 'y': 'Precisión'},
            title='Precisión por Clase'
        )
    ))),

    dbc.Row(dbc.Col(dcc.Graph(
        id='recall-bar',
        figure=px.bar(
            x=[key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']],
            y=[report[key]['recall'] for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']],
            labels={'x': 'Clases', 'y': 'Recall'},
            title='Recall por Clase'
        )
    ))),

    dbc.Row(dbc.Col(dcc.Graph(
        id='f1-score-bar',
        figure=px.bar(
            x=[key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']],
            y=[report[key]['f1-score'] for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']],
            labels={'x': 'Clases', 'y': 'F1 Score'},
            title='F1 Score por Clase'
        )
    ))),

    # Nueva Gráfica de Clustering de Clientes que sí abrieron cuenta
    dbc.Row(dbc.Col(dcc.Graph(
        id='kmeans-cluster-abrieron',
        figure=px.scatter(df_abrieron, x='IngresoAnual', y='Edad', color='Cluster',
                          title='Clientes que Abrieron Cuenta: Clustering K-Means',
                          labels={'Cluster': 'Grupo K-Means'},
                          color_continuous_scale=px.colors.sequential.Plasma)
    )))
])

if __name__ == '__main__':
    app.run_server(debug=True)