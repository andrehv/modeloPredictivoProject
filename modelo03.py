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
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import datetime

# Verificar nombres de las hojas
xls = pd.ExcelFile("ClientesBancoEjemplo.xlsx")
print(xls.sheet_names)  # Esto te mostrará todas las hojas

# Ajusta el nombre de la hoja según sea necesario
df = pd.read_excel("ClientesBancoEjemplo.xlsx", sheet_name="Ventas")

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

# Leer los datos de ventas para la proyección
df_sales = pd.read_excel("ClientesBancoEjemplo.xlsx", sheet_name='Ventas')
df_sales['Fecha'] = pd.to_datetime(df_sales['Fecha'])  # Asegurarse de que las fechas están en formato datetime
df_sales.set_index('Fecha', inplace=True)

# Proyección de tendencia de ventas usando ARIMA
model_arima = ARIMA(df_sales['Ventas'], order=(1, 1, 1))  # Parámetros simples para la demo
model_fit = model_arima.fit()
forecast = model_fit.get_forecast(steps=6)
forecast_index = [df_sales.index[-1] + datetime.timedelta(weeks=4 * i) for i in range(1, 7)]
forecast_df = pd.DataFrame({'Fecha': forecast_index, 'Forecast': forecast.predicted_mean})
forecast_df.set_index('Fecha', inplace=True)

# Descomposición de series temporales
decomposition = seasonal_decompose(df_sales['Ventas'], model='additive', period=12)  # Periodo para mensualidad

# Filtrar las ventas y no ventas
df_ventas = df_sales[df_sales['Ventas'] > 0]  # Ventas > 0 son ventas
df_no_ventas = df_sales[df_sales['Ventas'] == 0]  # Ventas == 0 son no ventas

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
    ))),

    # Gráfica 1: Línea de Tendencia de Ventas
    dbc.Row(dbc.Col(dcc.Graph(
        id='linea-ventas',
        figure=px.line(df_ventas, x=df_ventas.index, y='Ventas', title='Ventas Históricas')
    ))),

    # Gráfica 2: Línea de No Ventas
    dbc.Row(dbc.Col(dcc.Graph(
        id='linea-no-ventas',
        figure=px.line(df_no_ventas, x=df_no_ventas.index, y='Ventas', title='No Ventas Históricas')
    ))),

    # Gráfica 3: Línea con Pronóstico
    dbc.Row(dbc.Col(dcc.Graph(
        id='linea-pronostico',
        figure=go.Figure([
            go.Scatter(x=df_sales.index, y=df_sales['Ventas'], mode='lines', name='Ventas Históricas'),
            go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines+markers', name='Pronóstico')
        ]).update_layout(title='Pronóstico de Ventas para los Próximos 6 Meses')
    ))),

    # Gráfica 4: Boxplot Mensual de Ventas
    dbc.Row(dbc.Col(dcc.Graph(
        id='boxplot-mensual',
        figure=px.box(df_sales.resample('M').sum(), y='Ventas', title='Variabilidad Mensual de Ventas')
    ))),

    # Gráfica 5: Heatmap de Ventas por Día y Mes
    dbc.Row(dbc.Col(dcc.Graph(
        id='heatmap-ventas',
        figure=px.imshow(df_sales.pivot_table(index=df_sales.index.month, columns=df_sales.index.day, values='Ventas'),
                         labels=dict(x="Día del Mes", y="Mes del Año", color="Ventas"),
                         title='Mapa de Calor de Ventas por Día y Mes')
    ))),

    # Gráfica 6: Descomposición de la Serie Temporal
    dbc.Row([
        dbc.Col(dcc.Graph(
            id='decomposition-trend',
            figure=px.line(x=decomposition.trend.index, y=decomposition.trend, title='Tendencia de Ventas')
        ), width=6),
        dbc.Col(dcc.Graph(
            id='decomposition-seasonal',
            figure=px.line(x=decomposition.seasonal.index, y=decomposition.seasonal, title='Estacionalidad de Ventas')
        ), width=6)
    ])
])

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)