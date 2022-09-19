import plotly.express as px
import numpy as np
import plotly
from plotly.graph_objs import Scatter


def learning(design_matrix, t):
    return np.linalg.inv(design_matrix.T @ design_matrix) @ design_matrix.T @ t


def create_polynomial_design_matrix(X, M):
    N = X.shape[0]
    result = np.ones((N, M + 1))
    for i in range(N):
        for j in range(1, M + 1):
            result[i][j] = result[i][j - 1] * X[i]
    return result


def create_fourier_design_matrix(X, M):
    N = X.shape[0]
    result = np.ones((N, M + 1))
    for i in range(N):
        for j in range(1, M + 1):
            if j % 2 == 1:
                result[i][j] = np.sin((j // 2 + 1) * X[i])
            if j % 2 == 0:
                result[i][j] = np.cos(j // 2 * X[i])
    return result


def calculate_error(W, Z, F):
    return (1 / 2) * sum((Z - (W @ F.T)) ** 2)


N = 1000
X = np.linspace(0, 1, N)
Z = 20 * np.sin(2 * np.pi * 3 * X) + 100 * np.exp(X)
error = 10 * np.random.randn(N)
t = Z + error
M = 1
E = np.zeros(100)
helper = 0

for M in range(1, 100 + 1):
    F = create_polynomial_design_matrix(X, M)
    W = learning(F, t)
    E[M - 1] = calculate_error(W, Z, F)
    if M == 1 or M == 8 or M == 100:
        helper = round(E[M - 1], 2)
        df = px.data.tips()
        fig = px.scatter(df, x=X, y=t, opacity=0.65, title=f"M = {M}, error={helper}")
        fig.add_traces(Scatter(x=X, y=Z, name='z(x)'))
        fig.add_traces(Scatter(x=X, y=W @ F.T, name=f"полином степени {M}"))
        # fig.show()
        plotly.offline.plot(fig, filename=f'C:/plotly/polinom_{M}.html')

fig = px.line(x=range(100), y=E, title='Зависимость ошибки от степени полинома')
fig.update_layout(xaxis_title="ось х - степень полинома", yaxis_title="ось у - ошибка")
# fig.show()
plotly.offline.plot(fig, filename='C:/plotly/oshibka.html')

for M in range(1, 10 + 1):
    F = create_fourier_design_matrix(X, M)
    W = learning(F, t)
    E[M - 1] = calculate_error(W, Z, F)
    if M == 1 or M == 7 or M == 10:
        helper = round(E[M - 1], 2)
        df = px.data.tips()
        fig = px.scatter(df, x=X, y=t, opacity=0.65, title=f"M = {M}, error={helper}")
        fig.add_traces(Scatter(x=X, y=Z, name='z(x)'))
        fig.add_traces(Scatter(x=X, y=W @ F.T, name='forier'))
        # fig.show()
        plotly.offline.plot(fig, filename=f'C:/plotly/fourier_{M}.html')