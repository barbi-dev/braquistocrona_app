from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st 
##PARA APLICACION
st.set_page_config(page_title="BRAQUISTOCRONA", layout="centered")
st.title("Simulación: Comparación de trayectorias")
col1, col2 = st.columns([1,2], gap="medium")
with col1:
    st.subheader("Controles")
    # Parametros de entrada
    x0 = st.slider("coordenada x de punto A", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    y0 = st.slider("coordenada y de punto A", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    x1 = st.slider("coordenada x de punto B", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    y1 = st.slider("coordenada y de punto B", min_value=-10.0, max_value=0.0, value=-1.0, step=0.1)
dx=x1-x0
dy=y1-y0
#PARAMETROS
g=9.81
Ri=np.array([x0,y0])
Rf=np.array([x1,y1])
nsample=2000 #resolucion
# LINEA RECTA
def recta(A:np.ndarray, B:np.ndarray,n:int=nsample)->tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, n) #se parametriza la ec de la recta
    X = (1 - t) * A[0] + t * B[0]
    Y = (1 - t) * A[1] + t * B[1]
    return X, Y
#ARCO CIRCULAR
def circulo(A: np.ndarray, B: np.ndarray, sag_ratio: float = 0.25,n: int = nsample) -> tuple[np.ndarray, np.ndarray]:
    cuerda=B-A
    perp  = np.array([-cuerda[1], cuerda[0] ], dtype=float)
    perp /=np.linalg.norm(perp)
    L=np.linalg.norm(cuerda)
    sag=sag_ratio*L
    M=(A+B)/2
    centro=M+sag*perp
    radio=np.linalg.norm(A-centro)
    #angulos
    phi0=np.arctan2(A[1]-centro[1],A[0]-centro[0])
    phi1=np.arctan2(B[1]-centro[1],B[0]-centro[0])
    dphi  = phi1 - phi0
    if dphi < 0:                  #  CCW (negativo → arriba)
        dphi += 2*np.pi
    if dphi > np.pi:              # rama corta CCW
        dphi -= 2*np.pi
    phi= np.linspace(phi0,phi0+dphi,n)
    X     = centro[0] +radio*np.cos(phi)
    Y     = centro[1] +radio*np.sin(phi)
    return X, Y

#CICLOIDE: BRAQUISTOCRONA
#función que calcula el ángulo final para ir desde A hasta B
def braquistocrona_existe(dx,dy):
    def thetaf(dx: float, dy: float, theta0:float=2.0) -> float:
        theta1 = theta0
#para encontrar el angulo theta se resuelve por metodos numericos: Newton Raphson 
        for _ in range(100):
            f  = dy * (theta1 - np.sin(theta1)) - dx * (1 - np.cos(theta1))
            df = dy * (1 - np.cos(theta1)) - dx * np.sin(theta1)
            theta1 -= f / df
            if abs(f) < 1e-12:
                return theta1
        return None
    sol=thetaf(dx,np.abs(dy))
    if sol is None:
        return False, None
    else:
        return True, sol

#función que arroja valores para construir cicloide
def cicloide(A: np.ndarray, B: np.ndarray, n: int = nsample) -> tuple[np.ndarray, np.ndarray, float, float]:
    theta1 =  thetaf
    a=dx/(theta1-np.sin(theta1))
    t_total=np.sqrt(a/g)*theta1
    theta=np.linspace(0,theta1,n)
    X = A[0] + a * (theta - np.sin(theta))
    Y = A[1] - a * (1 - np.cos(theta))
    return X, Y, t_total, a

#función que calcula tiempos mientras particula se mueve de A a B
def time_path(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    y0 = Y[0]
    ds = np.hypot(np.diff(X), np.diff(Y))
    y_mid = 0.5*(Y[:-1] + Y[1:])
    delta_y= y0 - y_mid
    v = np.sqrt(2*g*delta_y)
    dt = ds / v
    if not np.all(np.isfinite(dt)):
        return None
    t  = np.concatenate(([0], np.cumsum(dt)))
    return t

Xrecta,Yrecta = recta(Ri, Rf)
Xcircle, Ycircle=circulo(Ri, Rf)
T_line_vec = time_path(Xrecta,Yrecta)
T_circ_vec = time_path(Xcircle, Ycircle)
T_line = T_line_vec[-1]
T_circ = T_circ_vec[-1]

#funcion para interpolar tiempo según el array t_vec
def interp(t, t_vec, X, Y):
    if t >= t_vec[-1]:
        return X[-1], Y[-1]
    x = np.interp(t, t_vec, X)
    y = np.interp(t, t_vec, Y)
    return x, y

# --- Streamlit ---
with col2:
    st.subheader("Animación interactiva")
    existe, thetaf = braquistocrona_existe(dx,dy) 
    if dx<=0 or dy >= 0 or existe==False:
        st.error("No son puntos válidos para una cicloide")
        st.stop()
    
    else:
        Xcic, Ycic, tcic, a = cicloide(Ri, Rf)

        Ts_validos = [t for t in [T_line, T_circ, tcic] if (t is not None and tcic<5.0)]
        if not Ts_validos:
            st.error("Ninguna trayectoria cicloide válida para estos puntos.")
            st.stop()

        T_max = max(Ts_validos)
        N_FRAMES = 500
        t_grid  = np.linspace(0, T_max, N_FRAMES)

        frame = st.slider("Elegir el frame (posición del punto)", 0, N_FRAMES-1, 0)
        t = t_grid[frame]
            # --- Grafico ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots()
        if T_line_vec is not None:
            x_line, y_line = interp(t, T_line_vec, Xrecta, Yrecta)
            ax.plot(Xrecta, Yrecta, 'b-', label=f"Recta (t = {T_line:.3f} s)")
            ax.plot(x_line, y_line, 'bo')

        # círculo
        if T_circ_vec is not None:
            x_circ, y_circ = interp(t, T_circ_vec, Xcircle, Ycircle)
            ax.plot(Xcircle, Ycircle, 'g-', label=f"Círculo (t = {T_circ:.3f} s)")
            ax.plot(x_circ, y_circ, 'go')

        # cicloide
        if tcic is not None:
            if t <= tcic:
                theta = t * np.sqrt(g/a)
                x_cyc = Ri[0] + a * (theta - np.sin(theta))
                y_cyc = Ri[1] - a * (1 - np.cos(theta))
            else:
                x_cyc, y_cyc = Rf
            ax.plot(Xcic, Ycic, 'r-', label=f"Cicloide (t = {tcic:.3f} s)")
            ax.plot(x_cyc, y_cyc, 'ro')
        if tcic > T_line*1.1:
            st.error("no hay cicloide válida para esos puntos")
            st.stop()

        ax.set_aspect('equal')
        ax.set_title(f"t = {t:.3f} s")
        ax.legend(fontsize=6, loc="upper right")

        st.pyplot(fig)
    
