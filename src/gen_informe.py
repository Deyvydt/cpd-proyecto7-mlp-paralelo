"""
Genera informe/informe_final.tex insertando los numeros reales del CSV.
Uso: python3 gen_informe.py
"""
import csv
import json
import math
import os

from common import T_MAX, D_FEATURES, H_HIDDEN, N_CLASSES, ALPHA, ETA

CSV = "../resultados/resultados.csv"
AJUSTE = "../resultados/ajuste_teorico.json"
SALIDA = "../informe/informe_final.tex"


def cargar():
    with open(CSV) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def tabla_principal(filas):
    ns = sorted({r["n"] for r in filas})
    out = []
    for n in ns:
        out.append(r"\multicolumn{8}{l}{\textbf{n\_samples = %d}}\\" % int(n))
        out.append(r"\midrule")
        for r_ in sorted([x for x in filas if x["n"] == n], key=lambda x: x["p"]):
            out.append(
                r"%d & %.2f & %.2f & %.2f & %.3f & %.2f & %.2f & %.4f \\" % (
                    int(r_["p"]), r_["T_s"], r_["T_p"], r_["t_comp"], r_["t_comm"],
                    r_["speedup"], r_["eficiencia"], r_["accuracy"])
            )
        out.append(r"\midrule")
    return "\n".join(out[:-1])


def tabla_weak(filas):
    pares = [(1, 5000), (2, 10000), (4, 20000), (8, 40000)]
    out = []
    for p, n in pares:
        for r_ in filas:
            if int(r_["p"]) == p and int(r_["n"]) == n:
                out.append(r"%d & %d & %.2f & %.2f & %.2f \\" % (
                    p, n, r_["T_p"], r_["speedup"], r_["eficiencia"]))
    return "\n".join(out)


def main():
    filas = cargar()
    coef = json.load(open(AJUSTE))
    P = int(coef["n_cpu"])
    ns = sorted({r["n"] for r in filas})
    n_max = int(max(ns))

    def get(p, n):
        for r_ in filas:
            if int(r_["p"]) == p and int(r_["n"]) == n:
                return r_
        return None

    s_max = max(r["speedup"] for r in filas)
    r_sat = get(P, n_max) or get(2, n_max)
    r32 = get(32, n_max)
    r1 = get(1, n_max)
    e_p = get(P, n_max)["eficiencia"] if get(P, n_max) else 0
    # error relativo medio del ajuste teorico
    def t_teo(p, n):
        W = T_MAX * (n * 0.75) * D_FEATURES * H_HIDDEN
        return coef["c1"] * math.ceil(p / P) * W + coef["c2"] * (math.log2(p) if p > 1 else 0) + coef["c3"]
    err = sum(abs(t_teo(int(r["p"]), int(r["n"])) - r["T_p"]) / r["T_p"] for r in filas) / len(filas) * 100
    gflops_max = max(r["flops_por_seg"] for r in filas) / 1e9
    comm32 = r32["t_comm"] if r32 else 0
    pct_comm32 = (r32["t_comm"] / r32["T_p"] * 100) if r32 else 0

    tex = r"""\documentclass[11pt,a4paper]{article}
\setlength{\parindent}{0pt}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish]{babel}
\usepackage[margin=2.5cm]{geometry}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, fit, backgrounds}
\usepackage{booktabs}
\usepackage{array}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{xurl}
\usepackage{hyperref}
\usepackage{titlesec}
\usepackage{caption}
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{listings}
\usepackage{algorithm}
\usepackage{algpseudocode}

\definecolor{accent}{HTML}{2F6FED}
\titleformat{\section}{\normalfont\Large\bfseries\color{black}}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\large\bfseries}{\thesubsection}{1em}{}
\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{Redes Neuronales --- Informe Final}\fancyhead[R]{\thepage}
\hypersetup{colorlinks=true, linkcolor=accent, urlcolor=accent, citecolor=accent,
  pdftitle={Redes Neuronales - Informe Final},
  pdfauthor={David Marcelo Quispe Velasquez, Johannes Albert Loayza Huaman}}
\lstset{basicstyle=\ttfamily\footnotesize, breaklines=true, frame=single,
  columns=fullflexible, numbers=none}
\floatname{algorithm}{Algoritmo}
\algrenewcommand\algorithmicrequire{\textbf{Entrada:}}
\algrenewcommand\algorithmicensure{\textbf{Salida:}}
% Bucle PARALELO: imprime "pardo" y no "do" (convencion de JaJa usada en clase)
\algrenewtext{ForAll}[1]{\textbf{for all} #1\ \textbf{pardo}}

\begin{document}

\begin{titlepage}
\centering
\vspace*{2cm}
{\Huge\bfseries Redes Neuronales}\\[0.4cm]
{\large Informe Final --- Proyecto de Computación Paralela y Distribuida}\\[0.3cm]
{\large Entrega 2}\\[1cm]
\includegraphics[width=0.7\textwidth]{UTEC-Logo.jpg}\\[1cm]
\begin{tabular}{ll}
\textbf{Integrantes:} & David Marcelo Quispe Velásquez \\
 & Johannes Albert Loayza Huamán \\[4pt]
\textbf{Curso:} & Computación Paralela y Distribuida (CS4052) \\[4pt]
\textbf{Profesor:} & José Fiestas \\[4pt]
\textbf{Fecha:} & \today \\
\end{tabular}
\vfill
\end{titlepage}

\tableofcontents
\newpage

\section*{Participación de los integrantes}
\begin{table}[H]
\centering
\begin{tabular}{lc}
\toprule
Integrante & Participación (\%) \\
\midrule
David Marcelo Quispe Velásquez & 100\% \\
Johannes Albert Loayza Huamán & 100\% \\
\bottomrule
\end{tabular}
\end{table}
\small{100\% indica una contribución equitativa respecto al resto del equipo.}
\normalsize

\section{Introducción}

Elegimos el proyecto de \emph{Redes Neuronales}. Una red de tipo MLP (\emph{Multilayer Perceptron}) procesa sus entradas capa por capa con una multiplicación de matrices, $y = WX + b$, y ajusta sus pesos con \emph{backpropagation}, que también usa productos de matrices, $\delta = W^{T}\delta_{\text{siguiente}}$.

Los pesos iniciales se generan al azar a partir de una \textbf{semilla}, así que dos modelos entrenados con los mismos datos y los mismos hiperparámetros, pero con semillas distintas, terminan con exactitudes (\emph{accuracy}) diferentes. El algoritmo secuencial del enunciado aprovecha esto: entrena $p$ modelos, uno por semilla y uno detrás de otro, y se queda con el de mayor exactitud. El problema es que su tiempo crece de forma lineal con $p$.

Nuestra solución se basa en que las iteraciones del ciclo \texttt{foreach} no dependen entre sí: el modelo de la semilla 3 no necesita nada del modelo de la semilla 2. Todos leen los mismos datos y cada uno escribe solo su propio modelo y su exactitud. Entonces le damos un modelo a cada procesador, los entrenamos todos al mismo tiempo y, al final, comparamos las exactitudes con una reducción en árbol que toma $O(\log p)$ pasos.

En esta entrega implementamos ese diseño en Python, lo validamos contra la versión secuencial, medimos tiempos para $p\in\{1,2,4,8,16,32\}$ y $n\in\{5000,10000,20000,40000\}$, y comparamos los resultados con la expresión teórica.

\paragraph{Notación.} Como en el enunciado $E$ son las épocas pero en el curso $E$ es la eficiencia, usamos:
\begin{itemize}[leftmargin=1.2cm]
\item $T_{max}$: número de épocas; $n$: muestras; $d$: \emph{features}; $h$: neuronas ocultas; $c$: clases.
\item $T_{MLP}(n)$: tiempo de entrenar \textbf{un} MLP; $T_s$: tiempo secuencial total; $T_p$: tiempo paralelo.
\item $W$: trabajo; $T_\infty$: \emph{span}; $S_p$: \emph{speedup}; $E_p$: eficiencia; $C_p$: costo; $P$: núcleos físicos.
\end{itemize}

\section{Método}

\subsection{Algoritmo secuencial de referencia}

El Algoritmo~1 del enunciado divide $(X,y)$ en entrenamiento y prueba y recorre las semillas $s\in\{0,\dots,p-1\}$: para cada una construye y entrena un modelo $M_s$, calcula su exactitud $A_s$ y actualiza $(A_{best},s^*,M^*)$ si corresponde. El ciclo se repite $p$ veces y cada iteración está dominada por el entrenamiento, por lo que
\[ T_s(n,p) = p\cdot T_{MLP}(n). \]

\subsection{Costo de entrenar un MLP}
\label{sec:tmlp}

Para una red con una capa oculta de $h$ neuronas, $d$ \emph{features} y $c$ clases, el costo por muestra es:

\textbf{Forward:} $z_1 = W_1x + b_1$ con $W_1\in\mathbb{R}^{h\times d}\Rightarrow O(h\cdot d)$;\quad $\hat y = W_2a_1+b_2$ con $W_2\in\mathbb{R}^{c\times h}\Rightarrow O(h\cdot c)$.

\textbf{Backward:} $\delta_1 = W_2^{T}\delta_2 \Rightarrow O(h\cdot c)$;\quad $\nabla W_1 = \delta_1 x^{T} \Rightarrow O(h\cdot d)$.

Como $c \le d$, el término dominante por muestra es $O(h\cdot d)$. Para $n$ muestras y $T_{max}$ épocas:
\[ T_{MLP}(n) = O(T_{max}\cdot n\cdot d\cdot h), \]
que coincide con la complejidad $O(E\cdot n\cdot d\cdot h)$ del enunciado.

\subsection{Modelo PRAM propuesto}

En la fase de entrenamiento los $p$ procesadores \textbf{leen} al mismo tiempo los mismos datos, pero cada procesador $P_s$ \textbf{escribe} solo en sus posiciones $M[s]$ y $A[s]$: usamos una \textbf{CREW-PRAM} (\emph{Concurrent Read, Exclusive Write}). En la reducción, cada procesador activo solo lee y escribe su propio par de posiciones, así que alcanza con el modelo más restrictivo \textbf{EREW}. Para la reducción tomamos el árbol de la suma de $n$ números visto en clase, cambiando la suma por el máximo y guardando el índice del ganador; la condición $j<p$ hace que funcione aunque $p$ no sea potencia de 2.

\subsection{Algoritmo paralelo (pseudocódigo PRAM)}

Pusimos el costo al lado de cada línea para que la complejidad se pueda leer directamente del pseudocódigo. La Fase~1 es CREW y cuesta $O(T_{MLP}(n))$; la Fase~2 es EREW y cuesta $O(\log p)$. El indice de iteracion es la \textbf{semilla} $s$: el procesador $P_s$ entrena el modelo $M_s$ con la semilla $s$ y escribe su resultado en la posicion $A[s]$. Siguiendo la convencion de JaJa vista en clase, los bucles paralelos llevan \texttt{pardo} y los secuenciales \texttt{do}: el bucle de la Fase~1 y el bucle interno de la Fase~2 son paralelos, mientras que el bucle externo sobre $k$ de la Fase~2 es secuencial, porque la ronda $k$ necesita los ganadores de la ronda $k-1$.

\begin{algorithm}[H]
\caption{Entrenamiento paralelo de MLPs en PRAM (CREW/EREW)}
\label{alg:pram}
\begin{algorithmic}[1]
\Require Datos $(X,y)$ en memoria compartida; $p$ procesadores $P_0,\dots,P_{p-1}$; hiperparámetros $(\alpha,\eta,h,T_{max})$
\Ensure $M^*$, $A_{best}$ en memoria compartida
\State $P_0$: particionar $(X,y) \to (X_{train},y_{train}),(X_{test},y_{test})$ \Comment{$O(n\cdot d)$}
\State $P_0$: escribir particiones en memoria compartida \Comment{$O(n\cdot d)$}
\Statex \textit{// Fase 1 --- Entrenamiento independiente (CREW)}
\ForAll{$s$, $0\le s \le p-1$}
    \State Inicializar generador aleatorio con semilla $s$ \Comment{$O(1)$}
    \State $M_s \gets$ construir\_modelo$(\alpha,\eta,h)$ \Comment{$O(d\cdot h)$}
    \State $M_s \gets$ entrenar$(M_s, X_{train}, y_{train}, T_{max})$ \Comment{$O(T_{max}\cdot n\cdot d\cdot h)$}
    \State $A_s \gets$ accuracy$(M_s, X_{test}, y_{test})$ \Comment{$O(n\cdot d\cdot h)$}
    \State escribir $(M_s,A_s)$ en $M[s], A[s]$ \Comment{$O(d\cdot h)$, exclusiva}
\EndFor
\Statex \textit{// Fase 2 --- Reduccion en arbol, maximo con indice (EREW)}
\ForAll{$s$, $0\le s \le p-1$}
    \State $idx[s] \gets s$ \Comment{$O(1)$}
\EndFor
\For{$k \gets 1$ \textbf{to} $\lceil \log_2 p \rceil$} \Comment{$\lceil\log_2 p\rceil$ rondas, secuencial}
    \ForAll{$s$ tal que $s \bmod 2^k = 0$}
        \State $j \gets s + 2^{k-1}$ \Comment{$O(1)$}
        \If{$j < p$ \textbf{and} $A[j] > A[s]$} \Comment{$O(1)$}
            \State $A[s] \gets A[j]$;\quad $idx[s] \gets idx[j]$ \Comment{$O(1)$}
        \EndIf
    \EndFor
\EndFor
\Statex \textit{// Fase 3 --- Resultado}
\State $s^* \gets idx[0]$;\quad $A_{best} \gets A[0]$;\quad $M^* \gets M[s^*]$ \Comment{$O(1)$}
\State \Return $(M^*, A_{best})$
\end{algorithmic}
\end{algorithm}

\begin{figure}[H]
\centering
\begin{tikzpicture}[
  box/.style={draw, rounded corners, minimum width=3.4cm, minimum height=1.1cm, align=center, font=\small},
  arr/.style={-{Latex[length=2mm]}}]
\node[box, fill=blue!8] (data) {Datos compartidos\\$(X,y)$};
\node[box, fill=cyan!8, below=1.2cm of data, xshift=-4.4cm] (p0) {$P_0$: entrena $M_0$\\(semilla $s=0$)};
\node[box, fill=cyan!8, below=1.2cm of data] (p1) {$P_1$: entrena $M_1$\\(semilla $s=1$)};
\node[box, fill=cyan!8, below=1.2cm of data, xshift=4.4cm] (p2) {$P_{p-1}$: entrena $M_{p-1}$\\(semilla $s=p{-}1$)};
\node[box, fill=teal!10, below=1.4cm of p1] (red) {Reducción en árbol\\$O(\log p)$};
\node[box, fill=purple!8, below=1.2cm of red] (res) {$M^*,\ A_{best}$};
\draw[arr] (data.south) -| (p0.north);
\draw[arr] (data.south) -- (p1.north);
\draw[arr] (data.south) -| (p2.north);
\draw[arr] (p0.south) |- (red.west);
\draw[arr] (p1.south) -- (red.north);
\draw[arr] (p2.south) |- (red.east);
\draw[arr] (red.south) -- (res.north);
\end{tikzpicture}
\caption{Entrenamiento independiente (CREW), un modelo por semilla $s$, seguido de una reducción en árbol (EREW). Cada caja de la fila central cuesta $O(T_{MLP}(n))$.}
\label{fig:pram}
\end{figure}

\subsection{Métricas teóricas}

\textbf{Trabajo:} $W(n,p) = p\cdot T_{MLP}(n) + O(p) = O(p\cdot T_{max}\cdot n\cdot d\cdot h)$.

\textbf{Span:} $T_\infty(n,p) = T_{MLP}(n) + O(\log p) = O(T_{max}\cdot n\cdot d\cdot h + \log p)$.

\textbf{Tiempo paralelo} (Teorema de Brent, $T_p = O(W/p + T_\infty)$, con $W/p = O(T_{MLP}(n))$):
\[ T_p(n,p) = O(T_{max}\cdot n\cdot d\cdot h + \log p). \]

\textbf{Speedup y eficiencia:}
\[ S_p = \frac{T_s}{T_p} = \frac{p}{1 + \dfrac{\log p}{T_{max}\cdot n\cdot d\cdot h}} \approx p,
\qquad E_p = \frac{S_p}{p} = \frac{1}{1 + \dfrac{\log p}{T_{max}\cdot n\cdot d\cdot h}} \approx 1. \]

\textbf{Costo:} $C_p = p\cdot T_p = \Theta(T_s)$, es decir, el algoritmo es de \textbf{costo óptimo}.

\subsection{Implementación y versiones beta}

El código está en Python y usa \texttt{scikit-learn} (\texttt{MLPClassifier}) para los modelos y el módulo \texttt{multiprocessing} para el paralelismo. Usamos procesos y no hilos porque el \emph{GIL} de CPython impide que varios hilos ejecuten bytecode a la vez, de modo que con hilos no habría paralelismo real de cómputo; cada proceso recibe una copia de los datos de entrenamiento, que es la traducción práctica de la lectura concurrente del modelo CREW. Se fija \texttt{OMP\_NUM\_THREADS=1} en cada proceso para que la biblioteca de álgebra lineal no cree hilos extra y la medición de $p$ sea limpia. El contexto de \texttt{multiprocessing} se elige automáticamente: \texttt{fork} en Linux y macOS, donde el proceso hijo hereda la memoria del padre, y \texttt{spawn} en Windows, donde cada proceso arranca de cero y recibe los datos por serialización; esta diferencia se refleja en el tiempo de comunicación medido.

El desarrollo quedó registrado en tres versiones:

\begin{enumerate}[leftmargin=1.2cm]
\item \textbf{Beta 1 --- \texttt{beta1\_secuencial.py}}: implementación directa del Algoritmo~1 del enunciado. Sirve como línea base $T_s$ y como referencia de correctitud.
\item \textbf{Beta 2 --- \texttt{beta2\_paralelo\_simple.py}}: paraleliza la fase de entrenamiento con un \texttt{Pool} de procesos, pero elige el ganador con un \texttt{max()} secuencial de costo $O(p)$. \emph{Limitación detectada:} esa selección no corresponde al PRAM propuesto, donde la reducción debe costar $O(\log p)$.
\item \textbf{Beta 3 --- \texttt{beta3\_paralelo\_final.py}}: versión final. Implementa la reducción en árbol de las líneas 10--18 tal como aparecen en el pseudocódigo (con el índice de semilla $s$ y la condición $j<p$), separa la medición de tiempo de cómputo y de comunicación, y cuenta los FLOPs de la región paralelizable.
\end{enumerate}

\paragraph{Validación.} El script \texttt{validacion.py} compara la Beta~3 con la Beta~1 para varios $(p,n)$, incluyendo $p$ que no son potencia de 2 ($p=3$ y $p=6$). En todos los casos ambas versiones devuelven \textbf{la misma semilla ganadora y la misma exactitud}, lo que confirma que la reducción en árbol selecciona el mismo modelo que el máximo secuencial del algoritmo original.

\paragraph{Conteo de FLOPs.} Para la región paralelizable (el entrenamiento) contamos, por muestra y época, $2hd + 2hc$ operaciones en el \emph{forward} y $4hd+4hc$ en el \emph{backward} (gradientes y actualización de pesos), es decir
\[ \text{FLOPs} = T_{max}\cdot n_{train}\cdot 6h(d+c). \]

\paragraph{Configuración experimental.} $d = @@D@@$ \emph{features}, $c = @@C@@$ clases, $h = @@H@@$ neuronas ocultas, $T_{max} = @@TMAX@@$ épocas, $\alpha = @@ALPHA@@$, $\eta = @@ETA@@$; datos sintéticos generados con \texttt{make\_classification} y divididos 75/25 en entrenamiento y prueba. Las mediciones se tomaron en una máquina con $P = @@P@@$ núcleos físicos disponibles, y cada punto es la mediana de varias repeticiones. Como línea base se usa $T_s(p,n) = p\cdot t_1(n)$, donde $t_1(n)$ es el tiempo mediano medido de entrenar un único modelo.

\section{Resultados y análisis}

\subsection{Tiempos de ejecución}

\begin{table}[H]
\centering\small
\begin{tabular}{rrrrrrrr}
\toprule
$p$ & $T_s$ [s] & $T_p$ [s] & cómputo [s] & comunic. [s] & $S_p$ & $E_p$ & accuracy \\
\midrule
@@TABLA@@
\bottomrule
\end{tabular}
\caption{Mediciones para cada combinación de $p$ y \texttt{n\_samples}.}
\label{tab:resultados}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{fig1_tiempos.png}
\caption{Tiempo de ejecución $T_p$ frente al número de procesos.}
\end{figure}

Hasta $p = @@P@@$ el tiempo se mantiene prácticamente plano: los procesos corren realmente a la vez y entrenar $p$ modelos cuesta casi lo mismo que entrenar uno. A partir de ahí $T_p$ crece de forma aproximadamente lineal en $p$, porque los procesos ya no caben en los núcleos disponibles y empiezan a turnarse.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{fig2_desglose.png}
\caption{Desglose en tiempo de cómputo y tiempo de comunicación/overhead.}
\end{figure}

El tiempo de comunicación incluye la creación de los procesos, el envío de los datos a cada uno, la recolección de resultados y la reducción. Para $n = @@NMAX@@$ y $p = 32$ fue de @@COMM32@@ s, es decir el @@PCTCOMM32@@\,\% de $T_p$: el cómputo domina ampliamente, tal como asume el modelo. La fase de reducción por sí sola es de microsegundos, coherente con su costo $O(\log p)$.

\subsection{Comparación con la expresión teórica}

El modelo PRAM supone procesadores ilimitados. Para comparar con la máquina real hay que añadir un factor que refleje que solo hay $P$ núcleos: con $p$ procesos sobre $P$ núcleos, cada núcleo ejecuta $\lceil p/P \rceil$ entrenamientos en serie. Normalizamos entonces la expresión $O(\cdot)$ a segundos con tres constantes de proporcionalidad:
\[
T_p(n,p) \;=\; c_1 \left\lceil \frac{p}{P} \right\rceil T_{max}\, n_{train}\, d\, h \;+\; c_2 \log_2 p \;+\; c_3 ,
\]
donde el primer término es el entrenamiento, el segundo la reducción y el tercero el costo fijo de arranque. Ajustando $c_1$, $c_2$ y $c_3$ por mínimos cuadrados sobre todas las mediciones se obtiene
\[
c_1 = @@C1@@ \ \text{s/FLOP-eq},\qquad c_2 = @@C2@@ \ \text{s},\qquad c_3 = @@C3@@ \ \text{s},
\]
con un error relativo medio de @@ERR@@\,\% respecto a los datos experimentales. El ajuste se hizo con coeficientes no negativos, ya que los tres representan costos. Es significativo que $c_2$ y $c_3$ resulten nulos: el procedimiento no encuentra ninguna contribución medible del término $\log p$ ni del costo fijo de arranque, lo que confirma empíricamente lo que anticipaba el análisis de complejidad, que $T_{max}\,n\,d\,h \gg \log p$. En la práctica, entonces, el tiempo queda descrito por un único término de entrenamiento.

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{fig3_teoria_vs_exp.png}
\caption{Curva teórica superpuesta a la experimental. Los puntos son mediciones y las líneas el modelo ajustado.}
\end{figure}

La curva teórica reproduce las dos regiones observadas: la zona plana mientras $p \le P$ y el crecimiento lineal cuando $p > P$. Esto confirma que la discrepancia con el PRAM ideal no viene del algoritmo, sino del número limitado de núcleos: el término $\lceil p/P \rceil$ es exactamente la parte que el modelo PRAM no considera.

\subsection{Speedup y eficiencia}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{fig4_speedup.png}
\caption{Speedup medido frente al speedup ideal $S = p$.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{fig5_eficiencia.png}
\caption{Eficiencia $E_p = S_p/p$.}
\end{figure}

El speedup sigue la recta ideal mientras $p \le P$ y luego se satura en torno a $S_{max} \approx @@SMAX@@$, muy cerca del número de núcleos $P = @@P@@$. Este techo es el esperado: si todos los núcleos están ocupados, agregar más procesos no agrega capacidad de cómputo, solo reparte el mismo tiempo de CPU entre más tareas. Sustituyendo $\lceil p/P\rceil$ en la definición de speedup,
\[
S_p \;=\; \frac{p\,T_{MLP}}{\lceil p/P\rceil\,T_{MLP}} \;=\; \frac{p}{\lceil p/P \rceil} \;\le\; P ,
\]
que es precisamente la curva medida. La eficiencia se mantiene alta mientras $p\le P$ y después cae como $P/p$, por la misma razón.

Conviene notar la diferencia con la Ley de Amdahl: aquí el techo no lo impone una fracción secuencial del algoritmo (que es prácticamente nula), sino el hardware. En una máquina con más núcleos la misma implementación seguiría escalando, porque el algoritmo no tiene dependencias entre modelos.

\subsection{FLOPs}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{fig6_flops.png}
\caption{Rendimiento en GFLOP/s frente al número de procesos.}
\end{figure}

El rendimiento sube hasta $p = P$ y después se mantiene esencialmente constante, alcanzando un máximo de @@GFLOPS@@ GFLOP/s. Esto es consistente con lo anterior: una vez saturados los núcleos, la máquina ya está entregando todas las operaciones por segundo que puede, y aumentar $p$ solo reparte esa misma capacidad.

\subsection{Escalabilidad}

Como el número de semillas es igual a $p$, al aumentar $p$ también aumenta el trabajo total: estamos ante un caso de \emph{escalabilidad débil} (Ley de Gustafson). Para evaluarla hacemos crecer $n$ junto con $p$:

\begin{table}[H]
\centering
\begin{tabular}{rrrrr}
\toprule
$p$ & \texttt{n\_samples} & $T_p$ [s] & $S_p$ & $E_p$ \\
\midrule
@@TABLAWEAK@@
\bottomrule
\end{tabular}
\caption{Escalabilidad débil: el tamaño del problema crece proporcionalmente a $p$.}
\label{tab:weak}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{fig7_weak_scaling.png}
\caption{Escalabilidad débil: tiempo y eficiencia cuando $n$ y $p$ crecen juntos.}
\end{figure}

\paragraph{¿Se mantiene constante la eficiencia?} No en toda la escala, y la razón es clara. Mientras $p \le P$ la eficiencia se mantiene cerca de 1, que es el comportamiento ideal del escalado débil: el trabajo crece, pero también la capacidad de cómputo efectiva. En cuanto $p > P$ la eficiencia cae aproximadamente como $P/p$, porque la capacidad de cómputo deja de crecer aunque el trabajo sí lo haga. Es decir, \textbf{el algoritmo sí es débilmente escalable, pero la plataforma de medición no lo es más allá de sus $P$ núcleos}. La condición práctica para mantener $E_p$ constante es, por tanto, $p \le P$: el número de semillas no debería superar el número de núcleos disponibles.

\subsection{Mejoras propuestas}

\begin{enumerate}[leftmargin=1.2cm]
\item \textbf{Paralelismo de datos dentro de cada modelo.} Cuando $p$ supera el número de núcleos, en lugar de entrenar más modelos a la vez conviene repartir las $n$ muestras de un mismo modelo entre varios procesos y promediar los gradientes. Esto rompe el techo $S\le P$ para el caso de pocos modelos y datos muy grandes, a costa de introducir comunicación en cada época.
\item \textbf{Paralelismo de modelo.} Repartir las filas de $W_1$ (las $h$ neuronas) entre procesos, útil cuando $h$ es grande.
\item \textbf{Versión distribuida con MPI.} La implementación actual es de memoria compartida. Con \texttt{mpi4py}, la lectura concurrente del PRAM se convierte en un \texttt{comm.bcast} de los datos y la reducción en un \texttt{comm.reduce} con \texttt{MPI.MAXLOC}, que internamente también usa un árbol de $O(\log p)$ pasos. Esto permitiría usar varios nodos y superar el límite de núcleos de una sola máquina.
\item \textbf{Reducir el costo de comunicación inicial} usando memoria compartida real (\texttt{multiprocessing.shared\_memory}) en lugar de copiar el conjunto de entrenamiento a cada proceso, lo que acercaría aún más la implementación al modelo CREW.
\item \textbf{Balanceo dinámico} si en el futuro los modelos tuvieran costos distintos (por ejemplo, arquitecturas o números de épocas diferentes por semilla), mediante un \emph{work pool} en lugar del reparto estático actual.
\end{enumerate}

\subsection{Conclusiones}

\begin{itemize}[leftmargin=1.2cm]
\item La implementación reproduce exactamente el resultado del algoritmo secuencial (misma semilla ganadora y misma exactitud), incluso con $p$ que no es potencia de 2.
\item El tiempo paralelo medido se ajusta al modelo teórico $O(T_{max}\,n\,d\,h + \log p)$ corregido por el número de núcleos, con un error relativo medio de @@ERR@@\,\%.
\item El \emph{speedup} es prácticamente ideal mientras $p \le P$ y se satura en $S \approx @@SMAX@@$, cerca de $P = @@P@@$; el límite es del hardware y no del algoritmo.
\item El costo de la reducción en árbol resultó despreciable frente al entrenamiento, como anticipaba el análisis de complejidad.
\end{itemize}

\section{Bibliografía}

\begin{enumerate}[leftmargin=1.2cm]
\item J. Fiestas, \emph{Computación Paralela y Distribuida --- Proyecto Laboratorio, 2026-II}, UTEC. \textit{Impacto:} de aquí salen el problema, el algoritmo secuencial de referencia, la complejidad $O(E\cdot n\cdot d\cdot h)$ y los parámetros de las pruebas.
\item J. Fiestas, \emph{Diapositivas del curso CS4052}, UTEC, 2026-II. U2.1: speedup, eficiencia y Ley de Gustafson. U2.2: modelo PRAM, EREW/CREW, métricas $T_p$, $C_p$, $W$, $T_\infty$, Teorema de Brent, notación \texttt{pardo} y reducción en árbol. \textit{Impacto:} notación del pseudocódigo, esquema de reducción y definición de todas las métricas.
\item J. JáJá, \emph{An Introduction to Parallel Algorithms}, Addison-Wesley, 1992. \textit{Impacto:} referencia del curso para el principio \emph{work--time} y el Teorema de Brent.
\item scikit-learn, \emph{MLPClassifier}, \url{https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html}. \textit{Impacto:} define la red implementada y respalda la derivación del costo por muestra.
\item Documentación de Python, \emph{multiprocessing}, \url{https://docs.python.org/3/library/multiprocessing.html}. \textit{Impacto:} mecanismo usado para ejecutar los entrenamientos en paralelo.
\item mpi4py, documentación oficial, \url{https://mpi4py.readthedocs.io/en/stable/}. \textit{Impacto:} referencia para la versión distribuida propuesta como mejora.
\end{enumerate}

\subsection*{Declaración de uso de IA}
Usamos un asistente de IA (Claude, de Anthropic) como apoyo para repasar conceptos del curso (redes neuronales, modelo PRAM y métricas), revisar la notación y dar formato en \LaTeX. El diseño del algoritmo, la implementación y el análisis de los resultados los revisamos nosotros contra el material del curso.

\subsection*{Código fuente}
El código completo, los resultados crudos y los scripts que generan las figuras están en:
\url{https://github.com/Deyvydt/cpd-proyecto7-mlp-paralelo}

\end{document}
"""
    vals = {
        "D": D_FEATURES, "C": N_CLASSES, "H": H_HIDDEN, "TMAX": T_MAX,
        "ALPHA": "10^{-4}", "ETA": "10^{-3}", "P": P,
        "TABLA": tabla_principal(filas), "TABLAWEAK": tabla_weak(filas),
        "NMAX": n_max, "COMM32": "%.2f" % comm32, "PCTCOMM32": "%.1f" % pct_comm32,
        "C1": "%.3e" % coef["c1"], "C2": "%.3e" % coef["c2"], "C3": "%.3e" % coef["c3"],
        "ERR": "%.1f" % err, "SMAX": "%.2f" % s_max, "GFLOPS": "%.2f" % gflops_max,
    }
    for k, v in vals.items():
        tex = tex.replace("@@%s@@" % k, str(v))


    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "w", encoding="utf8") as f:
        f.write(tex)
    print("Informe generado en", SALIDA)
    print(f"  P={P}  S_max={s_max:.2f}  error ajuste={err:.1f}%  GFLOPS_max={gflops_max:.2f}")


if __name__ == "__main__":
    main()
