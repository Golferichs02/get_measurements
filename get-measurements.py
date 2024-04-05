"""get-measures.py
    El proposito de este script de python es que se puedan medir objetos en
    el mundo real utilizando vision computacional, para calcular el perimetro
    de nuestros objetos en 2d.

    Autor:Jorge Alberto Rosales de Golferichs
    Organizacion: Universidad de Monterrey
    Contacto: jorge.rosalesd@udem.edu
    Fecha de creacion: Saturday 27 March 2024
    Ultima edicion: Friday 05 April 2024

Ejemplo de como usarlo (actualizarlo dependiendo de sus carpetas)
python get-measurements.py -c 1 -j .\calibration-parameters\calibration_data.json -z 30 
    
"""

import argparse as arg
import cv2
import math
import json
import sys
import os
import numpy as np


def user_interaction()->arg.ArgumentParser:
    """
    Recipila los argumentos proporcionados por el usuario.

    Retorna:
        args: Un objeto ArgumentParser que contiene los argumentos 
        proporcionados por el usuario.

    Esta función Recipila informacion crucial para el funcionamiento del
    programa como el índice de la cámara, la distancia en z y la ubicación 
    del archivo JSON.
    """


# Parse user's argument    
    parser = arg.ArgumentParser(description='Object detection')
    parser.add_argument('-c',
                        '--cam_index', 
                        type=int, 
                        required=True,
                        default='calibration-images',
                        help="Folder where the calibration images are")
    parser.add_argument('-z',
                        '--distance_z',
                        type=float,
                        required=True,
                        help="Distance in z ")
    parser.add_argument('-j',
                        '--json_file',
                        type=str,
                        required=True,
                        help="Json file location")

    args = parser.parse_args()

    return args

def initialise_camera(args:arg.ArgumentParser)->cv2.VideoCapture:   
    """
    Inicializa la cámara utilizando los argumentos proporcionados.

    Parámetros:
        args: Un objeto ArgumentParser que contiene el index de la camara.

    Retorna:
        cv2.VideoCapture: Un objeto VideoCapture que representa la cámara 
        inicializada.

    Esta función utiliza los argumentos proporcionados para inicializar una 
    cámara utilizando el índice de la cámara especificado por el usuario.
    """
    # Inicializa la cámara
    cap = cv2.VideoCapture(args.cam_index)
    return cap

# Lista para almacenar los puntos de la figura
points = []
# Bandera para indicar si la figura está cerrada
is_figure_closed = False
distancias=[]

def click_event(event, x, y, flags, param):
    """
    Función de manejo de eventos de clic izquierdo del mouse.

    Parámetros:
        event:El tipo de evento de OpenCV.
        x:La coordenada x del punto de clic.
        y:La coordenada y del punto de clic.
        flags:Indicadores adicionales de OpenCV.
        param:Parámetros adicionales pasados a través de cv2.setMouseCallback()

    Esta función maneja los eventos de clic del mouse. Si se detecta un clic 
    izquierdo y la figura no está cerrada,se guarda el punto en la lista de 
    puntos. Luego, llama la funcion de line segments donde se calcula y 
    muestra la información de las distancias de las lineas a tiempo real.
    """

    global is_figure_closed, points
    
    # Si el evento fue un clic izquierdo, guarda el punto
    if event == cv2.EVENT_LBUTTONDOWN and not is_figure_closed:
        points.append((x, y))

        # Coordenadas del centro de la imagen
        cx, cy = matriz[0, 2], matriz[1, 2]
        # Coordenada X respecto al centro en pixeles
        x_respecto_al_centro = x - cx
        y_respecto_al_centro = y - cy
        print(f"X: {x_respecto_al_centro} Y {y_respecto_al_centro}")
        # Una vez añadido el punto, calcula y muestra información de los segmentos de línea
        if len(points) >= 2:  # Asegura que hay al menos dos puntos para formar un segmento
            line_segments(points, matriz, dz)


def line_segments(points:list, camera_matrix:np, distance_z:float)->list:
    """
    Calcula y ordena los segmentos de línea formados por una lista de puntos
    en una imagen.

    Parámetros:
        points: Una lista de puntos que definen las coordenadas.
        camera_matrix:La matriz de la cámara que describe su calibración.
        distance_z: La distancia entre la camara y objeto.

    Retorna:
        list: Una lista de tuplas que contienen información de las distancias 
        reales sobre los segmentos de línea, ordenados por longitud de menor 
        a mayor.

    Esta función toma una lista de puntos que definen las coordenadas de las 
    lineas en una imagen junto con la matriz de la cámara y la distancia focal.
    Luego calcula la distancia real de las líneas formados por estos puntos, 
    los ordena de menor a mayor distancia de línea y devuelve una lista 
    ordenada de segmentos de línea junto con su longitud.
    """
    global ahora
    global segments_info 
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    segments_info = []
    
    for i in range(len(points)):
        
        next_index = (i + 1) % len(points)  # Obtén el siguiente punto para formar un segmento de línea
        # Coordenadas del punto actual y del siguiente
        u_i, v_i = points[i]
        u_next, v_next = points[next_index]
        # Convierte coordenadas de píxeles a coordenadas del mundo real
        X_i = (u_i - cx) * distance_z / fx
        Y_i = (v_i - cy) * distance_z / fy
        X_next = (u_next - cx) * distance_z / fx
        Y_next = (v_next - cy) * distance_z / fy

        # Calcula la distancia euclidiana entre los puntos consecutivos.
        dx = X_next - X_i
        dy = Y_next - Y_i
        distance = math.sqrt(dx**2 + dy**2)

        segments_info.append((i+1,distance))

        if i == ahora: # si se elimina se imprimira 2 veces al momento de 
            ahora += 1 # poner 1 linea (solo las primeras ocasiones) si se 
            continue # deja ya nunca pasara dicho error.

        # Hacer una copia de segments_info y ordenarla por longitud de línea
        segments_info_ordenada = sorted(segments_info, key=lambda x: x[1], reverse=False)

        # Comparar y imprimir la información de segmentos ordenados por longitud
        print("---------------------------------------------")
        print("Segmentos ordenados por longitud (de menor a mayor):")
        for numero_linea, distancia in segments_info_ordenada:
            
            print(f"P{numero_linea-1}_{numero_linea}: {distancia} cm")
        print("---------------------------------------------\n")

    return segments_info_ordenada


def calculate_perimeter(segments_info:list)->float: # calcula el perimetro de todas las distancias.
    """
    Calcula el perímetro sumando todas las distancias proporcionadas en la 
    lista de segmentos.

    Parámetros:
        segments_info (list): Una lista de tuplas donde cada tupla contiene 
        información sobre un segmento,donde el primer elemento es opcional 
        y el segundo elemento es la distancia del segmento.

    Retorna:
        float: El perímetro total calculado sumando todas las distancias de los segmentos.
    """
    perimeter = sum(distance for _, distance in segments_info)
    return perimeter

def draw_lines(frame:cv2, points:np, is_figure_closed:str)->None: # dibuja las lineas seleccionadas con clic izquierdo
    """
    Dibuja las líneas seleccionadas con clic izquierdo.

    Parámetros:
        frame: El marco sobre el que se dibujarán las líneas.
        points: Los puntos que definen los extremos de las líneas.
        is_figure_closed: Indica si la figura está cerrada o abierta.

    No devuelve ningún valor, pero modifica el marco de imagen (frame) 
    proporcionado.
    """
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2)
    if is_figure_closed and len(points) > 1:
        cv2.line(frame, points[-1], points[0], (255, 0, 0), 2)

def handle_user_interaction(matriz:np)->str: # perimite una nueva interaccion con el usuario
    global is_figure_closed, points,ahora, segments_info

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and len(points) >2 and not is_figure_closed: #presionar c para cerrar la figura
        is_figure_closed = True  # Marca la figura como cerrada
        segments_info=line_segments(points, matriz, dz) # imprime denuevo las lineas segmentadas incluyendo la cerrada
        perimeter = calculate_perimeter(segments_info) # calcula e imprime el perimetro
        print(f"Perímetro: {perimeter} centimetros")
        segments_info.clear() #borra las distancias
    elif key == ord('b'): #presionar b para borrar 
        points.clear() #elimina los puntos coordenados
        is_figure_closed = False # asegura que la figura este cerrada
        segments_info.clear() #elimina los puntos coordenados
        ahora = 1 # Reinicia la variable ahora a su valor inicial (se utiliza en line segmentation)
    return key

def main_loop_perimeter(cap:cv2,matriz:np,dist:np,args:arg.ArgumentParser)->None:

    """
    Función principal que maneja el bucle principal para el cálculo del 
    perímetro de figuras.

    Parámetros:
        cap (cv2.VideoCapture): El objeto 'VideoCapture' que proporciona el 
        flujo de video.
        matriz (np.ndarray): La matriz de calibración de la cámara.
        dist (np.ndarray): Coeficientes de distorsión de la cámara.
        args (argparse.ArgumentParser): Los argumentos pasados por línea 
        de comandos.

    Esta función inicia un bucle principal que captura y procesa el flujo de 
    video en tiempo real.Permite al usuario interactuar con la aplicación a 
    través de clics del mouse y teclas del teclado.El usuario puede dibujar 
    líneas en la imagen con click izquierdo, borrar la figura presionando la 
    letra "b", cerrar la figura presionando la letra "c" o salir del programa 
    presionando 'q'.
    """

    global is_figure_closed
    global dz #declaramos algunos valores globales porque click event no deja ingresar variables como funcion.
    dz=args.distance_z
    global ahora
    ahora=1

    cv2.namedWindow('Figures')
    cv2.setMouseCallback('Figures', click_event) #permite interactuar click izquierdo con el usuario

    while True: #inicia el ciclo 
        ret, frame = cap.read() 
        if not ret: # rompe el codigo si no captura ningun video
            print("Fallo la captura del video. Saliendo...")
            break
        
        frame = undistort_frame(frame, matriz, dist) # elimina distorsion de la imagen a tiempo real 

        draw_lines(frame, points, is_figure_closed) #dibuja las lineas cuando haya 2 coordenadas o mas 

        key = handle_user_interaction(matriz) # nueva interaccion con el usuario para diferentes utilidades como cerrar el programa
        #borrar la figura o cerrar la figura.

        if key == ord('q'):  # Si el usuario presiona 'q', salir del bucle
            break

        cv2.imshow('Figures', frame)

def load_calibration_parameters_from_json_file(
        args:arg.ArgumentParser)->None:
    """
    Load camera calibration parameters from a JSON file.

    Args:
        args: Parsed command-line arguments.

    Returns:
        camera_matrix: Camera matrix.
        distortion_coefficients: Distortion coefficients.

    This function may raise a warning if the JSON file 
    does not exist. In such a case, the program finishes.
    """

    # Check if JSON file exists
    json_filename = args.json_file
    check_file = os.path.isfile(json_filename)

    # If JSON file exists, load the calibration parameters
    if check_file:
        f = open(json_filename)
        json_data = json.load(f)
        f.close()
        
        camera_matrix = np.array(json_data['camera_matrix'])
        distortion_coefficients = np.array(json_data['distortion_coefficients'])
        return camera_matrix, distortion_coefficients
    
    # Otherwise, the program finishes
    else:
        print(f"The file {json_filename} does not exist!")
        sys.exit(-1)

def undistort_frame(frame:cv2, matriz:np, dist:np)->cv2:
    """
    Aplica la corrección de distorsión a un frame dado.

    Args:
        frame: El frame de imagen a corregir.
        mtx: La matriz de calibración de la camara.
        dist: Los coeficientes de distorsión.

    Returns:
        El frame corregido.
    """
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(matriz, dist, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, matriz, dist, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]
    return undistorted_frame

def close_windows(cap:cv2.VideoCapture)->None:
    """
    Cierra todas las ventanas de visualización.

    Parámetros:
        cap: El objeto 'VideoCapture' que se desea cerrar.

    Returns: Nada
    """
    
    # Destroy all visualisation windows
    cv2.destroyAllWindows()
    # Destroy 'VideoCapture' object
    cap.release()
    return None

def resize_calibration(mtx:np)->np:
    """
    Ajusta la matriz de calibración de la cámara según una nueva resolución 
    de imagen.

    Parámetros:
        mtx: La matriz de calibración de la cámara original.

    Retorna:
        matriz:La matriz de calibración de la cámara ajustada según la 
        nueva resolución.

    La función recalcula los parámetros de la matriz de calibración 
    (fx, fy, cx, cy) segúnla nueva resolución de imagen, conservando 
    la misma relación de aspecto. 
    """
    global matriz
    resx=4080/640
    resy=3072/480

    cx = mtx[0, 2]
    cy = mtx[1, 2]
    fx = mtx[0, 0]
    fy = mtx[1, 1]

    cx=cx/resx
    fx=fx/resx
    cy=cy/resy
    fy=fy/resy

    matriz=np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
    
    return matriz

def pipeline():
    """
    Función principal que ejecuta el programa de una forma ordenada.

    Esta función ejecuta el codigo completo del programa, incluyendo la 
    interacción con el usuario para obtener los parámetros necesarios, 
    inicializar la cámara, cargar los parámetros de calibración, ajustar
    la matriz de calibración, ejecutar el bucle principal para el cálculo 
    del perímetro de las figuras en tiempo real y finalmente cerrar todas 
    las ventanas y cerrar la captura de video.
    """
    
    args = user_interaction()
    cap = initialise_camera(args)
    mtx,dist=load_calibration_parameters_from_json_file(args)
    matriz=resize_calibration(mtx)
    main_loop_perimeter(cap,matriz,dist,args)
    close_windows(cap)

if __name__ == "__main__":
    pipeline()