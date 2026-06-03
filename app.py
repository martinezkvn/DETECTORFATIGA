from flask import Flask, render_template, request, jsonify
from threading import Thread
import cv2
import numpy as np
import mediapipe as mp
import time
import paho.mqtt.client as mqtt
import pygame
import sqlite3

# --------------------------------------------------------
# APP

app = Flask(__name__)

pygame.init()

# --------------------------------------------------------
# BASE DE DATOS

def conectar_db():
    return sqlite3.connect('detector_fatiga.db')

def crear_tabla():
    conn = sqlite3.connect('detector_fatiga.db')
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sesiones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            fatiga INTEGER,
            parpadeos INTEGER,
            bostezos INTEGER,
            microsonos INTEGER,
            estado TEXT
        )
    """)
    conn.commit()
    conn.close()

crear_tabla()

# --------------------------------------------------------
# MQTT

BROKER = "d9979a590ac64dd4bd436afd085eb837.s1.eu.hivemq.cloud"
PORT = 8883
TOPIC = "fatiga/alerta"

USERNAME = "detectorfatiga"
PASSWORD = "Jade123456"

client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)
client.tls_set()
client.connect(BROKER, PORT)
client.loop_start()

time.sleep(2)

# --------------------------------------------------------
# MEDIAPIPE

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------------------------------------------------------
# VARIABLES

contador_bostezos = 0
contador_parpadeos = 0
contador_microsonos = 0

CONTADOR_CUADROS_SOMNOLENCIA = 0

ALARMA_ACTIVA = False

VENTANA_MEDIA = 3

distancias_ojo_derecho = []
distancias_ojo_izquierdo = []

# --------------------------------------------------------
# OJOS

OJO_DERECHO = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

OJO_IZQUIERDO = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

# --------------------------------------------------------
# BOCA

PUNTO_BOCA_SUPERIOR = 13
PUNTO_BOCA_INFERIOR = 14

# --------------------------------------------------------
# ALARMA

def alerta_sonora():
    global ALARMA_ACTIVA
    while ALARMA_ACTIVA:
        pygame.mixer.Sound('alarme.wav').play()
        pygame.time.wait(2200)

# --------------------------------------------------------
# FUNCION OJOS

def calcular_altura_ojos(puntos):
    A = np.linalg.norm(np.array(puntos[15]) - np.array(puntos[1]))
    B = np.linalg.norm(np.array(puntos[14]) - np.array(puntos[2]))
    C = np.linalg.norm(np.array(puntos[13]) - np.array(puntos[3]))
    D = np.linalg.norm(np.array(puntos[12]) - np.array(puntos[4]))
    E = np.linalg.norm(np.array(puntos[11]) - np.array(puntos[5]))
    F = np.linalg.norm(np.array(puntos[10]) - np.array(puntos[6]))
    G = np.linalg.norm(np.array(puntos[9]) - np.array(puntos[7]))
    H = np.linalg.norm(np.array(puntos[0]) - np.array(puntos[8]))
    altura = (A+B+C+D+E+F+G)/(2*H)
    return altura

# --------------------------------------------------------
# FUNCION BOCA

def calcular_abertura_boca(puntos):
    boca_superior = puntos[PUNTO_BOCA_SUPERIOR]
    boca_inferior = puntos[PUNTO_BOCA_INFERIOR]
    distancia = np.linalg.norm(np.array(boca_superior) - np.array(boca_inferior))
    return distancia

# --------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

# --------------------------------------------------------

@app.route('/procesar', methods=['POST'])
def procesar():
    global contador_bostezos
    global contador_parpadeos
    global contador_microsonos
    global CONTADOR_CUADROS_SOMNOLENCIA
    global ALARMA_ACTIVA

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({
            "alerta": False, "fatiga": 0, "estado": "NORMAL",
            "parpadeos": 0, "bostezos": 0, "microsonos": 0, "landmarks": []
        })

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    alerta = False
    fatiga = 0
    estado = "NORMAL"
    landmarks = []

    if results.multi_face_landmarks:

        img_h, img_w = frame.shape[:2]

        puntos = np.array([
            np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
            for p in results.multi_face_landmarks[0].landmark
        ])

        for p in results.multi_face_landmarks[0].landmark:
            landmarks.append({
                "x": int(p.x * img_w),
                "y": int(p.y * img_h)
            })

        ojo_derecho = puntos[OJO_DERECHO]
        ojo_izquierdo = puntos[OJO_IZQUIERDO]

        distancia_ojo_derecho = calcular_altura_ojos(ojo_derecho)
        distancia_ojo_izquierdo = calcular_altura_ojos(ojo_izquierdo)

        distancias_ojo_derecho.append(distancia_ojo_derecho)
        distancias_ojo_izquierdo.append(distancia_ojo_izquierdo)

        if len(distancias_ojo_derecho) > VENTANA_MEDIA:
            distancias_ojo_derecho.pop(0)
        if len(distancias_ojo_izquierdo) > VENTANA_MEDIA:
            distancias_ojo_izquierdo.pop(0)

        media_ojo_derecho = np.mean(distancias_ojo_derecho)
        media_ojo_izquierdo = np.mean(distancias_ojo_izquierdo)
        media_abertura_ojos = (media_ojo_derecho + media_ojo_izquierdo) / 2

        if media_abertura_ojos < 0.35:
            CONTADOR_CUADROS_SOMNOLENCIA += 1
            contador_parpadeos += 1
            if CONTADOR_CUADROS_SOMNOLENCIA >= 5:
                contador_microsonos += 1
        else:
            CONTADOR_CUADROS_SOMNOLENCIA = 0
            ALARMA_ACTIVA = False

        abertura_boca = calcular_abertura_boca(puntos)
        if abertura_boca > 25:
            contador_bostezos += 1

        fatiga_base = 0
        if media_abertura_ojos < 0.35:
            fatiga_base += 60
        elif media_abertura_ojos < 0.45:
            fatiga_base += 40
        elif media_abertura_ojos < 0.55:
            fatiga_base += 20

        fatiga = min(int((CONTADOR_CUADROS_SOMNOLENCIA / 5) * 40) + fatiga_base, 100)

        if fatiga >= 80:
            alerta = True
            estado = "PELIGRO CRITICO"
            client.publish(TOPIC, "CRITICO")
            if not ALARMA_ACTIVA:
                ALARMA_ACTIVA = True
                t = Thread(target=alerta_sonora)
                t.daemon = True
                t.start()
        elif fatiga >= 60:
            estado = "PELIGRO"
            client.publish(TOPIC, "ALTO")
        elif fatiga >= 40:
            estado = "CANSANCIO"
            client.publish(TOPIC, "MEDIO")
        else:
            estado = "NORMAL"
            client.publish(TOPIC, "OFF")

    # ------------------------------------------------
    # GUARDAR EN BD

    print("Guardando...")
    try:
        conn = conectar_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sesiones (fatiga, parpadeos, bostezos, microsonos, estado) VALUES (?, ?, ?, ?, ?)",
            (fatiga, contador_parpadeos, contador_bostezos, contador_microsonos, estado)
        )
        conn.commit()
        cur.close()
        conn.close()
        print("OK")
    except Exception as e:
        print(f"Error BD: {e}")

    return jsonify({
        "alerta": alerta,
        "fatiga": fatiga,
        "estado": estado,
        "parpadeos": contador_parpadeos,
        "bostezos": contador_bostezos,
        "microsonos": contador_microsonos,
        "landmarks": landmarks
    })

# --------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
