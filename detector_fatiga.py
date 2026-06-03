# detector_fatiga.py

from flask import Flask, render_template_string, request, jsonify
import cv2
import pygame
import numpy as np
import pandas as pd
import mediapipe as mp
import time
import paho.mqtt.client as mqtt

from threading import Thread

# -----------------------------------------------------------------------------
# FLASK

app = Flask(__name__)

# -----------------------------------------------------------------------------
# MQTT HIVEMQ

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

# -----------------------------------------------------------------------------

ALARME_ON = False

JANELA_MEDIA = 10

CONTADOR_QUADROS_SONOLENCIA = 0

# -----------------------------------------------------------------------------
# CONTADORES

contador_bostezos = 0
contador_microsono = 0
contador_cabecadas = 0
contador_parpadeos = 0

# -----------------------------------------------------------------------------
# CONTROLADORES

bostezo_ativo = False
cabecada_ativa = False
parpadeo_ativo = False
microsono_ativo = False

# -----------------------------------------------------------------------------

distancias_olho_direito = []
distancias_olho_esquerdo = []

# -----------------------------------------------------------------------------

dados = pd.DataFrame(columns=["Media_Abertura_Olhos"])

# -----------------------------------------------------------------------------
# OJOS

OLHO_DIREITO = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

OLHO_ESQUERDO = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

# -----------------------------------------------------------------------------
# BOCA

PUNTO_BOCA_SUPERIOR = 13
PUNTO_BOCA_INFERIOR = 14

# -----------------------------------------------------------------------------
# MEDIAPIPE

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------------------------------------------------------
# ALARMA

pygame.init()

ALARME = pygame.mixer.Sound('alarme.wav')

# -----------------------------------------------------------------------------

def alerta_sonoro():

    global ALARME_ON

    while ALARME_ON:

        ALARME.play()

        pygame.time.wait(2000)

# -----------------------------------------------------------------------------

def calcular_altura_olhos(puntos):

    A = np.linalg.norm(np.array(puntos[15]) - np.array(puntos[1]))
    B = np.linalg.norm(np.array(puntos[14]) - np.array(puntos[2]))
    C = np.linalg.norm(np.array(puntos[13]) - np.array(puntos[3]))
    D = np.linalg.norm(np.array(puntos[12]) - np.array(puntos[4]))
    E = np.linalg.norm(np.array(puntos[11]) - np.array(puntos[5]))
    F = np.linalg.norm(np.array(puntos[10]) - np.array(puntos[6]))
    G = np.linalg.norm(np.array(puntos[9]) - np.array(puntos[7]))
    H = np.linalg.norm(np.array(puntos[0]) - np.array(puntos[8]))

    altura_olhos = (A + B + C + D + E + F + G) / (2 * H)

    return altura_olhos

# -----------------------------------------------------------------------------

def calcular_abertura_boca(puntos):

    boca_superior = puntos[PUNTO_BOCA_SUPERIOR]

    boca_inferior = puntos[PUNTO_BOCA_INFERIOR]

    distancia = np.linalg.norm(
        np.array(boca_superior) - np.array(boca_inferior)
    )

    return distancia

# -----------------------------------------------------------------------------
# HTML

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>Detector Fatiga</title>

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<style>

body{
    background:#ece8f3;
    font-family:Arial;
    text-align:center;
    padding:20px;
}

video{
    width:90%;
    max-width:500px;
    border-radius:20px;
    border:4px solid #00ff99;
    background:black;
}

button{
    padding:15px 30px;
    border:none;
    border-radius:10px;
    background:#00cc88;
    color:white;
    font-size:20px;
}

.card{
    background:white;
    padding:15px;
    margin:10px;
    border-radius:15px;
}

.alerta{
    font-size:30px;
    font-weight:bold;
}

</style>

</head>

<body>

<h1>Detector Inteligente Fatiga</h1>

<button onclick="iniciarCamara()">
INICIAR CAMARA
</button>

<br><br>

<video id="video"
autoplay
playsinline
muted></video>

<canvas id="canvas"
style="display:none;"></canvas>

<div class="alerta" id="estado">

Esperando camara...

</div>

<div class="card">
Fatiga: <span id="fatiga">0</span>%
</div>

<div class="card">
Parpadeos: <span id="parpadeos">0</span>
</div>

<div class="card">
Bostezos: <span id="bostezos">0</span>
</div>

<div class="card">
Microsueños: <span id="microsonos">0</span>
</div>

<audio id="alarma" src="/alarma"></audio>

<script>

const video = document.getElementById('video');

const canvas = document.getElementById('canvas');

const estado = document.getElementById('estado');

const alarma = document.getElementById('alarma');

let cameraStarted = false;

// -----------------------------------------------------

async function iniciarCamara(){

    try{

        const stream =
        await navigator.mediaDevices.getUserMedia({

            video:{
                facingMode:"user"
            },

            audio:false

        });

        video.srcObject = stream;

        cameraStarted = true;

        estado.innerHTML = "Camara iniciada";

        estado.style.color = "green";

    }catch(error){

        alert("No se pudo abrir la camara");

    }

}

// -----------------------------------------------------

setInterval(async ()=>{

    if(!cameraStarted) return;

    canvas.width = video.videoWidth;

    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');

    ctx.drawImage(video,0,0);

    canvas.toBlob(async(blob)=>{

        const formData = new FormData();

        formData.append("frame", blob);

        const response = await fetch('/procesar', {

            method:'POST',

            body:formData

        });

        const data = await response.json();

        document.getElementById("fatiga").innerHTML =
        data.fadiga;

        document.getElementById("parpadeos").innerHTML =
        data.parpadeos;

        document.getElementById("bostezos").innerHTML =
        data.bostezos;

        document.getElementById("microsonos").innerHTML =
        data.microsonos;

        if(data.alerta){

            estado.innerHTML = "PELIGRO FATIGA";

            estado.style.color = "red";

            alarma.play();

        }else{

            estado.innerHTML = "Normal";

            estado.style.color = "green";

        }

    }, 'image/jpeg');

},500);

</script>

</body>

</html>

"""

# -----------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(HTML)

# -----------------------------------------------------------------------------

@app.route('/alarma')
def alarma():
    return open("alarme.wav", "rb").read()

# -----------------------------------------------------------------------------

@app.route('/procesar', methods=['POST'])
def procesar():

    global contador_bostezos
    global contador_microsono
    global contador_cabecadas
    global contador_parpadeos
    global CONTADOR_QUADROS_SONOLENCIA
    global ALARME_ON

    file = request.files['frame']

    npimg = np.frombuffer(file.read(), np.uint8)

    original_frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    original_frame = cv2.flip(original_frame, 1)

    rgb_frame = cv2.cvtColor(
        original_frame,
        cv2.COLOR_BGR2RGB
    )

    resultados = face_mesh.process(rgb_frame)

    fadiga = 0

    alerta = False

    if resultados.multi_face_landmarks:

        todos_pontos_referencia = np.array([

            np.multiply(
                [p.x, p.y],
                [640, 480]
            ).astype(int)

            for p in resultados.multi_face_landmarks[0].landmark

        ])
 # ------------------------------------------------
        # TRACKING
        olho_direito = todos_pontos_referencia[OLHO_DIREITO]

        olho_esquerdo = todos_pontos_referencia[OLHO_ESQUERDO]
        
        # DETECCION OJOS

        distancia_olho_direito = calcular_altura_olhos(
            olho_direito
        )

        distancia_olho_esquerdo = calcular_altura_olhos(
            olho_esquerdo
        )

        distancias_olho_direito.append(
            distancia_olho_direito
        )

        distancias_olho_esquerdo.append(
            distancia_olho_esquerdo
        )

        if len(distancias_olho_direito) > JANELA_MEDIA:
            distancias_olho_direito.pop(0)

        if len(distancias_olho_esquerdo) > JANELA_MEDIA:
            distancias_olho_esquerdo.pop(0)

        media_olho_direito = np.mean(
            distancias_olho_direito
        )

        media_olho_esquerdo = np.mean(
            distancias_olho_esquerdo
        )

        media_abertura_olhos = (

            media_olho_direito +
            media_olho_esquerdo

        ) / 2

        # ------somnolencia-----------------------------------------

        if media_abertura_olhos < 0.20:

            CONTADOR_QUADROS_SONOLENCIA += 1

            contador_parpadeos += 1

            if CONTADOR_QUADROS_SONOLENCIA >= 15:

                contador_microsono += 1

        else:

            CONTADOR_QUADROS_SONOLENCIA = 0

            ALARME_ON = False

        # ---------------------------------------------------------

        abertura_boca = calcular_abertura_boca(
            todos_pontos_referencia
        )

        if abertura_boca > 25:

            contador_bostezos += 1

        # ---------------------------------------------------------

        fadiga = min(
            int((CONTADOR_QUADROS_SONOLENCIA / 15) * 100),
            100
        )

        # ---------------------------------------------------------

        if fadiga >= 80:

            alerta = True

            client.publish(TOPIC, "ON")

            if not ALARME_ON:

                ALARME_ON = True

                t = Thread(target=alerta_sonoro)

                t.daemon = True

                t.start()

        else:

            client.publish(TOPIC, "OFF")

            ALARME_ON = False

            

    return jsonify({

        "alerta": alerta,

        "fadiga": fadiga,

        "parpadeos": contador_parpadeos,

        "bostezos": contador_bostezos,

        "microsonos": contador_microsono

    })

# -----------------------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )