import traceback
from PIL import Image
import io
import os
from keras.models import load_model
from PIL import Image
import numpy as np
import pika
import ssl
#from receive import main
from path import PATH, MQ

# load model
model = load_model('brain-tumor-model.h5')
modelRetino = load_model('derma_diseases_detection.h5')


def names(number):
    if(number == 0):
        return "no_tumor"
    elif(number == 1):
        return "glioma-tumor"
    elif(number == 2):
        return "meningioma-tumor"
    elif(number == 3):
        return "meningioma-tumor"


def retinas(number):
    if(number == 0):
        return 'Normal'
    elif(number == 1):
        return 'Mild'
    elif(number == 2):
        return 'Moderate'
    elif(number == 3):
        return 'Severe'
    elif(number == 4):
        return 'Proliferative'


def riddleRetina(n):
    oracle = ""
    # load image and verify if it's an image
    try:
        binary_data = io.BytesIO(n)

        img = Image.open(binary_data)

    except:
        oracle = "not an image"

    try:
        x = np.array(img.resize((224, 224)))
        x = x.reshape(1, 224, 224, 3)
        answ = modelRetino.predict_on_batch(x)
        classification = np.where(answ == np.amax(answ))[1][0]
        oracle = str(answ[0][classification]*100) + \
            '% Confidence This Is ' + retinas(classification)
    except:
        oracle = "bad image"
    return oracle


def riddle(n):
    oracle = ""
    # load image and verify if it's an image
    try:
        binary_data = io.BytesIO(n)

        img = Image.open(binary_data)
    except:
        oracle = "not an image"

    try:
        x = np.array(img.resize((150, 150)))
        x = x.reshape(1, 150, 150, 3)
        answ = model.predict_on_batch(x)
        classification = np.where(answ == np.amax(answ))[1][0]
        oracle = str(answ[0][classification]*100) + \
            '% Confidence This Is ' + names(classification)
    except:
        oracle = "bad image"
    return oracle


context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS)
ssl_options = pika.SSLOptions(context, MQ)
connection = pika.BlockingConnection(pika.ConnectionParameters(
    ssl_options=ssl_options, port=5671, host=MQ, credentials=pika.PlainCredentials("kino", "1593578915935789")))
# connection = pika.BlockingConnection(
#     pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='rpc_brain')
channel.queue_declare(queue='rpc_retino')


def brain(ch, method, props, body):

    #  n = int.from_bytes(body, "big")

    print("classifying image...")
    response = riddle(body)
    print(response)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(
                         correlation_id=props.correlation_id),
                     body=str(response))

    ch.basic_ack(delivery_tag=method.delivery_tag)


def retino(ch, method, props, body):

    print("classifying image...")
    response = riddleRetina(body)
    print(response)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(
                         correlation_id=props.correlation_id),
                     body=str(response))

    ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='rpc_brain', on_message_callback=brain)
    channel.basic_consume(queue='rpc_retino', on_message_callback=retino)
    print(" [x] Awaiting RPC requests")
    channel.start_consuming()
    print(' [*] Waiting for messages. To exit press CTRL+C')


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except:
        main()
