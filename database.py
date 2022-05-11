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
classes = os.listdir(PATH+'Training')


def names(number):
    if(number == 0):
        return classes[0]
    elif(number == 1):
        return classes[1]
    elif(number == 2):
        return classes[2]
    elif(number == 3):
        return classes[3]


def riddle(n):
    # load image and verify if it's an image
    try:
        binary_data = io.BytesIO(n)

        img = Image.open(binary_data)
    except:
        return "not an image"

    x = np.array(img.resize((150, 150)))
    x = x.reshape(1, 150, 150, 3)
    answ = model.predict_on_batch(x)
    classification = np.where(answ == np.amax(answ))[1][0]
    oracle = str(answ[0][classification]*100) + \
        '% Confidence This Is ' + names(classification)
    return oracle


context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS)
ssl_options = pika.SSLOptions(context, MQ)
connection = pika.BlockingConnection(pika.ConnectionParameters(
    ssl_options=ssl_options, port=5671, host=MQ, credentials=pika.PlainCredentials("kino", "1593578915935789")))
channel = connection.channel()
channel.queue_declare(queue='rpc_queue')


def on_request(ch, method, props, body):

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


def main():
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='rpc_queue', on_message_callback=on_request)
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
