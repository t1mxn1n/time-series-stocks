import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()


def img_to_base64(path_img):
    with open(f"{path_img}", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string


def delete_img(path_img):
    os.remove(path_img)


def imgur_upload(path_img):
    url = "https://api.imgur.com/3/image"

    payload = {'image': img_to_base64(path_img),
               'description': 'face detection (lab work for cloud technology)'}

    headers = {
        'Authorization': f'Client-ID {os.getenv("imgur_api")}'
    }

    response = requests.post(url, headers=headers, data=payload)
    if not response:
        return {'error': 'not response', 'code': response.status_code}
    response_json = response.json()
    if response.status_code == 200:
        delete_img(path_img)
        return {'link': response_json['data']['link']}
    return {'error': response_json}
