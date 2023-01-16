from pathlib import Path

import os
import torch
import random
import requests
import mutagen
from flask import Flask, request, Response, abort
from flask_cors import CORS, cross_origin
from unidecode import unidecode
from fake_useragent import UserAgent


from osu_dreamer.osu.hit_objects import TimingPoint
from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.model import Model
from osu_dreamer import generate_mapset

app = Flask(__name__)
CORS(app, support_credentials=True)


def random_hex_string(num): return hex(random.randrange(16**num))[2:]


@app.route("/generate", methods=["GET"])
@cross_origin(supports_credentials=True)
def generate():
    # get args
    args = request.args.to_dict()
    audio_url = args.get("audio")
    bpm = int(args.get("bpm", 110))
    title = args.get("title")
    artist = args.get("artist")
    samples = int(args.get("samples", 1))

    # download & write audio
    response = requests.get(audio_url, headers={
        "User-Agent": UserAgent().chrome})

    audio_path = Path(f"{random_hex_string(7)}.mp3")
    audio_handle = open(audio_path, 'wb')

    audio_handle.write(response.content)

    # parse file metadata
    tags = mutagen.File(audio_path, easy=True)

    if title == None or title == "null":
        try:
            title = tags['title'][0]
        except KeyError:
            title = "unknown"

    if artist == None or artist == "null":
        try:
            artist = tags['artist'][0]
        except KeyError:
            artist = "unkown"

    # generate & write beatmapset
    try:
        mapset = generate_mapset(
            model,  # model
            audio_path,  # audio
            bpm,  # timing
            samples,  # samples (how many to generate)
            unidecode(title),  # title
            unidecode(artist),  # artist
            16,  # sample steps
            True,  # ddim
            "./generated_beatmaps" # download directory
        )
    except:
        audio_handle.close()
        os.remove(audio_path)
        abort(404)
    
    file_handle = open(mapset, "rb")

    # stream files and cleanup
    def stream_and_remove_file():
        yield from file_handle
        file_handle.close()
        audio_handle.close()
        os.remove(mapset)
        os.remove(audio_path)

    return Response(
        stream_and_remove_file(),
        headers={'Content-Disposition': (f'attachment; filename="{Path(mapset).stem}.osz"')},
        mimetype="application/x-osu-archive",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='generate osu!std maps from raw audio')
    parser.add_argument('model_path', metavar='MODEL_PATH',
                        type=Path, help='trained model (.ckpt)')

    server_args = parser.add_argument_group('server arguments')
    server_args.add_argument('--port', type=int,
                             default=8000, help='port to run the server on')

    args = parser.parse_args()

    # load model
    model = Model.load_from_checkpoint(args.model_path).eval()

    if torch.cuda.is_available():
        print('using GPU accelerated inference')
        model = model.cuda()
    else:
        print('WARNING: no GPU found - inference will be slow')

    app.run(host="0.0.0.0", port=args.port)
