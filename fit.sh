#!/bin/bash

case $1 in
    latent)
        echo "training latent"
        ;;
    denoiser)
        echo "training denoiser"
        ;;
    *)
        echo "usage: $0 {latent,denoiser}"
        exit 1
        ;;
esac

if [ -n "$2" ]; then
    LAST_RUN=$2
fi

while : ; do
    flags=()

    poetry run python -m osu_dreamer fit-$1 ${LAST_RUN:+--ckpt-path="$LAST_RUN"}
    if [ $? -ne 139 ]; then
        break
    fi
    LAST_RUN=$(find $(ls -td runs/latent/* | head -n1) -name '*.ckpt')
done