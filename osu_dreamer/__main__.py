
from jaxtyping import install_import_hook
install_import_hook(__package__, "beartype.beartype") # type: ignore

import click

from .scripts.predict import predict
from .scripts.fit_denoiser import fit_denoiser
from .scripts.fit_latent import fit_latent
from .scripts.generate_data import generate_data

@click.group()
def main():
    pass

main.add_command(generate_data)
main.add_command(fit_denoiser)
main.add_command(fit_latent)
main.add_command(predict)

if __name__ == "__main__":
    main()