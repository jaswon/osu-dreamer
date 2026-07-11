
try:
    from jaxtyping import install_import_hook
    install_import_hook(__package__, "beartype.beartype") # type: ignore
except ImportError:
    # no runtime type assertions
    pass

import click

from .scripts.predict import predict
from .scripts.fit_denoiser import fit_denoiser
from .scripts.fit_latent import fit_latent
from .scripts.fit_style import fit_style
from .scripts.generate_data import generate_data
from .scripts.encode_latents import encode_latents
from .scripts.export_inference import export_inference

@click.group()
def main():
    pass

main.add_command(generate_data)
main.add_command(encode_latents)
main.add_command(fit_denoiser)
main.add_command(fit_latent)
main.add_command(fit_style)
main.add_command(predict)
main.add_command(export_inference)

if __name__ == "__main__":
    main()