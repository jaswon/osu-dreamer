
import click

from osu_dreamer.models.inference.artifact import save_inference

@click.command()
@click.option(  '--latent-ckpt-path', type=click.Path(exists= True, dir_okay=False), default='latent.ckpt', help='path to the latent checkpoint')
@click.option('--denoiser-ckpt-path', type=click.Path(exists= True, dir_okay=False), default='denoiser.ckpt', help='path to the denoiser checkpoint')
@click.option(   '--style-ckpt-path', type=click.Path(exists= True, dir_okay=False), default='style.ckpt', help='path to the style model checkpoint')
@click.option(       '--output-path', type=click.Path(exists=False, dir_okay=False), default='inference.pt', help='artifact output path')
def export_inference(latent_ckpt_path: str, denoiser_ckpt_path: str, style_ckpt_path: str, output_path: str):
    """export an inference model artifact from training checkpoints"""

    save_inference(latent_ckpt_path, denoiser_ckpt_path, style_ckpt_path, output_path)