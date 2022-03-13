import logging
import click
import entity_vocab
import esa_model 

logger = logging.getLogger(__name__)


@click.group()
def cli():
    fmt = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=fmt)

cli.add_command(entity_vocab.build_entity_vocab)
cli.add_command(esa_model.setup_esa_model)
cli.add_command(esa_model.run_model)

if __name__ == "__main__":
    cli()