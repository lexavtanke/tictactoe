import click
from .fit import fit
from .play import play, self_play
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

@click.command()
@click.option("--mode", default="play", help="fit, play, self_play")
@click.option("--file_name", default="pytorch_dqn")
@click.option("--opponent_file_name", default="pytorch_dqn")
def main(mode="play", file_name="pytorch_dqn", opponent_file_name="pytorch_dqn"):
    if mode == "fit":
        fit(result_file_name=file_name)
        # sys.stdout.buffer.write(res)
        # sys.stdout.flush()
    elif mode == "play":
        play(file_name)
    elif mode == "self_play":
        self_play(file_name, opponent_file_name)

if __name__ == "__main__":
    main()