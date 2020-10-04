import shutil
import webbrowser

from utils import new_folder
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

# https://gist.github.com/himaprasoonpt/66e990fbceea322b1548f823263b8b3e

class Visualizer(object):
    def __init__(self, log_dir, server_id='localhost', port=6006, clean_up=True):
        self.log_dir = log_dir
        self.board = program.TensorBoard()
        self.writer = None
        self.clean_up = clean_up

        self.board.configure(argv=[None, '--logdir', log_dir, 
            '--host', server_id, '--port', str(port)])

    def start(self):
        if self.clean_up:
            new_folder(self.log_dir)

        url = self.board.launch()
        print('\nLaunched TensorBoard at url:', url)
        webbrowser.open(url)
        self.writer = SummaryWriter(self.log_dir)

    def stop(self):
        self.writer.close()

        if self.clean_up:
            shutil.rmtree(self.log_dir)

    def get_writer(self):
        return self.writer