from visdom import server, Visdom   # Forcefully toggled off debug mode in source
import multiprocessing
from zmq.eventloop import ioloop
ioloop.install()  # Needs to happen before any tornado imports!
import tornado.ioloop      # noqa E402: gotta install ioloop first
import tornado.httpserver  # noqa E402: gotta install ioloop first
import logging
import torch
import time

import torch

class VisServer(object):
    def __init__(self, port):
        self.port = port
        self.proc = None
        logging.basicConfig(filename='vis_server.log', level=logging.DEBUG,
                            filemode='w')

    def instance_runner(self):
        loop = ioloop.IOLoop()
        app = server.Application(port=self.port)

        http_server = tornado.httpserver.HTTPServer(
            app, max_buffer_size=1024 ** 3)
        http_server.listen(self.port)
        logging.info('Started instance')
        print('\nSearch "<server-id>:'+str(self.port)+'" in browser for a '\
              'training visualization.')
        print('If running locally, <server-id> = localhost.')
        loop.start()
        http_server.stop()
        logging.info('Ended instance')

    def launch_instance(self):
        self.proc = multiprocessing.Process(
            target=self.instance_runner)
        self.proc.start()


class Visualizer(object):
    def __init__(self, port=8097):
        self.server = VisServer(port)
        self.vis = None
        self.windows = {}

    def start(self):
        self.server.launch_instance()
        time.sleep(0.01) # short pause
        self.vis = Visdom()

    def stop(self):
        self.server.proc.terminate()

    def plot(self, y_ax, line_name, win_name, x, y, color=None):
        # color is a np uint8 array shape 1x3
        if win_name not in self.windows:
            yaxis_dict = {'title': y_ax}
            if y_ax.split(' ')[0].lower() == 'log':
                yaxis_dict['type'] = 'log'
            else:
                yaxis_dict['type'] = 'linear'

            self.windows[win_name] = self.vis.line(X=torch.tensor([x], 
                dtype=torch.float), Y=torch.tensor([y], dtype=torch.float), 
                opts=dict(legend=[line_name], title=win_name, xlabel='Epochs',
                    ylabel=y_ax, linecolor=color, layoutopts={'plotly':
                                                {'yaxis':yaxis_dict}}))
        else:
            self.vis.line(X=torch.tensor([x]), Y=torch.tensor([y]), 
                          win=self.windows[win_name], name=line_name, 
                          update = 'append')

    def show_image(self, title, tensor_stack):
        tensor_stack = tensor_stack.type(torch.ByteTensor)
        self.windows[title] = self.vis.image(tensor_stack, win=title, opts={'caption':title})