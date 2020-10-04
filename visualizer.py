from visdom import server, Visdom   # Forcefully toggled off debug mode in source
import multiprocessing
from zmq.eventloop import ioloop
ioloop.install()
import tornado.ioloop
import tornado.httpserver 
import webbrowser
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
        print('If running locally, <server-id> = localhost.\n')
        loop.start()
        http_server.stop()
        logging.info('Ended instance')

    def launch_instance(self):
        self.proc = multiprocessing.Process(
            target=self.instance_runner)
        webbrowser.open('http://localhost:8097')
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

    def plot(self, line_name, win_name, x, y, is_log=True):
        y_ax = ('Log ' if is_log else '') + 'Loss'
        if win_name not in self.windows:
            yaxis_dict = {'title': y_ax}
            if is_log:
                yaxis_dict['type'] = 'log'
            else:
                yaxis_dict['type'] = 'linear'

            self.windows[win_name] = self.vis.line(X=torch.tensor([x], 
                dtype=torch.float), Y=torch.tensor([y], dtype=torch.float), 
                opts=dict(legend=[line_name], title=win_name, xlabel='Epochs',
                ylabel=y_ax, layoutopts={'plotly':{'yaxis':yaxis_dict}}))
        else:
            self.vis.line(X=torch.tensor([x]), Y=torch.tensor([y]), 
                          win=self.windows[win_name], name=line_name, 
                          update = 'append')

    def show_image(self, title, tensor_stack):
        tensor_stack = tensor_stack.type(torch.ByteTensor)
        self.windows[title] = self.vis.image(tensor_stack, win=title, opts={'caption':title})