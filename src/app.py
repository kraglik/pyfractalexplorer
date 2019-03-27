import math

import pyopencl as cl
import pygame
import time
from pygame.locals import *
import numpy as np

from .fractals import Fractal, fractals
from .render import Render
from .camera import Camera


class App:
    def __init__(self,
                 platform_id=0,
                 device_id=0,
                 device_type=cl.device_type.ALL,
                 width=500,
                 height=500,
                 fullscreen=False):

        self.width = width
        self.height = height

        self.platform_id = platform_id or 0
        self.device_id = device_id or 0
        self.device_type = device_type or cl.device_type.ALL

        self.platform = cl.get_platforms()[self.platform_id]
        self.device = self.platform.get_devices()[self.device_id]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        pygame.init()

        if fullscreen:
            info = pygame.display.Info()

            self.width, self.height = info.current_w, info.current_h

            self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption('Fractal Explorer')
        pygame.mouse.set_visible(0)

        self.camera = Camera(
            self.device, self.context, self.queue, mouse_speed=5.0
        )
        self.render = Render(
            self.device,
            self.context,
            self.queue,
            self.camera,
            width=self.width,
            height=self.height
        )

        self.run()

    def render_fractal(self):
        self.render.render()

    def save_image(self, path):
        self.render.save(path)

    def run(self):
        surface = pygame.pixelcopy.make_surface(
            self.render.host_buffer.reshape((self.width, self.height, 4))[:, :, :3]
        )

        default_movement_speed = 0.5
        movement_speed = default_movement_speed

        speed_stack = []

        epsilon = self.render.epsilon

        key_map = {
            'w': False,
            'a': False,
            's': False,
            'd': False,
            'lctrl': False,
            'space': False
        }

        fractal_index = 0

        data = [{
            "movement_speed": default_movement_speed,
            "epsilon": epsilon,
            "position": fractal.get_initial_camera_position(),
            "target": fractal.get_initial_camera_target()
        } for fractal in self.render.fractals]

        self.render.fractal = self.render.fractals[fractal_index]
        self.camera.position = data[fractal_index]["position"]
        self.camera.look_at(data[fractal_index]["target"])

        initial_time = time.time()

        while True:
            time_before_render = time.time()

            self.render.fractal_by_name["Kaleido Fractal"].set_parameters(
                {
                    "time": float(time.time() - initial_time)
                },
                self.render.fractal_by_name["Kaleido Fractal"].get_color()
            )

            self.render.render()
            pygame.surfarray.blit_array(
                surface,
                self.render.host_buffer.reshape((self.width, self.height, 4))[:, :, :3]
            )
            self.screen.blit(surface, (0, 0))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == QUIT:
                    return

                elif event.type == MOUSEMOTION:
                    mouse_position = pygame.mouse.get_pos()
                    pygame.mouse.set_pos((self.width / 2, self.height / 2))

                    self.camera.rotate(
                        (mouse_position[0] - self.width // 2) / (self.width // 2),
                        (mouse_position[1] - self.height // 2) / (self.height // 2)
                    )

                elif event.type == KEYDOWN:

                    if event.key == K_ESCAPE:
                        return

                    elif event.key == K_PAGEUP:
                        data[fractal_index] = {
                            "movement_speed": default_movement_speed,
                            "epsilon": epsilon,
                            "position": self.camera.position,
                            "target": self.camera.position + self.camera.direction
                        }

                        fractal_index = (fractal_index + 1) % len(data)

                        movement_speed = data[fractal_index]["movement_speed"]
                        epsilon = data[fractal_index]["epsilon"]
                        self.camera.position = data[fractal_index]["position"]
                        self.camera.look_at(data[fractal_index]["target"])

                    elif event.key == K_PAGEDOWN:
                        data[fractal_index] = {
                            "movement_speed": default_movement_speed,
                            "epsilon": epsilon,
                            "position": self.camera.position,
                            "target": self.camera.position + self.camera.direction
                        }

                        fractal_index = (fractal_index - 1) % len(data)

                        movement_speed = data[fractal_index]["movement_speed"]
                        epsilon = data[fractal_index]["epsilon"]
                        self.camera.position = data[fractal_index]["position"]
                        self.camera.look_at(data[fractal_index]["target"])

                    elif event.key == K_w:
                        key_map['w'] = True
                    elif event.key == K_s:
                        key_map['s'] = True
                    elif event.key == K_a:
                        key_map['a'] = True
                    elif event.key == K_d:
                        key_map['d'] = True
                    elif event.key == K_LCTRL:
                        key_map['lctrl'] = True
                    elif event.key == K_SPACE:
                        key_map['space'] = True
                    elif event.key == K_q:
                        speed_stack.insert(0, movement_speed)
                    elif event.key == K_e:
                        if len(speed_stack) > 0:
                            movement_speed = speed_stack[0]
                            speed_stack = speed_stack[1:]
                        else:
                            movement_speed = default_movement_speed

                    elif event.key == K_UP:
                        movement_speed *= 2
                    elif event.key == K_DOWN:
                        movement_speed /= 2

                    elif event.key == K_RIGHT:
                        epsilon *= 2
                    elif event.key == K_LEFT:
                        epsilon /= 2

                elif event.type == KEYUP:
                    if event.key == K_w:
                        key_map['w'] = False
                    elif event.key == K_s:
                        key_map['s'] = False
                    elif event.key == K_a:
                        key_map['a'] = False
                    elif event.key == K_d:
                        key_map['d'] = False
                    elif event.key == K_LCTRL:
                        key_map['lctrl'] = False
                    elif event.key == K_SPACE:
                        key_map['space'] = False

                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 4:
                        self.camera.zoom *= 1.1
                    elif event.button == 5:
                        self.camera.zoom *= 1 / 1.1
                    elif event.button == 2:
                        self.camera.zoom = 1.0

            delta = time.time() - time_before_render

            shift = np.array([0, 0, 0], dtype=np.float32)

            if key_map['w']:
                shift += self.camera.direction
            if key_map['s']:
                shift -= self.camera.direction
            if key_map['a']:
                shift -= self.camera.right
            if key_map['d']:
                shift += self.camera.right
            if key_map['space']:
                shift += self.camera.up
            if key_map['lctrl']:
                shift -= self.camera.up

            shift_l = np.linalg.norm(shift)

            if shift_l > 0:
                shift /= shift_l

            shift *= delta * movement_speed

            self.camera.position += shift
            self.render.epsilon = epsilon / self.camera.zoom
            self.render.fractal = self.render.fractals[fractal_index]
