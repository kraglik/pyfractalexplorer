import pyopencl as cl
import pygame
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
                 height=500):

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

        self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption('Fractal Explorer')
        pygame.mouse.set_visible(0)

        self.camera = Camera(
            self.device, self.context, self.queue
        )
        self.render = Render(
            self.device,
            self.context,
            self.queue,
            self.camera,
            fractals[0], None, None,
            width=self.width,
            height=self.height
        )

        self.run()

    def render_fractal(self):
        self.render.render()

    def save_image(self, path):
        self.render.save(path)

    def run(self):
        surface = pygame.pixelcopy.make_surface(self.render.host_buffer[:, :, :3])

        mouse_speed = 0.025

        shift_forward = 0
        shift_right = 0

        movement_speed = 0.05

        epsilon = self.render.epsilon

        while True:
            self.render.render()
            pygame.surfarray.blit_array(surface, self.render.host_buffer[:, :, :3])
            self.screen.blit(surface, (0, 0))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == QUIT:
                    return

                elif event.type == MOUSEMOTION:
                    mouse_position = pygame.mouse.get_pos()
                    pygame.mouse.set_pos((self.width / 2, self.height / 2))

                    self.camera.rotate(
                        (mouse_position[0] - self.width // 2) / (self.width // 2) * mouse_speed,
                        (mouse_position[1] - self.height // 2) / (self.height // 2) * mouse_speed
                    )

                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        return

                    if event.key == K_w:
                        shift_forward += movement_speed
                    elif event.key == K_s:
                        shift_forward -= movement_speed
                    elif event.key == K_a:
                        shift_right -= movement_speed
                    elif event.key == K_d:
                        shift_right += movement_speed

                    elif event.key == K_UP:
                        movement_speed *= 2
                        shift_forward *= 2
                        shift_right *= 2
                    elif event.key == K_DOWN:
                        movement_speed /= 2
                        shift_forward /= 2
                        shift_right /= 2

                    elif event.key == K_RIGHT:
                        epsilon *= 2
                    elif event.key == K_LEFT:
                        epsilon /= 2

                elif event.type == KEYUP:
                    if event.key == K_w:
                        shift_forward -= movement_speed
                    elif event.key == K_s:
                        shift_forward += movement_speed
                    elif event.key == K_a:
                        shift_right += movement_speed
                    elif event.key == K_d:
                        shift_right -= movement_speed

            self.camera.position += self.camera.direction * shift_forward
            self.camera.position += self.camera.right * shift_right
            self.render.epsilon = epsilon
