import pyopencl as cl
import pygame
import time
from datetime import datetime
from pygame.locals import *
import numpy as np
import os

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
            self.render.host_buffer.reshape((self.width, self.height, 3))
        )

        default_movement_speed = 0.5
        movement_speed = default_movement_speed

        speed_stack = []
        time_enabled = False

        epsilon = self.render.epsilon

        amplitude = 0.0

        fractal_index = 0

        data = [{
            "movement_speed": default_movement_speed,
            "epsilon": epsilon,
            "position": fractal.get_initial_camera_position(),
            "target": fractal.get_initial_camera_target(),
            "steps": self.render.ray_steps_limit,
            "amplitude": amplitude
        } for fractal in self.render.fractals]

        self.render.fractal = self.render.fractals[fractal_index]
        self.camera.position = data[fractal_index]["position"]
        self.camera.look_at(data[fractal_index]["target"])

        start_datetime = datetime.now()
        start_time = time.time()

        screenshots_path = "screenshots/"
        n_screenshots = 0

        while True:
            time_before_render = time.time()
            fractal_changed = False

            data[fractal_index] = {
                "movement_speed": default_movement_speed,
                "epsilon": epsilon,
                "position": self.camera.position,
                "target": self.camera.position + self.camera.direction,
                "steps": self.render.ray_steps_limit,
                "amplitude": amplitude
            }

            self.render.fractal.set_time(0.0 if not time_enabled else (time.time() - start_time))
            self.render.fractal.set_amplitude(amplitude)

            self.render.render()
            pygame.surfarray.blit_array(
                surface,
                self.render.host_buffer.reshape((self.width, self.height, 3))
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
                        (mouse_position[0] - self.width // 2) / 250,
                        (mouse_position[1] - self.height // 2) / 250
                    )

                elif event.type == KEYDOWN:

                    if event.key == K_ESCAPE:
                        return

                    elif event.key == K_t:
                        time_enabled = not time_enabled

                    elif event.key == K_COMMA:
                        amplitude *= 1 / 1.1

                    elif event.key == K_PERIOD:
                        if amplitude == 0:
                            amplitude = 0.01
                        else:
                            amplitude *= 1.1

                    elif event.key == K_l:
                        self.render.render_simple = not self.render.render_simple

                    elif event.key == K_p:
                        if not os.path.exists(screenshots_path):
                            os.makedirs(screenshots_path)

                        n_screenshots += 1

                        self.render.save(screenshots_path + str(datetime.now()) + ".png")

                    elif event.key == K_EQUALS:
                        self.render.ray_steps_limit += 10
                    elif event.key == K_MINUS:
                        self.render.ray_steps_limit -= 10

                    elif event.key == K_PAGEUP:
                        fractal_index = (fractal_index + 1) % len(data)
                        fractal_changed = True
                    elif event.key == K_PAGEDOWN:
                        fractal_index = (fractal_index - 1) % len(data)
                        fractal_changed = True

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

                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 4:
                        self.camera.zoom *= 1.1
                    elif event.button == 5:
                        self.camera.zoom *= 1 / 1.1
                    elif event.button == 2:
                        self.camera.zoom = 1.0

            delta = time.time() - time_before_render

            shift = np.array([0, 0, 0], dtype=np.float32)

            key_map = pygame.key.get_pressed()

            if key_map[pygame.K_w]:
                shift += self.camera.direction
            if key_map[pygame.K_s]:
                shift -= self.camera.direction
            if key_map[pygame.K_a]:
                shift -= self.camera.right
            if key_map[pygame.K_d]:
                shift += self.camera.right
            if key_map[pygame.K_SPACE]:
                shift += self.camera.up
            if key_map[pygame.K_LCTRL]:
                shift -= self.camera.up

            shift_l = np.linalg.norm(shift)

            if shift_l > 0:
                shift /= shift_l

            shift *= delta * movement_speed

            self.camera.position += shift
            self.render.epsilon = epsilon / self.camera.zoom
            self.render.fractal = self.render.fractals[fractal_index]

            if fractal_changed:
                movement_speed = data[fractal_index]["movement_speed"]
                epsilon = data[fractal_index]["epsilon"]
                self.camera.position = data[fractal_index]["position"]
                self.camera.look_at(data[fractal_index]["target"])
                self.render.ray_steps_limit = data[fractal_index]["steps"]
                amplitude = data[fractal_index]["amplitude"]
