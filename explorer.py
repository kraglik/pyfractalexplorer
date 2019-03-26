import numpy as np
import pyopencl as cl
import pyopencl.tools
import pyopencl.cltypes
import pyopencl.array
from PIL import Image
from sklearn.preprocessing import normalize


# Создаем контекст приложения, чтобы было вообще на чем вычислять
platform = cl.get_platforms()[0]
device = platform.get_devices(cl.device_type.ALL)[0]

# Собственно контекст
context = cl.Context([device])
# Очередь задач
queue = cl.CommandQueue(context)

# Создаем кастомный тип, который можно было бы использовать и из кода ядра, и из кода хоста
camera_dtype, camera_decl = cl.tools.match_dtype_to_c_struct(
    device,
    "Camera",
    np.dtype([
        ("pos", cl.cltypes.float3),
        ("dir", cl.cltypes.float3),
        ("up", cl.cltypes.float3),
        ("right", cl.cltypes.float3),
        ("view_plane_distance", cl.cltypes.float),
        ("ratio", cl.cltypes.float),
        ("shift_multiplier", cl.cltypes.float),
        ("height", cl.cltypes.int),
        ("width", cl.cltypes.int)
    ])
)

camera_dtype = cl.tools.get_or_register_dtype("Camera", camera_dtype)


# Обертка над этим типом
class Camera:
    def __init__(self,
                 pos=np.array([-10, 0, 0], cl.cltypes.float),
                 dir=np.array([1, 0, 0], cl.cltypes.float),
                 target=None,
                 up=np.array([0, 1, 0], cl.cltypes.float),
                 view_plane_distance=1.0,
                 ratio=1.0,
                 shift_multiplier=0.01,
                 height=1000,
                 width=1000):
        self.pos = pos

        self.dir = dir if target is None else target - pos
        self.dir /= np.linalg.norm(self.dir)
        self.height = height
        self.width = width

        self.up = up / np.linalg.norm(up)

        self.right = np.cross(dir, up)
        self.right /= np.linalg.norm(self.right)

        self.view_plane_distance = view_plane_distance
        self.ratio = ratio
        self.shift_multiplier = shift_multiplier

    @property
    def cl(self):
        camera = np.array([(
            tuple(self.pos) + (0,),
            tuple(self.dir) + (0,),
            tuple(self.up) + (0,),
            tuple(self.right) + (0,),
            self.view_plane_distance,
            self.ratio,
            self.shift_multiplier,
            self.height,
            self.width
        )], camera_dtype)[0]

        return camera


world_props_dtype, world_props_decl = cl.tools.match_dtype_to_c_struct(
    device,
    "WorldProps",
    np.dtype([
        ("shift_value", cl.cltypes.float),
        ("epsilon", cl.cltypes.float),
        ("it_limit", cl.cltypes.int),
        ("r_min", cl.cltypes.float),
        ("escape_time", cl.cltypes.float),
        ("scale", cl.cltypes.float)
    ])
)

world_props_dtype = cl.tools.get_or_register_dtype("WorldProps", world_props_dtype)


class WorldProps:
    def __init__(self,
                 shift_value=0.99,
                 epsilon=0.0001,
                 it_limit=128,
                 r_min=0.5,
                 escape_time=100.0,
                 scale=2.39128):

        self.shift_value = shift_value
        self.epsilon = epsilon
        self.it_limit = it_limit
        self.r_min = r_min
        self.escape_time = escape_time
        self.scale = scale

    @property
    def cl(self):
        props = np.array([(
            self.shift_value,
            self.epsilon,
            self.it_limit,
            self.r_min,
            self.escape_time,
            self.scale
        )], world_props_dtype)[0]

        return props


# Читаем код ядра
with open("kernels/mandelbox.cl", 'r') as f:
    mandelbox_kernel = f.read()


# Создаем программу, вставляя в нее автоматически сгенерированный код для наших кастомных типов
program = cl.Program(context, camera_decl + world_props_decl + mandelbox_kernel).build()

camera = Camera(width=1500, height=1500)
world_props = WorldProps()

# Выдлеяем память под итоговую картинку
image_buffer_host = np.empty((camera.height, camera.width), cl.cltypes.uchar3)

# Выделяем память на GPU
camera_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, camera.cl.nbytes)
world_props_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, world_props.cl.nbytes)
image_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, image_buffer_host.nbytes)

# Ставим копирование данных в очередь задач
cl.enqueue_copy(queue, camera_buffer, camera.cl)
cl.enqueue_copy(queue, world_props_buffer, world_props.cl)

# Запускаем программу
kernel = program.render(
    queue,                      # Очередь задач
    image_buffer_host.shape,    # Форма сетки, на которой будем вычислять
    None,                       # Для локальных индексов, сейчас нинужно
    camera_buffer,              # Первый аргумент ядра -- указатель на структуру с камерой
    world_props_buffer,         # Второй аргумент -- указатель на структуру с параметрами отрисовки
    image_buffer                # Третий аргумент -- массив пикселей, в который будем вести запись
)

# Копируем отрисованную картинку в память хоста (из GPU в CPU)
cl.enqueue_copy(queue, image_buffer_host, image_buffer)

# Поскольку uchar3 не читается Pillow, "слегка" переделываем
raw_image = np.stack([
    image_buffer_host[:]["x"],
    image_buffer_host[:]["y"],
    image_buffer_host[:]["z"]
], axis=2)

# Сохраняем картинку
image = Image.fromarray(raw_image)
image.save("image.png", "png")

