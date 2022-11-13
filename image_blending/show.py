from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


SIZE_CONSTANT = 10000  # константа, превышающий размер изображений
COLORS = [(250, 50, 50),
          (50, 250, 50),
          (50, 50, 250),
          (250, 250, 50),
          (250, 50, 250),
          (50, 250, 250),
          (20, 150, 250),
          (150, 20, 250),
          (250, 20, 150),
          (250, 150, 20),
          (20, 250, 150),
          (150, 250, 20)]


def _read_image_if_need(*args) -> List[np.ndarray]:
    """
    Получаем список путей или изображений и при необходимости читает их

    :param args: List[Union[str, np.ndarray]]: список из путей и изображений
    :return: List[np.ndarray]: список изображений
    """
    images = []
    for image in args:
        if isinstance(image, str):
            image = cv2.imread(image)
        images.append(image)
    return images


def get_colors(n_colors: int) -> List[Tuple[int, int, int]]:
    if n_colors <= len(COLORS):
        return COLORS[:n_colors]
    else:
        random_colors = np.random.randint(0, 255, (n_colors - len(COLORS), 3)).tolist()
    return COLORS + random_colors


def show_image(
    image: Union[str, np.ndarray],
    figsize: Tuple[int, int] = (20, 20),
    title: Optional[str] = None,
    cmap: Optional[str] = None,
    axis: bool = False,
) -> None:
    """
    Показывает изображение

    :param image: Union[str, np.ndarray]: изображение или путь к изображению
    :param figsize: Tuple[int, int]: размер изобаржения
    :param title: Optional[str]: подпись к изображению
    :param cmap: Optional[str]: название цветовой палитры, например ``Greys_r``
    :param axis: bool: нужно ли рисовать оси
    """
    image = _read_image_if_need(image)[0]
    plt.figure(figsize=figsize)
    plt.title(title)
    if not axis:
        plt.axis("off")
    plt.imshow(image, cmap)
    plt.show()


def subplot_images(
    images: List[Union[str, np.ndarray]],
    n_rows: int,
    n_columns: int,
    figsize: Tuple[int, int] = (20, 20),
    titles: Optional[Union[str, List[str]]] = None,
    cmap: Optional[str] = None,
    axis: bool = False,
    layout_pad: Tuple[float, float] = (1, 1),
) -> None:
    """
    Показывает список изображений в виде таблицы n_rows * n_columns

    :param images: List[Union[str, np.ndarray]]: список изображений или путей к изобаржениям
    :param n_rows: int: количество строк
    :param n_columns: int: количество столбцов
    :param figsize: Tuple[int, int]: размер изобаржения
    :param titles: Optional[Union[str, List[str]]]: подписи к изображений, если подпись одна, то
    она будет дублироваться для каждого изображения
    :param cmap: Optional[str]: название цветовой палитры, например ``Greys_r``
    :param axis: bool: нужно ли рисовать оси
    :param layout_pad: Tuple[float, float]: padding (width/height) между изображениями
    """
    assert len(images) <= n_rows * n_columns
    if titles is None or isinstance(titles, str):
        titles = [titles] * len(images)
    plt.figure(figsize=figsize)
    images = _read_image_if_need(*images)
    for i, image in enumerate(images, start=1):
        plt.subplot(n_rows, n_columns, i, ymargin=0.5, xmargin=0.5)
        plt.title(titles[i - 1])
        if not axis:
            plt.axis("off")
        plt.tight_layout(w_pad=layout_pad[0], h_pad=layout_pad[1])
        plt.imshow(image, cmap)
    plt.show()


def draw_points(
    image: Union[str, np.ndarray],
    points: List[Tuple[float, float]],
    figsize=(20, 30),
    diameter=10,
    color: Union[str, Tuple[int, int, int]] = (50, 250, 50),
    inline: bool = True,
    cmap: Optional[str] = None,
    axis: bool = False,
) -> Optional[np.ndarray]:
    """
    Рисует точки на изображении

    :param image: Union[str, np.ndarray]: изображение или путь к изображению
    :param points: List[Tuple[float, float]]: список точек в формате (x, y)
    :param figsize: Tuple[int, int]: размер изображения
    :param diameter: int: диаметр точек
    :param color: Union[str, Tuple[int, int, int]]: цвет точек, можно задать в ручную в виде
    (int, int, int), если не задано, то в этом случае все точки будут иметь
    различные случайные цвета
    :param inline: bool: если True, изображение будет показано, иначе будет возвращено
    :param cmap: Optional[str]: название цветовой палитры, например ``Greys_r``
    :param axis: bool: нужно ли рисовать оси
    :return: Optional[np.ndarray]: Если inline is False, функция вернет изображение с
    нарисованными точками
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    image = image.copy()
    current_color = color  # если цвет определен, но current_color равен ему и не поменяется
    if color is None:  # если цвет не определен, то цвета будут взяты из палитры цветов
        colors = get_colors(len(points))
    for idx, (x, y) in enumerate(points):
        if color is None:
            current_color = colors[idx]
        cv2.circle(image, (int(x), int(y)), diameter, current_color, -1)
    if inline:
        plt.figure(figsize=figsize)
        if not axis:
            plt.axis("off")
        plt.imshow(image, cmap=cmap)
        plt.show()
        return None
    return image
