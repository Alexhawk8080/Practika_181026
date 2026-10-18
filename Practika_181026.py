# Practika_181026.py
"""Практический сценарий: обучение CNN на MNIST и прогон по пользовательским картинкам.

Внешние зависимости: tensorflow/keras, numpy, PIL, matplotlib.

Скрипт консольный (без GUI) и работает в два этапа:

  Этап 1. Создание и обучение модели — инлайненный бэкенд
          (TrainSettings -> build_model -> train_model на реальном MNIST).
          После load_mnist рисуется сетка примеров данных, после обучения —
          кривые обучения (loss/accuracy по эпохам).
  Этап 2. Прогон обученной модели по тестовым картинкам из папки digit_test/
          (сырые PNG, нарисованные мышкой, светлый фон / тёмный штрих),
          с покадровым выводом предсказаний и итоговой сводкой, а также
          набором matplotlib-графиков.

Все графики сохраняются как PNG в папку plots_181026/ рядом со скриптом
(бэкенд matplotlib — Agg, plt.show() не используется: устойчивая работа
без дисплея). В конце выводится список сохранённых файлов.

Точка входа: main().
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # надёжный бэкенд без дисплея
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import keras  # noqa: E402
from keras import layers  # noqa: E402

# Папка с пользовательскими тестовыми картинками (относительно корня проекта)
TEST_DIR = Path(__file__).resolve().parent / "digit_test"
# Папка для сохранения графиков
PLOTS_DIR = Path(__file__).resolve().parent / "plots_181026"
# Поддерживаемое расширение (регистр не важен)
ALLOWED_EXT = {".png"}

# Список сохранённых графиков (заполняется по ходу работы)
SAVED_PLOTS: list = []


OUT_SIZE = 28
INNER_SIZE = 20
GRAY_THRESHOLD = 128


def _to_gray_array(image):
    """Приводит вход к 2D numpy uint8 array (H, W)."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("L"))
    else:
        image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"ожидается 2D массив, получено shape={image.shape}"
        )
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _resize(arr, size_hw):
    """Ресайз 2D uint8 массива через PIL (LANCZOS). size_hw = (h, w)."""
    img = Image.fromarray(arr, mode="L")
    return np.asarray(img.resize((size_hw[1], size_hw[0]), Image.LANCZOS))


def preprocess_mnist_like(image):
    """Режим B: как в MNIST — bbox, масштаб, центровка по центру масс.

    Вход: PIL.Image "L" или 2D array 0..255 (светлый фон, тёмный штрих).
    Выход: numpy array (28, 28) float32, значения 0..1
    (низкий фон, высокий штрих — формат MNIST).
    """
    arr = _to_gray_array(image)
    strokes = arr < GRAY_THRESHOLD

    if not strokes.any():
        return np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float32)

    ys, xs = np.where(strokes)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = arr[y0:y1, x0:x1]

    h, w = cropped.shape
    scale = INNER_SIZE / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = _resize(cropped, (new_h, new_w))

    inverted = (255 - resized).astype(np.float64)

    total = inverted.sum()
    if total == 0:
        return np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float32)

    ys_i, xs_i = np.indices(inverted.shape)
    cy = float((ys_i * inverted).sum() / total)
    cx = float((xs_i * inverted).sum() / total)

    target = (OUT_SIZE - 1) / 2.0
    top = int(round(target - cy))
    left = int(round(target - cx))

    canvas = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float64)

    dest_y0 = max(0, top)
    dest_x0 = max(0, left)
    dest_y1 = min(OUT_SIZE, top + new_h)
    dest_x1 = min(OUT_SIZE, left + new_w)

    src_y0 = dest_y0 - top
    src_x0 = dest_x0 - left
    src_y1 = src_y0 + (dest_y1 - dest_y0)
    src_x1 = src_x0 + (dest_x1 - dest_x0)

    canvas[dest_y0:dest_y1, dest_x0:dest_x1] = (
        inverted[src_y0:src_y1, src_x0:src_x1]
    )

    return canvas.astype(np.float32) / 255.0


OPTIMIZERS = ("adam", "sgd", "rmsprop")


@dataclass
class TrainSettings:
   

    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    optimizer: str = "adam"
    conv_filters: list = field(default_factory=lambda: [28, 56, 56])
    dense_units: int = 56

    def __post_init__(self):
        if self.optimizer.lower() not in OPTIMIZERS:
            raise ValueError(
                f"оптимизатор {self.optimizer!r} не поддерживается; "
                f"выбери из {OPTIMIZERS}"
            )
        if len(self.conv_filters) != 3:
            raise ValueError(
                f"conv_filters должен содержать ровно 3 значения, "
                f"получено {len(self.conv_filters)}"
            )
        if any(int(f) < 1 for f in self.conv_filters):
            raise ValueError("все conv_filters должны быть >= 1")
        if self.epochs < 1:
            raise ValueError("epochs >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size >= 1")
        if self.dense_units < 1:
            raise ValueError("dense_units >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate > 0")


@dataclass
class TrainResult:
    """Результат обучения."""

    model: object
    history: dict
    test_loss: float
    test_accuracy: float


def _make_optimizer(name: str, lr: float):
    """Создаёт оптимизатор по имени и learning rate."""
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError(f"неизвестный оптимизатор: {name}")


def build_model(settings: TrainSettings,
                input_shape=(28, 28, 1),
                num_classes: int = 10):
    """Собирает и компилирует модель по настройкам.

    Архитектура: 3 Conv2D+MaxPool блока, Flatten, Dense, выход softmax.
    """
    f1, f2, f3 = (int(x) for x in settings.conv_filters)
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(f1, (4, 4), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(f2, (4, 4), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(f3, (4, 4), activation="relu"),
        layers.Flatten(),
        layers.Dense(int(settings.dense_units), activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=_make_optimizer(settings.optimizer, settings.learning_rate),
        loss="sparse_categorical_crossentropy",  # from_logits=False
        metrics=["accuracy"],
    )
    return model


class _ProgressCallback(keras.callbacks.Callback):
    """Пробрасывает наружу on_epoch_end и проверяет should_stop."""

    def __init__(self, on_epoch_end=None, should_stop=None):
        super().__init__()
        self._on_epoch_end = on_epoch_end
        self._should_stop = should_stop

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self._on_epoch_end is not None:
            self._on_epoch_end(epoch, dict(logs))
        if self._should_stop is not None and self._should_stop():
            self.model.stop_training = True


def load_mnist():
    """MNIST с осью канала и нормализацией в 0..1.

    Возвращает ((x_train, y_train), (x_test, y_test)):
      x_*: float32 (N, 28, 28, 1), значения 0..1
      y_*: uint8 (N,)
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train[..., None].astype("float32")) / 255.0
    x_test = (x_test[..., None].astype("float32")) / 255.0
    return (x_train, y_train), (x_test, y_test)


def train_model(settings: TrainSettings, *,
                train_data=None, test_data=None,
                on_epoch_end=None, should_stop=None,
                verbose=0) -> TrainResult:
    """Обучает модель.

    train_data/test_data — кортежи (x, y). Если не переданы — грузится MNIST.
    on_epoch_end(epoch, logs) — вызывается после каждой эпохи (epoch 0-based).
    should_stop() -> bool — если True, обучение останавливается после текущей.

    Возвращает TrainResult.
    """
    if train_data is None or test_data is None:
        (x_train, y_train), (x_test, y_test) = load_mnist()
    else:
        (x_train, y_train) = train_data
        (x_test, y_test) = test_data

    model = build_model(settings)
    cb = _ProgressCallback(on_epoch_end=on_epoch_end, should_stop=should_stop)
    history = model.fit(
        x_train, y_train,
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        validation_data=(x_test, y_test),
        callbacks=[cb],
        verbose=verbose,
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return TrainResult(
        model=model,
        history={k: list(v) for k, v in history.history.items()},
        test_loss=float(loss),
        test_accuracy=float(acc),
    )



def _save_fig(fig, name: str) -> None:
    """Сохраняет фигуру в plots_181026/ и закрывает её.

    Args:
        fig: matplotlib.figure.Figure.
        name: имя файла (например "training_curves.png").
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    SAVED_PLOTS.append(out)
    print(f"  график сохранён: {out.name}")


def plot_training_curves(history: dict, settings: TrainSettings) -> None:
    """Кривые обучения: loss и accuracy по эпохам (train + validation)."""
    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["loss"], "o-", label="train loss")
    ax1.plot(epochs, history["val_loss"], "s--", label="val loss")
    ax1.set_title("Loss по эпохам")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.4)
    ax1.legend()

    ax2.plot(epochs, history["accuracy"], "o-", label="train accuracy")
    ax2.plot(epochs, history["val_accuracy"], "s--", label="val accuracy")
    ax2.set_title("Accuracy по эпохам")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.4)
    ax2.legend()

    fig.suptitle(
        f"Обучение CNN на MNIST "
        f"(epochs={settings.epochs}, batch={settings.batch_size}, "
        f"opt={settings.optimizer})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_fig(fig, "training_curves.png")


def plot_mnist_samples(x_train, y_train, n_rows: int = 3, n_cols: int = 4) -> None:
    """Сетка примеров MNIST с подписями классов (после load_mnist)."""
    n = n_rows * n_cols
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_train), size=n, replace=False)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(x_train[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"класс: {int(y_train[i])}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Примеры обучающих данных MNIST ({n} шт.)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, "mnist_samples.png")


def plot_predictions_grid(samples, probs, preds) -> None:
    """Сетка «сырая картинка + 28x28-препроцессинг» с подписями истина→предсказание.

    Args:
        samples: список кортежей (path, true_label, raw_arr, prep_arr),
                 до 12 штук (ошибки — первыми).
        probs: массив вероятностей (N, 10) только для выбранных samples.
        preds: массив предсказаний (N,) только для выбранных samples.
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(4.2, n * 1.9),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    if n == 1:
        axes = axes[None, :]

    for row, ((path, true_label, raw_arr, prep_arr), p, pred) in \
            enumerate(zip(samples, probs, preds)):
        ok = int(pred) == true_label
        mark = "\u2713" if ok else "\u2717"  # ✓ / ✗
        color = "tab:green" if ok else "tab:red"

        ax_raw = axes[row, 0]
        ax_raw.imshow(raw_arr, cmap="gray")
        ax_raw.set_ylabel(f"«{path.name}»", fontsize=7, rotation=0,
                          ha="right", va="center")
        ax_raw.set_title(f"сырая {raw_arr.shape[0]}×{raw_arr.shape[1]}", fontsize=8)
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])

        ax_prep = axes[row, 1]
        ax_prep.imshow(prep_arr, cmap="gray", vmin=0, vmax=1)
        ax_prep.set_title("28×28 препроцессинг", fontsize=8)
        ax_prep.set_xticks([])
        ax_prep.set_yticks([])
        ax_prep.set_xlabel(
            f"истина={true_label} → предсказание={int(pred)} {mark}",
            fontsize=9, color=color,
        )

    fig.suptitle(
        "Предсказания по digit_test (слева сырые картинки, справа препроцессинг)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_fig(fig, "predictions_grid.png")


def plot_confusion_matrix(confusion: np.ndarray) -> None:
    """Heatmap матрицы ошибок 10×10 с аннотациями."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(confusion, cmap="Blues")

    thresh = confusion.max() / 2.0 if confusion.max() > 0 else 0.5
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(int(confusion[i, j])),
                    ha="center", va="center", fontsize=9,
                    color="white" if confusion[i, j] > thresh else "black")

    ax.set_title("Матрица ошибок по digit_test (строки = истина, столбцы = предсказание)",
                 fontsize=11)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.colorbar(im, ax=ax, label="Количество")
    fig.tight_layout()
    _save_fig(fig, "confusion_matrix.png")


def plot_per_class_accuracy(confusion: np.ndarray) -> None:
    """Bar-chart точности по классам (%) для digit_test."""
    accs, counts = [], []
    for cls in range(10):
        total = int(confusion[cls].sum())
        correct = int(confusion[cls, cls])
        accs.append(correct / total * 100.0 if total else np.nan)
        counts.append(total)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([str(c) for c in range(10)], accs,
                  color="tab:blue", edgecolor="black")
    for bar, acc, cnt in zip(bars, accs, counts):
        if np.isnan(acc):
            bar.set_alpha(0.15)
            bar.set_hatch("//")
            label = "нет данных"
        else:
            label = f"{acc:.1f}%\n(n={cnt})"
        ax.text(bar.get_x() + bar.get_width() / 2.0, 2.0, label,
                ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.set_title("Точность по классам (digit_test)", fontsize=12)
    ax.set_xlabel("Класс")
    ax.set_ylabel("Accuracy, %")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    _save_fig(fig, "per_class_accuracy.png")


def plot_error_probability(path: Path, true_label: int, pred: int,
                           probs_row: np.ndarray, index: int) -> None:
    """Гистограмма топ-10 вероятностей классов для ошибочного предсказания."""
    order = np.argsort(probs_row)[::-1][:10]
    values = probs_row[order]

    colors = []
    for d in order:
        if d == true_label:
            colors.append("tab:red")      # истинный класс
        elif d == pred:
            colors.append("tab:green")    # предсказанный (ошибочный)
        else:
            colors.append("lightgray")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(int(d)) for d in order], values, color=colors, edgecolor="black")
    for x, v in enumerate(values):
        ax.text(x, v + 0.01, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 1.12)
    ax.set_title(
        f"Ошибка #{index + 1}: {path.name}\n"
        f"истина={true_label} (красный) → предсказано={pred} (зелёный)",
        fontsize=11,
    )
    ax.set_xlabel("Класс")
    ax.set_ylabel("Вероятность")
    ax.grid(True, axis="y", alpha=0.4)

    # Легенда-пояснение
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="tab:red", edgecolor="black", label="истинный класс"),
        Patch(facecolor="tab:green", edgecolor="black", label="предсказанный класс"),
        Patch(facecolor="lightgray", edgecolor="black", label="остальные"),
    ], fontsize=8, loc="upper right")

    fig.tight_layout()
    _save_fig(fig, f"error_{index + 1:02d}_true{true_label}_pred{pred}.png")


# =====================================================================
# Основной сценарий
# =====================================================================

def _header(title: str) -> None:
    """Печатает заголовок-разделитель секции."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def collect_test_images(test_dir: Path) -> list:
    """Рекурсивно собирает PNG-картинки из digit_test.

    Подпапка интерпретируется как истинная метка (имя = цифра 0..9).
    Поддерживаются только .png (регистронезависимо).

    Args:
        test_dir: путь к папке digit_test.

    Returns:
        Список кортежей (путь: Path, истинная метка: int).
    """
    items = []
    for path in sorted(test_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_EXT:
            continue
        try:
            label = int(path.parent.name)
        except ValueError:
            # Подпапка не является цифрой — пропускаем
            print(f"  пропуск (подпапка не цифра): {path}")
            continue
        if not 0 <= label <= 9:
            print(f"  пропуск (метка вне 0..9): {path}")
            continue
        items.append((path, label))
    return items


def run_training() -> "tuple[object, TrainSettings, TrainResult]":
    """Этап 1: сборка и обучение модели на MNIST.

    Параметры обучения зафиксированы (соответствуют сохранённой модели
    приложения): epochs=10, batch_size=32, lr=0.001, adam,
    conv_filters=[28, 56, 56], dense_units=56.

    Returns:
        (обученная модель, использованные настройки, результат обучения)
    """
    _header("Секция 1. Создание и обучение модели")

    settings = TrainSettings(
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        optimizer="adam",
        conv_filters=[28, 56, 56],
        dense_units=56,
    )
    print("Параметры обучения:")
    print(f"  epochs={settings.epochs}, batch_size={settings.batch_size}, "
          f"learning_rate={settings.learning_rate}, optimizer={settings.optimizer}")
    print(f"  conv_filters={settings.conv_filters}, dense_units={settings.dense_units}")

    print()
    print("Загрузка MNIST...")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f"  train: {x_train.shape[0]} примеров, "
          f"test: {x_test.shape[0]} примеров, shape={x_train.shape[1:]}")

    # График: примеры данных MNIST (после load_mnist)
    plot_mnist_samples(x_train, y_train, n_rows=3, n_cols=4)

    model = build_model(settings)
    print()
    print("Архитектура модели:")
    model.summary()

    print()
    print("Обучение на реальном MNIST (10 эпох)...")

    def on_epoch_end(epoch: int, logs: dict) -> None:
        """Вывод прогресса после каждой эпохи."""
        loss = logs.get("loss")
        acc = logs.get("accuracy")
        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_accuracy")
        print(f"  Эпоха {epoch + 1}/{settings.epochs}: "
              f"loss={loss:.4f}, acc={acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    result = train_model(
        settings,
        train_data=(x_train, y_train),
        test_data=(x_test, y_test),
        on_epoch_end=on_epoch_end,
        verbose=0,
    )

    # График: кривые обучения
    plot_training_curves(result.history, settings)

    print()
    print(f"Финальные метрики на тестовой выборке MNIST:")
    print(f"  test_loss     = {result.test_loss:.4f}")
    print(f"  test_accuracy = {result.test_accuracy:.4f} "
          f"({result.test_accuracy * 100:.2f}%)")
    return result.model, settings, result


def run_prediction(model) -> None:
    """Этап 2: прогон модели по картинкам из digit_test.

    Каждый PNG: загрузка PIL (grayscale "L"), препроцессинг
    preprocess_mnist_like (режим B, как в приложении по умолчанию),
    batch-предсказание. Вывод покадровый + итоговая сводка с матрицей
    ошибок + набор matplotlib-графиков (сетка предсказаний, матрица
    ошибок, точность по классам, гистограммы вероятностей ошибок).

    Args:
        model: обученная Keras-модель.
    """
    _header("Секция 2. Прогон по тестовым картинкам digit_test")

    if not TEST_DIR.is_dir():
        print(f"Папка {TEST_DIR} отсутствует — секция 2 пропущена.")
        return
    items = collect_test_images(TEST_DIR)
    if not items:
        print(f"В папке {TEST_DIR} нет подходящих PNG — секция 2 пропущена.")
        return

    # --- Загрузка и препроцессинг всех картинок ---
    arrays = []
    loaded = []  # (путь, истина, сырой массив) только для успешно загруженных
    for path, label in items:
        try:
            img = Image.open(path).convert("L")
            raw_arr = np.asarray(img)
            arr = preprocess_mnist_like(img)
        except Exception as exc:
            print(f"  ошибка обработки {path.name}: {exc}")
            continue
        arrays.append(arr)
        loaded.append((path, label, raw_arr))

    if not loaded:
        print("Ни одна картинка не обработана — секция 2 пропущена.")
        return

    # Batch-предсказание одним массивом
    batch = np.stack(arrays)[..., None].astype("float32")  # (N, 28, 28, 1)
    probs = model.predict(batch, verbose=0)
    preds = probs.argmax(axis=1)

    # --- Покадровый вывод ---
    print(f"Всего картинок: {len(loaded)}")
    print()
    errors = []  # (имя, истина, предсказание)
    confusion = np.zeros((10, 10), dtype=int)  # [истина][предсказание]

    for (path, true_label, _raw), pred, p in zip(loaded, preds, probs):
        ok = int(pred) == true_label
        top3 = np.argsort(p)[::-1][:3]
        top3_str = ", ".join(f"{d}:{p[d] * 100:.1f}%" for d in top3)
        mark = "ПРАВИЛЬНО" if ok else "НЕПРАВИЛЬНО"
        print(f"{path.name}: истина={true_label}, предсказано={pred} "
              f"[{mark}] | топ-3: {top3_str}")
        confusion[true_label, int(pred)] += 1
        if not ok:
            errors.append((path, true_label, int(pred)))

    # --- Итоговая сводка ---
    total = len(loaded)
    correct = total - len(errors)
    accuracy = correct / total * 100.0

    print()
    print("-" * 70)
    print("Итоговая сводка")
    print("-" * 70)
    print(f"Всего картинок : {total}")
    print(f"Правильных     : {correct}")
    print(f"Точность       : {accuracy:.2f}%")
    print()
    print("Компактная таблица по классам (класс: угадано/всего):")
    for cls in range(10):
        row_total = int(confusion[cls].sum())
        if row_total == 0:
            continue
        row_correct = int(confusion[cls, cls])
        status = "OK" if row_correct == row_total else "!"
        print(f"  {cls}: {row_correct}/{row_total}  {status}")
    print()
    print("Матрица ошибок 10x10 (строки = истина, столбцы = предсказание):")
    header = "     " + " ".join(f"{d:>4d}" for d in range(10))
    print(header)
    for cls in range(10):
        row = " ".join(f"{confusion[cls, d]:>4d}" for d in range(10))
        print(f"  {cls}  {row}")
    print()
    if errors:
        print(f"Список ошибок ({len(errors)}):")
        for path, true_label, pred in errors:
            print(f"  {path} : {true_label} -> {pred}")
    else:
        print("Ошибок нет — все картинки распознаны правильно.")

    # --- matplotlib-графики по digit_test ---
    print()
    print("Сохранение графиков по digit_test...")

    # 1. Сетка предсказаний: ошибки первыми, добираем корректными до 12
    error_positions = [i for i in range(len(loaded))
                       if int(preds[i]) != loaded[i][1]]
    correct_positions = [i for i in range(len(loaded))
                         if int(preds[i]) == loaded[i][1]]
    grid_positions = error_positions + correct_positions
    grid_positions = grid_positions[:12]

    samples = [
        (loaded[i][0], loaded[i][1], loaded[i][2], arrays[i])
        for i in grid_positions
    ]
    plot_predictions_grid(samples, probs[grid_positions], preds[grid_positions])

    # 2. Матрица ошибок
    plot_confusion_matrix(confusion)

    # 3. Точность по классам
    plot_per_class_accuracy(confusion)

    # 4. Гистограммы вероятностей для ошибок (до 12 графиков)
    if errors:
        print(f"Гистограммы вероятностей для {len(errors)} ошибок "
              f"(истинный класс выделен красным):")
        for idx, (path, true_label, pred) in enumerate(errors[:12]):
            plot_error_probability(path, true_label, pred,
                                   probs[idx], idx)
    else:
        print("Ошибок нет — гистограммы ошибок не строятся.")


def main() -> None:
    """Точка входа: обучение модели, затем прогон по digit_test."""
    # Этап 1: обучение выполняется всегда
    model, _settings, result = run_training()

    # Этап 2: прогон по пользовательским картинкам (с устойчивостью к пустой папке)
    try:
        run_prediction(model)
    except Exception as exc:
        print(f"Ошибка при прогоне по digit_test: {exc}")

    # Итоговая сводка по графикам
    _header("Сохранённые графики")
    if SAVED_PLOTS:
        print(f"Папка: {PLOTS_DIR}")
        for p in SAVED_PLOTS:
            print(f"  {p.name}")
        print(f"Всего файлов: {len(SAVED_PLOTS)}")
    else:
        print("Графики не сохранялись.")

    print()
    print(f"Финальная точность на MNIST test: "
          f"{result.test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
