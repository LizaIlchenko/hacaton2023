import cv2
import pandas as pd

# Путь к видеофайлу
video_path = "vidosic.mp4"

# Создание объекта VideoCapture для чтения видеофайла
video_capture = cv2.VideoCapture(video_path)

# Создание окна с видеоплеером
cv2.namedWindow("Video")

# Создание пустого списка для хранения меток
labels = []

# Функция для обработки кликов мыши на видео
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Получение временной метки в миллисекундах
        timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC)
        # Добавление метки в список
        labels.append((timestamp, x, y))

# Установка обратного вызова мыши
cv2.setMouseCallback("Video", mouse_callback)

# Чтение и отображение видеокадров
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Отображение меток на кадре
    for label in labels:
        timestamp, x, y = label
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    # Отображение текущего кадра
    cv2.imshow("Video", frame)
    
    # Обработка нажатия клавиши 'q' для выхода из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрытие окна с видеоплеером
cv2.destroyAllWindows()

# Создание DataFrame для хранения меток
df = pd.DataFrame(labels, columns=["timestamp", "x", "y"])

# Сохранение DataFrame в CSV-файл
df.to_csv("labels.csv", index=False)
