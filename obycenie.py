import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import YOLOv3  # Подключаем модель YOLOv3
from dataset import CustomDataset  # Подключаем пользовательский датасет

# Параметры обучения
batch_size = 16
epochs = 100
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пути к директориям с данными
train_images_dir = "path_to_train_images_directory"
train_annotations_dir = "path_to_train_annotations_directory"
val_images_dir = "path_to_validation_images_directory"
val_annotations_dir = "path_to_validation_annotations_directory"

# Загрузка датасета обучения и валидации
train_dataset = CustomDataset(train_images_dir, train_annotations_dir)
val_dataset = CustomDataset(val_images_dir, val_annotations_dir)

# Определение DataLoader для обучения и валидации
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Инициализация модели YOLOv3
model = YOLOv3(num_classes=4).to(device)

# Определение функции потерь
criterion = nn.MSELoss()

# Определение оптимизатора
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Цикл обучения
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Прямой проход через модель
        outputs = model(images)
        
        # Расчет функции потерь
        loss = criterion(outputs, targets)
        train_loss += loss.item()
        
        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()
    
    # Вывод потерь обучения
    print("Epoch [{}/{}], Train Loss: {:.4f}".format(epoch+1, epochs, train_loss/len(train_loader)))
    
    # Валидация модели
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Прямой проход через модель
            outputs = model(images)
            
            # Расчет функции потерь
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    # Вывод потерь валидации
    print("Epoch [{}/{}], Val Loss: {:.4f}".format(epoch+1, epochs, val_loss/len(val_loader)))

# Сохранение обученной модели
torch.save(model.state_dict(), "yolov3_model.pth")
