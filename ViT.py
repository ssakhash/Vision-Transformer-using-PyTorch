import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

class Config:
    img_size = 32
    patch_size = 4
    num_classes = 100
    dim = 128
    depth = 8
    heads = 16
    mlp_dim = 256
    channels = 3
    dropout = 0.1
    weight_decay = 0.01  # L2 regularization

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(dim, forward_expansion * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * dim, dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        return self.dropout(self.norm2(forward + x))

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout):
        super(VisionTransformer, self).__init__()

        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.depth = depth

        self.patch_embedding = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(TransformerBlock(dim, heads, dropout, mlp_dim))

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)

        for block in self.transformer:
            x = block(x, x, x)

        x = self.mlp_head(x[:, 0])
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("GPU: ", torch.cuda.get_device_name(torch.device))

    # Data loading and preprocessing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Data augmentation - random crop
        transforms.RandomHorizontalFlip(),  # Data augmentation - random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    model = VisionTransformer(
        Config.img_size,
        Config.patch_size,
        Config.num_classes,
        Config.dim,
        Config.depth,
        Config.heads,
        Config.mlp_dim,
        Config.channels,
        Config.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=Config.weight_decay)  # Lower learning rate and added weight decay
    scheduler = ReduceLROnPlateau(optimizer, 'min')  # New learning rate scheduler

    # Training loop
    for epoch in range(20):  # Add tqdm for epoch progress
        for i, data in enumerate(tqdm(trainloader, 0)):  # Add tqdm for batch progress within each epoch
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step(loss)  # Updated to use ReduceLROnPlateau scheduler
        print(f"Epoch {epoch + 1}, Loss: {np.round(loss.item(), 3)}")
    print('Finished Training')

if __name__ == "__main__":
    main()
