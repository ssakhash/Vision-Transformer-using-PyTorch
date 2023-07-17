import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

class Config:
    img_size = 128  # Smaller image size
    patch_size = 8  # Smaller patch size
    num_classes = 100
    dim = 1024  # Smaller dimension
    depth = 6  # Smaller depth
    heads = 8  # Fewer heads
    mlp_dim = 2048  # Smaller mlp_dim
    channels = 3
    dropout = 0.1
    weight_decay = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, config):
        super(VisionTransformer, self).__init__()

        self.config = config
        self.device = config.device

        # Load a pre-trained ResNet and remove the classification head
        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2]).to(self.device)

        self.num_patches = (config.img_size // config.patch_size) ** 2
        self.patch_embedding = nn.Conv2d(config.dim, config.dim, kernel_size=config.patch_size, stride=config.patch_size, padding=0).to(self.device)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, config.dim)).to(self.device)
        self.class_token = nn.Parameter(torch.randn(1, 1, config.dim)).to(self.device)

        self.dropout = nn.Dropout(config.dropout).to(self.device)

        self.transformer = nn.ModuleList([])
        for _ in range(config.depth):
            self.transformer.append(TransformerBlock(config.dim, config.heads, config.dropout, config.mlp_dim).to(self.device))

        self.mlp_head = nn.Linear(config.dim, config.num_classes).to(self.device)

    def forward(self, x):
        x = self.cnn(x)  # Pass input through CNN feature extractor
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)  # Flatten CNN output and apply patch embedding
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
    device = Config.device
    print("GPU: ", torch.cuda.get_device_name(device))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(Config.img_size),  # Resizing for smaller image size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)  # Smaller batch size

    model = VisionTransformer(Config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=Config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    accum_gradient = 4  # Gradient accumulation steps

    for epoch in range(20):
        epoch_loss = 0
        for i, data in enumerate(tqdm(trainloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Gradient accumulation
            loss = loss / accum_gradient
            loss.backward()

            if (i+1) % accum_gradient == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        scheduler.step(epoch_loss / len(trainloader))
        print(f"Epoch {epoch + 1}, Loss: {np.round(epoch_loss / len(trainloader), 3)}")
    print('Finished Training')

if __name__ == "__main__":
    main()
