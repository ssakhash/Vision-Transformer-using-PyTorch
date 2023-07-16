import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class Config:
    img_size = 32
    patch_size = 4
    num_classes = 100
    dim = 64
    depth = 6
    heads = 8
    mlp_dim = 128
    channels = 3
    dropout = 0.1

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

        self.image_size = image_size
        self.patch_size = patch_size
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

    # Data loading and preprocessing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    model = VisionTransformer(Config.img_size, Config.patch_size, Config.num_classes, Config.dim, Config.depth, Config.heads, Config.mlp_dim, Config.channels, Config.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10): 
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    print('Finished Training')

if __name__ == "__main__":
    main()
