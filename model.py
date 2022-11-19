import os
import torch

class PrintSize(torch.nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()
    def forward(self, x):
        print(x.shape)

class SudokuNet(torch.nn.Module):
    def __init__(self):
        super(SudokuNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(int(32*28*28/4),2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)
    
    def inference(self, x):
        output = self.net(x)
        return output.max(1)[1]

    def save_weights(self, path):
        if os.path.dirname(path) != "":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device}!")
    model = SudokuNet()
    model.to(device)
    print(model)
    fake_input = torch.randn((2,1,28,28))
    output = model(fake_input.to(device))
    print(output)
    print(model.inference(fake_input.to(device)))
    model.load_weights("weights/sudokunet.pt")
    output = model(fake_input.to(device))
    print(output)
    print(model.inference(fake_input.to(device)))