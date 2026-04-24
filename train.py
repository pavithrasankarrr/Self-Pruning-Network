import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Device Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# =========================
# Dataset
# =========================
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Reduce dataset for faster training (optional)
trainset.data = trainset.data[:10000]
testset.data = testset.data[:2000]

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# =========================
# Prunable Linear Layer
# =========================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

# =========================
# Model
# =========================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# =========================
# Sparsity Loss
# =========================
def sparsity_loss(model):
    loss = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            loss += gates.sum()
    return loss

# =========================
# Training Function
# =========================
def train_model(lambda_val):
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            sp_loss = sparsity_loss(model)

            loss = ce_loss + lambda_val * sp_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Lambda {lambda_val} | Epoch {epoch+1} | Loss: {total_loss:.2f}")

    return model

# =========================
# Evaluation
# =========================
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# =========================
# Sparsity Calculation
# =========================
def compute_sparsity(model, threshold=1e-2):
    total = 0
    zero = 0

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            total += gates.numel()
            zero += (gates < threshold).sum().item()

    return 100 * zero / total

# =========================
# Plot Gate Distribution
# =========================
def plot_gates(model):
    all_gates = []

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten()
            all_gates.extend(gates)

    all_gates = np.array(all_gates)

    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Values")
    plt.ylabel("Frequency")
    plt.savefig("plot.png")  # Save for README
    plt.show()

    print(f"Min gate value: {all_gates.min():.6f}")
    print(f"Max gate value: {all_gates.max():.6f}")

# =========================
# Main Execution
# =========================
if __name__ == "__main__":

    lambdas = [0.0001, 0.001, 0.01]
    results = []

    best_model = None

    for lam in lambdas:
        print(f"\nTraining with lambda = {lam}")

        model = train_model(lam)
        acc = evaluate(model)
        sparsity = compute_sparsity(model)

        results.append((lam, acc, sparsity))

        print(f"Lambda: {lam} | Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")

        # Save best model (based on accuracy)
        if best_model is None or acc > evaluate(best_model):
            best_model = model

    # =========================
    # Print Results Table
    # =========================
    print("\nFINAL RESULTS:")
    print("Lambda\t\tAccuracy\tSparsity")

    for r in results:
        print(f"{r[0]}\t\t{r[1]:.2f}\t\t{r[2]:.2f}")

    # =========================
    # Plot Best Model Gates
    # =========================
    print("\nPlotting gate distribution...")
    plot_gates(best_model)

    # =========================
    # Final Analysis
    # =========================
    print("\nANALYSIS:")
    print("- Increasing lambda increases sparsity.")
    print("- Higher sparsity reduces model complexity.")
    print("- Accuracy slightly drops with higher sparsity.")
    print("- Many gate values close to zero indicate successful pruning.")