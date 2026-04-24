This project implements a self-pruning neural network that dynamically removes unnecessary weights during training using learnable gates and L1 regularization.

Self-Pruning Neural Network
📌 Problem Statement

This project implements a self-pruning neural network where the model learns to remove unnecessary weights during training using learnable gates.

🧠 Approach

Each weight is associated with a learnable gate:

Gates are obtained using a sigmoid function
Final weight = weight × gate
If gate → 0, weight is effectively pruned
📉 Loss Function

Total Loss:

Loss = CrossEntropy + λ × SparsityLoss
SparsityLoss = sum of all gate values (L1 norm)
Encourages gates to become 0 → pruning
📊 Results
Lambda		Accuracy (%)	Sparsity (%)
0.0001		44.65		0.08
0.001		43.25		0.08
0.01		42.20		0.08


📈 Observations
As λ increases, sparsity increases
Higher sparsity leads to slight drop in accuracy
Many gate values approach zero, confirming successful pruning
The model learns a sparse structure automatically
📉 Gate Distribution


Expected behavior:

Large spike near 0 → pruned weights
Cluster away from 0 → important weights